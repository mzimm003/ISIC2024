..    include:: <isonum.txt>

Models
========

While the competition proceeds, I will attempt several iterations of a
classifier model. These will be a combination of my own ideas, research, and 
ideas from the community Kaggle provides.

Preprocessing
---------------

    For each model I provide preprocessing options. Some preprocessing is
    strictly necessary, but some make work in combination with certain models
    better than others. How the preprocessing options are used will be specified
    for each model. For the ISIC dataset, two major components exist. For each
    skin lesion their is an image and a set of recorded features about the case.
    Both must be preprocessed, but are done so distinctively.

Images
^^^^^^^^

    .. container:: twocol

        .. container:: leftside

            **Rescaling Color** - Color in images is represented by 3 channels per
            pixel. One each for red, green, and blue. The intensity of the color in
            their respective channel is described on a range from 0 to 255. Machine
            learning models often do better on smaller scales so it is important to be
            able to redefine the intensity. In general, this process will linearly scale
            the channels (e.g. map [0,255] |rarr| [0,1]).

        .. container:: rightside

            .. image:: figures/rescale_color.png

    .. container:: twocol

        .. container:: leftside

            **Padding** - The images of the dataset are all different dimensions; varied
            widths and heights. For most models, it is important they be consistent. One
            way to accomplish this is simply expanding the smaller images to match the 
            largest one. Padding will add pixels around the edges of an image to create
            the desired dimensions. The extra space can be filled in many ways, but for
            our purposes the filled space will match the edge-most pixels of the
            original image.


        .. container:: rightside

            .. image:: figures/padding.png

    .. container:: twocol

        .. container:: leftside
    
            **Cropping** - Another way to resize all images consistently, is to crop the
            larger images to match the size of the smallest image. This has an added
            benefit of augmenting our dataset in that most images will have multiple
            representations, as a window smaller than the image is moved randomly about
            to create the cropping.

        .. container:: rightside

            .. image:: figures/cropping.png

Features
^^^^^^^^^^^
    .. container:: twocol

        .. container:: leftside

            **Selection** - Choosing which features are relevant and which might leak
            target information to the model in training is important to the
            generalization of the model. A few examples:

                * Features like ID tags can be excluded for being unique and arbitrary.
                * Features which all contain the same value can be can be excluded for
                  being uninformative.
                * Features which describe biopsy results can be excluded for leaking
                  target information (as a benign lesion would have no biopsy results).

        .. container:: rightside

            .. image:: figures/selection.png
                
    .. container:: twocol

        .. container:: leftside

            **Ordinal Encoding** - Some features are made up of various categories.
            If these are described in text, it is difficult for many models to use.
            Since numerical values are more easily understood, each category
            within a feature a is assigned unique number, easily translating the
            information while preserving the idea behind the categories.

        .. container:: rightside

            .. image:: figures/ordinal_encoding.png
                
    .. container:: twocol

        .. container:: leftside

            **Fill NaN** - When data is incomplete, values are generally still
            expected by the models for every data point. So, a decision must be
            made on how to fill in missing data. Average values from the data
            points that are complete can be used, or just a value that would
            otherwise never exist for that feature.

        .. container:: rightside

            .. image:: figures/fill_nan.png

.. _V1.0:

Version 1.0
-----------
    A vision transformer taking features as query tokens for the decoder. Image and
    features are preprocessed, features are fed to a feature reducer, then all
    combined by a transformer to produce a classification.

Preprocessing
^^^^^^^^^^^^^^^^
    **Images** - All image channels are linearly rescaled from [0,255] to [0,1].
    In the case of the ISIC dataset, the smallest images forced a smaller cropping
    of images than desired, so first images are padded to 200x200 (images larger
    than this are unpadded), then all images are cropped to 125x125. The cropping
    window positioning is selected randomly each time the image is loaded from the
    dataset.

    **Features** - 
        
        * Exclusions: Identification features are excluded from training data for being
          irrelevant to diagnosis. These include *"isic_id", "patient_id", "lesion_id",
          "attribution", "copyright_license"*. Further, features which exist only because
          of a confirmed diagnosis are excluded, including *"iddx_full", "iddx_1",
          "iddx_2", "iddx_3", "iddx_4", "iddx_5", "mel_mitotic_index", "mel_thick_mm",
          "tbp_lv_dnn_lesion_confidence"*.
        * Ordinal Encoding: All text classifications which remain are assigned a
          unique (within each feature) id number in place of the text description.      
        * Fill NaN: Some *"age_approx"* values are missing, so these are filled as
          -1 to help the model distinguish and lean less on this less distinctive
          information.

Model
^^^^^^^^^

    .. _mod_arc:
    .. figure:: figures/model.png

        Basic flow of model architecture.

    :numref:`mod_arc` shows how the information provided by the ISIC dataset is 
    processed. First, a feature reducer transforms the features which compliment the
    images. This focuses the model on the most meaningful feature information
    allowing for more effective use of the available data. In particular, for this 
    iteration of the model, Principal Component Analysis (PCA) is used including
    enough dimensions to explain 99.99% of variance in the data.

    Next, embeddings are created for both the image and the reduced feature set.
    For the features, this is a small fully connected neural network; 2 layers with
    a ReLU activation in between, the initial layer 64 nodes wide, the next twice
    that, with the idea to create two 64 feature queries for the transformer
    decoder. For the image, two embeddings are created. One, a patch embedding to 
    reduce the sequence length input into the transformer encoder, following the
    idea of :cite:t:`vaswani2023attentionneed`. Here, a patch of pixels
    have their channel values concatenated, trading a greater number of features for
    fewer transformer inputs. Further, a linear transformation is applied to allow 
    for varied patch sizes while maintaining a consistent feature dimension between 
    all embeddings. Two, a positional embedding is used to maintain information of
    relative placement between patches. An embedding space of learnable parameters
    is created the size of NxM, where N is the number of patches and M the desired
    dimension of the features (again, 64 in this case).

    The image embeddings, patch and positional, are then summed before taken as
    input to the attention-based transformer encoder. The encoder has 4 layers of
    attention with 8 heads, add and normalization, and 1024 dimension feed forward
    networks (typical transformer encoder layers provided by
    :cite:t:`vaswani2023attentionneed`). The result then used as memory in
    conjunction with the queries created of the feature embeddings as input to the
    decoder. The decoder is of similar dimension to the encoder.

    Finally, the two queries create 2 sets of 1024 dimension outputs from the
    transformer, which are flattened and passed to a linear layer to reduce all the
    information down to logits representing whether the lesion is benign (dim 0) or
    malignant (dim 1).

Training
^^^^^^^^^^
    To train the model first the feature reducer, PCA, is fit to the available
    feature data. This process is quick and straightforward.

    The trained feature reducer can then be used to feed the classifier model. 
    Training the classifier requires a balancing of the classifications. There
    exist 400,666 benign lesions to 393 malignant, an imbalance which causes
    little to be learned about malignant lesions. To address this, all available
    malignant examples are duplicated within the dataset to create a roughly
    equal number number of positive and negative classification examples.

    With a balanced training set, a K-fold scheme is used to divide the dataset
    into training and validation subsets. 4 folds were used in training.

    An Adam based optimizer is used for its ability to achieve reasonable
    results without significant effort put into tuning of hyperparameters.

    To assess loss, cross entropy is used, taking the logits of the transformer
    compared to the target provided by the data set.

Results - pAUC: 0.021
^^^^^^^^^^^^^^^^^^^^^^^
    Over 4 iterations over the 4 folds, accuracy, precision, and recall end up
    over 99%. However, once tested in competition, the score achieved is quite
    poor, an pAUC of 0.021.

Lessons Learned
^^^^^^^^^^^^^^^^^^
    A poor score was expected, as multiple epochs have not yet been introduced
    to the training regimen. However, given the training results compared to the
    test, it is clear there is also a significant amount of information leakage.
    Care must be taken in the balancing of the dataset so that some of the
    malignant examples are held for the validation set and are in no way a part
    of the training set. Further, it is important the K-fold process not use the
    same model for different folds. Attention to these issues should make for a 
    better generalizing model.

    Weights in the loss function may also help better balance the dataset and 
    enable better generalization, but will come at a cost of requiring many
    training epochs.

.. _V1.1:

Version 1.1
-----------
    Taking from the lessons learned in 1.0, corrections have been made to 
    prevent information leakage between training and validation data.
    Additionally, scalable training has been introduced courtesy of the Ray
    python library :cite:`moritz2018raydistributedframeworkemerging`.

Training
^^^^^^^^^^
    To address the training set class imbalance, and to avoid data leakage, 
    weights have been applied to the cross entropy loss calculation. The impact 
    of loss calculated based on benign labels then is significantly less than
    that of the malignant labels, with a weight of 393/401,059 to malignant's
    400,666/401,059. Given the very few number of malignant examples, it is 
    still expected many epochs will be necessary for good performance.
    
    Further, the K-fold training scheme has been revised to create as many
    models to train as their exist folds. However, for this training round in
    particular, a single model is trained, and the dataset is simply split, 
    using 80% of it for training and the remaining 20% for validation, ensuring
    a proportional number of classification examples in each split.

    A future goal remains to balance the the dataset by duplicating malignant
    examples. Each epoch will be more effective, and it affords the opportunity
    to augment the dataset in other ways like various transformations of the
    images.

    With Ray :cite:`moritz2018raydistributedframeworkemerging`, hardware
    requirements can be defined per training instance. Then, depending on the
    resources made available to the ray server, multiple training instances can
    be run simultaneously, seen below in :numref:`parallel_training`.
    Additionally, Ray provides a tuning module which allows an easy means of
    exploring multiple training configurations, along with the application of
    optimization algorithms. Specifically, for this version, 4 feature reduction
    techniques are explored: none at all, and principal component analysis fit
    to explain 80%, 99%, and 99.99% of data variance.
    
    .. _parallel_training:
    .. figure:: figures/parallel_training.png

        Parallel training enabled by Ray library.
    
    Each model was trained with only a difference in the dataset feature
    reduction transformation and the following configuration in common:

        ===============  =============
        Option           Value
        ===============  =============
        Epochs           20
        Optimizer        Adam
        Learning Rate    0.00005
        ===============  =============

Results - pAUC: 0.109
^^^^^^^^^^^^^^^^^^^^^^^^^^^
    Much more promising than the previous version, validation results are now
    much less than perfect, even after 5 times the training iterations. It seems
    the data leakage problems have been addressed. Then, our validation results 
    are much more reliable in determining effective models.
    :numref:`1.1tensorboard` shows our slowly converging loss, true for both the
    training and validation set, meaning learning is occurring and at least some 
    generalization of whats being learned can be expected. For this dataset,
    because of the large proportion of benign results, 'accuracy' is poorly
    representative of the models capability with respect to correctly 
    identifying malignant lesions. So as exciting as the greater than 99%
    accuracy may be, more importantly to this case we track precision and
    recall, focusing on the malignant examples.
    
    .. _1.1tensorboard:
    .. figure:: figures/1.1tensorboard.png

        Training results for classifier trained with PCA explaining 99.99% 
        (00000), 99% (00001), 80% (00002) variance, and no feature reduction 
        (00003).

    While no result for recall or precision is great in 
    :numref:`1.1tensorboard`, it is clear that no feature reduction, and PCA
    capturing the most variance at 99.99% perform best. So, these are submitted
    to the kaggle competition to ensure we are on the right track, with a marked
    improvement in pAUC of .100 for the version using PCA, and .109 for the
    version with no feature reduction at all.

Lessons Learned
^^^^^^^^^^^^^^^^
    Seen in :numref:`1.1tensorboard`, precision and recall leave a lot to be
    desired, yet it is clear the models are still learning from the continually
    declining loss. Additional epochs will likely be of great use to the models,
    though will cost significant time. A more complex learning rate scheme may
    also be of use, scheduling a decaying rate for instance, enabling aggressive
    learning up front, while still including nuanced learning capability toward
    the end.
    
    
.. _V1.2:

Version 1.2
-----------
    Recognizing 2 decoder feature queries as an arbitrary choice, I have
    modified the sequence the decoder receives to be a set of queries, one for
    each feature.
    
Model
^^^^^^^^^^
    The feature set, reduced or otherwise, is separated creating
    a vector for each feature where all other features are zero. These vectors
    are then fed to an embedding space, a dense neural network, which first
    linearly projects each vector individually, passes that projection through
    an activation function, then performs another linear projection
    interconnecting the results. This provides the vectors a chance for
    embedding considering only themselves as well as an embedding based on
    relationships. These embedded feature vectors are then passed to the
    decoder. The updated process is visualized in :numref:`feature_embedding_update`,
    all other elements to the model remain the same.

    .. _feature_embedding_update:
    .. figure:: figures/feature_embedding_update.png

        Update to feature embedding process, including a mask to allow each
        feature to produce a decoder query.

Results - pAUC: 0.138
^^^^^^^^^^^^^^^^^^^^^^^^
    :numref:`1.2tensorboard` shows the new feature embedding scheme
    much improved, particularly for the model including feature reduction, as 
    accuracy, precision, and recall all score higher. For the model without
    feature reduction, precision scores as well as the previous models, lags in
    recall, but seems to do considerably better in accuracy. The final test then
    is a submission to the Kaggle competition, and the model with feature
    reduction scores 0.131, while the model without scores 0.138. 

    .. _1.2tensorboard:
    .. figure:: figures/1.2tensorboard.png

        Training results for classifier trained with old feature embedding
        structure (d9499) and the new structure (91029), coupled with PCA
        explaining 99.99% (00000) and no feature reduction (00003). Results have
        been smoothed for clarity.

Lessons Learned
^^^^^^^^^^^^^^^^^^^^
    The most obvious lesson is that the metrics currently used still make it
    difficult to discern which model will perform best. Though more complicated,
    a pAUC metric run during validation would be more relevant to the goals of
    this competition.

    Clearly the structure of the embeddings fed to the decoder is important. It
    could be the increased length of the sequence that is more useful, the 
    holding features distinct in their embedding before relating them, or a
    mix of both. Since the decoder already serves all kinds of inter-relational
    analysis, it could serve to keep the features separate throughout the
    embedding process, but the current process was chosen to save space.
    An arbitrarily longer sequence could also be generated, which may be
    worthwhile in the current structure where feature inter-relations are
    embedded more than individual, so it could make sense to provide a sequence 
    with up to as many vectors as there are combinations of features.

.. _V1.3:

Version 1.3
-----------
    To gain better generalization performance, I introduce greater randomization
    in the preprocessing, particularly of the images.

Preprocessing
^^^^^^^^^^^^^^^^
    It can be difficult to anticipate the quality of the images to be received
    in cases like different users submitting cell phone photos, the emulated
    scenario for this competition. In all these cases of preprocessing, the aim
    is to mitigate the impact of non-standard qualities which can be commonly 
    varied by smart phone filters and user indifference to orientation.

    .. container:: twocol

        .. container:: leftside

            **Brightness Adjustment** - A constant increase or decrease in all
            pixel RGB values, capped at either side of the spectrum (white can't
            get whiter, black can't get blacker).

        .. container:: rightside

            .. image:: figures/brightness_adjustment.png

    .. container:: twocol

        .. container:: leftside

            **Contrast Adjustment** - A multiplicative increase or decrease in
            all pixel RGB values, capped at either side of the spectrum.

        .. container:: rightside

            .. image:: figures/contrast_adjustment.png

    .. container:: twocol

        .. container:: leftside

            **Flip** - An exchange of pixels, mirrored across a horizontal or
            vertical axis, or both.

        .. container:: rightside

            .. image:: figures/flip_adjustment.png

    In training, contrast adjustments apply a random multiplier between 0.5 and
    2. Brightness adjustments are a random value between -50 and 50.

Results - pAUC: 0.126
^^^^^^^^^^^^^^^^^^^^^^^^
    :numref:`1.3tensorboard` shows a significant hit to accuracy. This could
    be due to the extreme values applied in brightness and contrast adjustments.
    Recall seems to be flipped, much improved, but given the accuracy results,
    and recall's steady decrease as training continues, more than anything
    the additional preprocessing seems to encourage excessive false positives.
    When submitted to Kaggle, the model with feature reduction scores 0.083,
    while the model without scores 0.126.

    .. _1.3tensorboard:
    .. figure:: figures/1.3tensorboard.png

        Training results for classifier trained with additional preprocessing
        methods relative to the previous version. 4e72c is the new experiment
        results, with 00000 using the feature reducer and 00001 not. Results
        have been smoothed for clarity.

Lessons Learned
^^^^^^^^^^^^^^^^
    While the additional preprocessing may improve generalization in theory,
    it does so by creating a more confusing dataset. Then, for the model to
    overcome the confusion, training must be made more efficient, or extended
    at the least.

.. _V1.4:

Version 1.4
-----------
    To accommodate the confusion introduced by the random preprocessing 
    techniques, the same model will be run for a greater number of epochs.
    Previously, trainings lasted for 20 epochs, so this will be run for 100.

Results - pAUC: 0.149
^^^^^^^^^^^^^^^^^^^^^^^^
    After 20 epochs, nearing convergence in loss, :numref:`1.4tensorboard` shows
    an additional drop and continued learning, throughout the 100 epochs. Then,
    the extended learning session for the randomized data inputs was worthwhile.
    That said, most metrics appear similar to the previous iteration, with a
    small improvement in accuracy. Since the validation set contains mostly
    benign examples, even a small change in accuracy should represent a
    significant reduction in false positives. 
    
    When submitted to Kaggle, the model with feature reduction scores 0.127,
    while the model without scores 0.149.

    .. _1.4tensorboard:
    .. figure:: figures/1.4tensorboard.png

        Training results for classifier trained for 100 epochs. Model 0 is using
        the feature reducer and model 1 is not. Results have been smoothed for
        clarity.

Lessons Learned
^^^^^^^^^^^^^^^^^^
    Again, it is difficult to decipher results without a specific measure for
    pAUC. Further, while the longer training time does come with improvement,
    it also comes at great cost. A more efficient training method should be
    explored.

.. _V1.5:

Version 1.5
-----------
    In an attempt to create a more efficient training cycle, I will utilize a
    cyclical learning rate. :cite:t:`smith2017cyclicallearningratestraining`
    provides a study of the idea with encouraging results. Training tended to
    be slightly improved by metric, and this result should be achieved with
    fewer epochs.

    To start, :cite:t:`smith2017cyclicallearningratestraining` suggests a dry 
    run using a slowly increasing learning rate. Then, noting when learning
    takes place, i.e. metrics are quickly improved, and when it ceases, i.e.
    metrics become unstable, provides an ideal range for the cycle.
    
    The process is shown in :numref:`1.5tensorboard_lrcycle` below. To ensure
    confidence in explicit ability of the models pAUC has also been introduced
    as a metric. Then, considering pAUC, learning for model 0 seems to start
    immediately and become unstable sometime before epoch 4. Model 1 on the
    other hand starts learning about epoch 3 and becomes unstable before epoch
    7. So, the learning rates chosen range from [1.0e-6, 3.0e-5] for model 0,
    and from [2.5e-5, 5.0e-5] for model 1.

    .. _1.5tensorboard_lrcycle:
    .. figure:: figures/1.5tensorboard_lrcycle.png

        Training dry run. Model 0 is using the feature reducer and model 1 is
        not. Learning rate is changed throughout training but not during
        validation.

Results - pAUC: 0.145
^^^^^^^^^^^^^^^^^^^^^^^^
    Interestingly, the cyclic learning rate seems to save an initially unstable
    training, perhaps saving the model from a poor local minima as the learning
    rate grows. This is seen in :numref:`1.5tensorboard` where accuracy is
    measured at 0 until a drastic shift in the loss trend about epoch 9. That
    said, it does not appear the cycle has helped the models learn more quickly,
    nor achieve a better result overall. The learning pace may be due to the
    random preprocessing of data, now coupled with a random learning rate as
    batches of data could be drawn in any order. This is a less deterministic
    training pipeline than presented in the :cite:t:`smith2017cyclicallearningratestraining`
    paper. Additionally, while they showed improved results in a cyclical
    learning rate of a static learning rate, the improvement was marginal,
    and likely thwarted by the random training pipeline.
    
    When submitted to Kaggle, the model with feature reduction scores 0.135,
    while the model without scores 0.145.

    .. _1.5tensorboard:
    .. figure:: figures/1.5tensorboard.png

        Training results for classifier using a cyclic learning rate. Model 0
        is using the feature reducer and model 1 is not.

Lessons Learned
^^^^^^^^^^^^^^^^^
    Given how rare the malignant examples remain, it is important to get the
    learning rate range and size of the cycle period right. This is something
    that may not be so straightforward with the introduction of random
    preprocessing in the training pipeline. Moreover, it may be inappropriate
    to use a cyclic learning rate with an unbalanced dataset as it is difficult
    to ensure the sparser labels receive similar attention to the data set in
    general. In the worst case, the sparse labels are only presented ever at
    lower learning rates, causing little learning despite the weight in the
    loss calculation. This may be offset by a longer cycle period, spanning
    several epochs.

.. _V1.6:

Version 1.6
-----------
    Finally, in an effort to better incorporate loss throughout the model,
    ensuring propagation of loss values and steering the model toward its
    ultimate goal of malignancy classification, I introduce intermediate
    classification/loss based on each layer of the decoder.

    For each layer of the decoder, a shared fully connected neural network takes
    its output to decide how the model would classify the data point so far. A
    loss is calculated and used along with the other auxiliary losses and the
    final classification's loss to determine how to adjust the parameters of the
    model. The losses are added, and the loss gradients with respect to each
    parameter is calculated as usual.

    Since the success of the cycled learning rate in this case is undetermined,
    I will train with this new loss scheme both with a static and cyclical
    learning rate. The static learning rate will be 5.0e-5 and the range for
    the cycle will be [1.0e-7, 2.5e-5], determined as before by the process
    shown in :numref:`1.6tensorboard_lrcycle`.

    .. _1.6tensorboard_lrcycle:
    .. figure:: figures/1.6tensorboard_lrcycle.png

        Training dry run. Model 0 is using the feature reducer and model 1 is
        not. Learning rate is changed throughout training but not during
        validation.

Results - pAUC: 0.146
^^^^^^^^^^^^^^^^^^^^^^^^
    Locally, results seemed promising, with the static learning rate, no feature
    reduction model performing well above 0.15 pAUC. However the remaining
    models did not fair as well and performed more similarly to what's already
    been seen. To that end, this could be a fluke, where the model is not
    actually well generalized, but happens to fit the validation set well. This
    is all but confirmed by its submission to Kaggle.
    
    When submitted to Kaggle, the model with feature reduction scores 0.131,
    while the model without scores 0.146.

    .. _1.6tensorboard:
    .. figure:: figures/1.6tensorboard.png

        Training results for classifier with auxiliary loss scheme. Model 0
        (with feature reduction) and 1 (without feature reduction) are using a
        static learning rate. Model 2 (with feature reduction) and 3 (without
        feature reduction) are using a cyclical learning rate.

Lessons Learned
^^^^^^^^^^^^^^^^^
    Despite an excellent measured training performance, testing revealed no
    improvement at all. It is clear then why techniques like K-fold are so
    invaluable. As much as it takes more time and computing power, the
    additional folds will really help determine a consistently well generalizing
    model from one that gets lucky. Something very important to be confident in
    prior to production.

Final Results
----------------
    With the close of the competition, scoring has been updated to include the
    complete testing dataset. Initial scores were based only on 28% of the
    testing dataset. Here are the updated results.

    +-------------------+------------+------------+-----------+------------+
    |                   | Kaggle Score                                     |
    +-------------------+------------+------------+-----------+------------+
    |                   | Initial                 | Final                  |
    +-------------------+------------+------------+-----------+------------+
    | Version           | Fet Red    | No Fet Red | Fet Red   | No Fet Red |
    +===================+============+============+===========+============+
    | :ref:`V1.0`       | 0.021      | 0.021      | 0.022     | 0.022      |
    +-------------------+------------+------------+-----------+------------+
    | :ref:`V1.1`       | 0.100      | 0.109      | 0.097     | 0.103      |
    +-------------------+------------+------------+-----------+------------+
    | :ref:`V1.2`       | 0.131      | 0.138      | 0.110     | 0.121      |
    +-------------------+------------+------------+-----------+------------+
    | :ref:`V1.3`       | 0.087      | 0.130      | 0.096     | 0.101      |
    +-------------------+------------+------------+-----------+------------+
    | :ref:`V1.4`       | 0.127      | **0.149**  | 0.106     | 0.122      |
    +-------------------+------------+------------+-----------+------------+
    | :ref:`V1.5`       | 0.135      | 0.145      | 0.117     | **0.126**  |
    +-------------------+------------+------------+-----------+------------+
    | :ref:`V1.6`       | 0.131      | 0.146      | 0.111     | 0.125      |
    +-------------------+------------+------------+-----------+------------+

        .. rst-class:: center

            *All scores for Feature Reduction based models refer to a PCA
            transformation explaining 99.99% of variance.*

Lessons learned
^^^^^^^^^^^^^^^^^
    Many of my initial scores were much higher than the final score,
    demonstrating some overfit to the initial test set. There was nothing in
    particular I implemented that sought to fit the test set, but it is possible
    the initial test set is more similar to the training data than the remaining
    test set. For that, it would be necessary to pursue further means of
    supplementing the training set, similar to the random preprocessing done.
    Possibly there exist other similar training sets which could be used to
    expand what I have available. Even if only images are available, it could
    be useful to train part of the model.

    With a goal of clearing 0.15 pAUC, I came close, until the remainder of the
    test set was released. I am pleased with the improvements made, though some
    were more effective than others. Each was relevant to shaping how I thought
    about this problem and better informed the next iteration. One thing I
    believe held back my models' performance is the fact that it was trained
    from scratch each time. In the future I would like to fine tune a
    pre-trained classification model, or other image based model, already
    familiar with the features in an image. Other competitors also had similar
    yet distinct ideas on how to combine the feature and image data, which I
    found interesting and would like to explore. For example, one ViT based
    model would train strictly on the image, then the result of the ViT's
    classification would be included in the feature set for the remainder of the
    model to produce the final classification. In all, there are many ways to
    tackle this problem and it was an exciting competition to be a part of.

