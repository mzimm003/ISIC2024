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

Version 1
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
    * Fill NaN: Some *"age_approx"* values are missing, so these are filled as
      -1 to help the model distinguish and lean less on this less distinctive
      information.

Model
^^^^^^^^^

Training
^^^^^^^^^^
Balance dataset....