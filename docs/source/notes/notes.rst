..    include:: <isonum.txt>

Notes
=====

August 3
----------

Realized LDA necessarily reduces features to at most 1 less than the number of
classes which will not be useful for a binary classification problem. Instead,
I will concentrate on unsupervised feature reduction algorithms like Principal
component analysis.

Further, the features included in the training set but not the test set, may
correlate so strongly with the target label, that a model trained on this
information would not provide helpful inferences to a model not trained on this
information. Then, before attempting a teacher-student training structure, I
will simply proceed with the following:

    #. Perform feature reduction techniques on training set, limited to features
       only available in the test set. Explore various unsupervised feature
       reduction algorithms starting with PCA.
    #. Train main model using reduced feature dataset and images, optimizing
       with respect to cross entropy loss. Use a encoder-decoder transformer
       over the image, and an embedding of the features may work as the query
       to the decoder. Specifics to be determined.
    #. I will put together a single run as quickly as possible, then setup in
       ray for potential parallel runs and discrete optimization algorithms
       to find best configurations (like which feature reduction performs best.)

So far I have introduced infrastructure to enable preprocessing of data, and
fitting of some arbitrary unsupervised learner.

Preprocessing includes:

    * Rescaling of picture color values from [0,255] |rarr| [0,1]
    * Exclusion of uninformative data labels:
        * ID tags
        * Fully missing data
        * Data all of the same result
    * Exclusion of overly informative data labels (e.g. labels which are not the
      target but which contain data only when the target is positive)
    * Integer encoding of catorigcal data represented by text
    * Filling missing values as appropriate