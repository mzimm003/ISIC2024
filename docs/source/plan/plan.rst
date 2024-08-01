Plan
=====
Use a transformer based classifier to classify images. Leverage extra 
information in training set to create a teacher model. Create main model using
only what is available on test set, trained traditionally (cross entropy
loss) as well as distillation loss. Preprocess data with feature reduction
techniques to try to narrow the number of features trained on.

Steps
-----

#. Perform feature reduction techniques on training set. Having access to labels
   Linear Discriminant Analysis may do well.
#. Use training set, restricted to features identified in previous step, to
   train a teacher model. Using a encoder-decoder transformer over the image,
   an embedding of the features may work as the query to the decoder. Specifics
   to be determined.
#. Limit training set to features only available in test set, and again perform
   feature reduction.
#. Train main model using limited features, scoring loss on labels and teacher
   inference logits. Model structure can be the same as, but potentially smaller 
   than, the teacher model.