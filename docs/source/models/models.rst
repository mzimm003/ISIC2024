..    include:: <isonum.txt>

Models
========

While the competition proceeds, I will attempt several iterations of a
classifier model. These will be a combination of my own ideas, research, and 
ideas from the community Kaggle provides.

.. collapse::

    Preprocessing
-------------------
    For each model I provide preprocessing options. Some preprocessing is
    strictly necessary, but some make work in combination with certain models
    better than others. How the preprocessing options are used will be specified
    for each model. For the ISIC dataset, two major components exist. For each
    skin lesion their is an image and a set of recorded features about the case.
    Both must be preprocessed, but are done so distinctively.

    Images
^^^^^^^^^^^^
    **Rescaling Color** - Color in images is represented by 3 channels per
    pixel. One each for red, green, and blue. The intensity of the color in
    their respective channel is described on a range from 0 to 255. Machine
    learning models often do better on smaller scales so it is important to be
    able to redefine the intensity. In general, this process will linearly scale
    the channels (e.g. map [0,255] |rarr| [0,1]).

    **Padding** - The images of the dataset are all different dimensions; varied
    widths and heights. For most models, it is important they be consistent. One
    way to accomplish this is simply expanding the smaller images to match the 
    largest one. Padding will add pixels around the edges of an image to create
    the desired dimensions. 

    Features
^^^^^^^^^^^^^

Version 1
-----------
A vision transformer taking features 

Preprocessing
^^^^^^^^^^^^^^^^

