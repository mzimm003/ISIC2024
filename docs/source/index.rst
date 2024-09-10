.. ISIC 2024 documentation master file, created by
   sphinx-quickstart on Wed Jul 31 23:13:54 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ISIC 2024
=======================

.. toctree::
   :maxdepth: 2
   :titlesonly:

   Competition Site <https://www.kaggle.com/competitions/isic-2024-challenge>
   plan/plan
   models/models
   notes/notes
   references

A challenge to develop image-based algorithms to identify histologically
confirmed skin cancer cases with single-lesion crops from 3D total body
photos (TBP). :cite:`isic-2024-challenge`

Latest
--------

With the close of the competition, scoring has been updated to include the
complete testing dataset. Initial scores were based only on 28% of the
testing dataset. Here are the updated results.

Click on any version to learn more about it.

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