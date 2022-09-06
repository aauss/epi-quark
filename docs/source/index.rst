.. epi-quark documentation master file, created by
   sphinx-quickstart on Thu Nov  4 14:19:38 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to epi-quark's documentation!
=====================================

**epi-quark** is a library that, based on labeled data, scores outbreak detection, annotation or prediction algorithms by framing their tasks as multi-class and multi-label classification.

The motivation for this library is the lack of scoring methods that produces comparable scores for algorithms that use different aggregation and testing strategies.

Furthermore, this library also allows you to apply epidemiologically meaningful weighting. Scores can be weighted by case counts (detecting large outbreaks is more important than detecting small ones) or by spatio-temporal accuracy, i.e., the user might prefer detection close to the location, onset date, and duration of the actual outbreak.

.. note::

   This project is under active development.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   usage
   modules



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
