.. epi-quark documentation master file, created by
   sphinx-quickstart on Thu Nov  4 14:19:38 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to epi-quark's documentation!
=====================================

**epi-quark** is a scoring library that helps to evaluate the prediction, detection, and annotation of infectious disease outbreaks. 

The motivation for this library is the lack of scoring methods that produces comparable scores for algorithms that use different aggregation and testing strategies.

Furthermore, this library also allows you to apply epidemiologically meaningful weighting. We can wait by the case counts (detecting large outbreaks is more important than detecting small ones) or by spatio-temporal accuracy, i.e., we prefer detected outbreaks to be precise in location, onset, and length of the outbreak. 

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
