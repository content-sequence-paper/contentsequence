contentsequence
===============

`contentsequence` is a Python package for measuring content sequences in unstructured data. Developed alongside *The Shape of Content: Toward an Anatomy of Content Sequences in Unstructured Data Analysis*, it implements a Bayesian linear–sinusoidal model that estimates six dimensions of a sequence: opening position, opening direction, trend, frequency of peaks and valleys, amplitude of peaks and valleys, and the position of maxima and minima.

The package is modality-independent and uses [PyMC](https://www.pymc.io/) for
Bayesian inference and [ArviZ](https://python.arviz.org/) for diagnostics and model comparison.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   getting_started
   models
   priors
   inference
   example
   api
   citation

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
