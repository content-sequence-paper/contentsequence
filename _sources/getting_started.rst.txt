Getting started
===============

Installation
------------

From the package source directory:

.. code-block:: console

   python -m pip install -e .

Add notebook or documentation dependencies with ``.[notebook]`` or ``.[docs]``.

Fit a sequence
--------------

An example below uses a dummy content sequence with known set of parameters:
  - linear intercept = 50
  - linear trend = 4
  - sinusoidal amplitude = 10
  - sinusoidal frequency = 1.5 cycles per normalized sequence
  - sinusoidal phase = 0.5

.. code-block:: python

   import numpy as np
   import arviz as az
   import contentsequence as cs

   time = np.linspace(0, 1, 100)
   sequence = 50 + 4 * time + 10 * np.sin(3 * np.pi * time - 0.5)

   model = cs.LinearSinusoidalModel()
   model.time_data = time
   model.sequence_data = sequence
   model.fit()

   az.summary(model.idata)


Input Requirements
------------------

``time_data`` and ``sequence_data`` must be finite, non-empty, one-dimensional
arrays of equal length. Time must be strictly increasing inside ``[0, 1]``.
Observations must lie inside the finite increasing ``sequence_bounds`` supplied
when the model is created.

Change priors before the first sampling call. For example:

.. code-block:: python

   model.priors.linear_intercept.mu = 10
   model.priors.linear_intercept.sigma = 1

Inference output
----------------

The returned :class:`arviz.InferenceData` combines prior predictive, posterior,
posterior predictive, sample statistics, pointwise log likelihood, and observed
data. The latent fitted curve is ``mu`` and replicated observations are stored as
``observed`` in the posterior-predictive group.
