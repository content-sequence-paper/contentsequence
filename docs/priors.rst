Priors
======

Sequence-scaled defaults
------------------------

Let ``R = upper - lower`` and ``C = (upper + lower) / 2``. Default polynomial
models use:

* ``linear intercept ~ Normal(C, R / 4)``;
* higher polynomial coefficients ``~ Normal(0, R / 2)``;
* ``sigma ~ HalfNormal(R / 10)``.

The linear-sinusoidal model uses ``Normal(C, R / 4)`` for its linear intercept and
``Normal(0, R / 4)`` for its trend. Its amplitude is a positive
``TruncatedNormal(R / 10, R / 20)``, frequency is a positive
``TruncatedNormal(1.5, 0.3)``, phase is ``VonMises(0, 5)``, and degrees of freedom
are ``2 + Exponential(scale=10)``.

These defaults make a new model operational on any finite outcome range. They
remain assumptions and must be checked against domain knowledge.

Changing a prior
----------------

Change an individual prior value directly before fitting:

.. code-block:: python

   model.priors.linear_intercept.mu = 10
   model.priors.linear_intercept.sigma = 1

Other priors have the same structure, e.g.,

.. code-block:: python

   model.priors.sinusoidal_amplitude.mu = 5
   model.priors.sinusoidal_amplitude.sigma = 2

Some priors have a set lower bound, e.g., the Student-t degrees of freedom prior.
This can be changed similarly:

.. code-block:: python

   model.likelihood_priors.dof.lower = 5


Prior predictive checking
-------------------------

Set both data arrays, then run:

.. code-block:: python

   prior = model.sample_prior_predictive(draws=500, random_seed=42)

Plot both ``prior.prior["mu"]`` and
``prior.prior_predictive["observed"]``. Priors should
generate scientifically plausible trajectories without being so narrow that
they preclude meaningful alternatives.
