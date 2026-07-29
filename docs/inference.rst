Inference and diagnostics
=========================

Workflow
--------

The convenience method :meth:`contentsequence.base_models.BaseModel.fit` runs:

#. Prior predictive sampling.
#. NUTS tuning and posterior sampling.
#. Posterior predictive sampling.

Use the three individual sampling methods when a workflow needs finer control.
Keyword arguments such as ``chains`` and ``cores`` are forwarded to
``pymc.sample`` by ``fit``.

Diagnostics
-----------

Before interpreting parameters:

* inspect trace and rank plots;
* confirm there are no divergences;
* check rank-normalized R-hat values are close to 1;
* check effective sample sizes are adequate for the summaries of interest;
* inspect prior and posterior predictive behavior;
* increase tuning or ``target_accept`` when diagnostics indicate a problem.

Posterior prediction
--------------------

``mu`` represents uncertainty in the latent mean. Posterior-predictive
``observed`` additionally includes observation noise. Use ArviZ highest-density
intervals for either quantity:

.. code-block:: python

   import arviz as az

   mu_hdi = az.hdi(idata.posterior["mu"], hdi_prob=0.95)["mu"]
   predictive_hdi = az.hdi(
       idata.posterior_predictive["observed"], hdi_prob=0.95
   )["observed"]

Model comparison
----------------

Models fitted to the same observations can be compared with:

.. code-block:: python

   comparison = az.compare(
       {"constant": constant.idata, "linear": linear.idata},
       ic="loo",
       method="stacking",
   )

Interpret LOO only after diagnostics and predictive checks pass for every model.
Review Pareto-k or warning indicators and avoid treating small ranking differences
as decisive without considering their uncertainty.
