Models
======

Available paper models
----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 20 30

   * - Model
     - Latent mean
     - Likelihood
     - Interpretation variables
   * - :class:`contentsequence.models.ConstantModel`
     - ``c``
     - Bounded Normal
     - Opening position and zero opening direction
   * - :class:`contentsequence.models.LinearModel`
     - ``c + m*t``
     - Bounded Normal
     - Opening variables and extrema locations
   * - :class:`contentsequence.models.QuadraticModel`
     - ``c + m*t + q*t**2``
     - Bounded Normal
     - Opening variables and extrema locations
   * - :class:`contentsequence.models.LinearSinusoidalModel`
     - ``c + m*t + a*sin(2*pi*f*t - u)``
     - Bounded Student-t
     - Opening variables, linear trend, and extrema locations

``CurvilinearModel`` is an alias for ``QuadraticModel``. Constant functions have
no unique extrema locations, so the constant model does not record them.

Generic models
--------------

:class:`contentsequence.base_models.GenericPolynomialSinusoidalModel` combines degree ``N``
and ``M`` harmonics with a Student-t likelihood. The corresponding
:class:`contentsequence.base_models.GenericPolynomialSinusoidalNormalModel` uses a Normal
likelihood.

.. code-block:: python

   import contentsequence as cs

   model = cs.GenericPolynomialSinusoidalModel(N=2, M=2)

The generic classes expose latent polynomial and sinusoidal parameters but do not
add paper-specific interpretation variables. Add these in a specialized subclass
when their definition is available for the requested structure.

Likelihood composition
----------------------

The structural base and likelihood components are separate. New fixed-likelihood
models can combine :class:`contentsequence.base_models.GenericPolynomialSinusoidalBaseModel`
with a class implementing the protected ``_create_likelihood`` hook. Likelihood
mixins must be placed first in the inheritance list so Python resolves the hook
from the mixin.
