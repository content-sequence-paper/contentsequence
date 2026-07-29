"""Base classes for Bayesian content-sequence models."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az

from abc import ABC, abstractmethod
from typing import Tuple

from .priors import *

class BaseModel(ABC):
    """Base class for Bayesian models over one-dimensional time series.

    Attributes
    ----------
    model : pymc.Model | None
        Lazily built PyMC model graph.
    idata : arviz.InferenceData
        Aggregated inference outputs from prior, posterior, and predictive runs.
    priors : PolynomialSinusoidalPrior | None
        Prior configuration for the latent mean structure.
    sequence_bounds : tuple[float, float]
        Inclusive lower and upper bounds for observed sequence values.
    """

    def __init__(self, sequence_bounds: Tuple[float, float] = (0.0, 100.0)) -> None:
        """Initialize model state.

        Parameters
        ----------
        sequence_bounds
            Finite ``(lower, upper)`` bounds for the observed measure. The lower
            bound must be strictly smaller than the upper bound.
        """

        # Public attributes
        self.priors = None
        self.model: pm.Model | None = None
        self.idata: az.InferenceData = az.InferenceData()
        self.sequence_bounds = self._validate_sequence_bounds(sequence_bounds)

        # Private attributes
        self._time_data: None | np.ndarray = None
        self._sequence_data: None | np.ndarray = None

    @abstractmethod
    def create_model(self):
        """
        Validate required state before subclasses build a PyMC model.
        """
        if self._time_data is None or self._sequence_data is None:
            raise ValueError("Both `time_data` and `sequence_data` must be set first.")

    @property
    def time_data(self):
        """numpy.ndarray | None: Normalized sequence locations.

        Assigned values must be finite, non-empty, one-dimensional, strictly
        increasing, and contained in ``[0, 1]``.
        """
        return self._time_data

    @time_data.setter
    def time_data(self, time_data):
        """Set normalized locations after validating shape, values, and length."""
        time_array = np.asarray(time_data, dtype=float)
        self._check_one_dimensional(time_array, "time_data")
        self._check_finite(time_array, "time_data")
        self._check_data_len(time_array, self.sequence_data)
        self._check_time_validity(time_array)
        self._time_data = time_array

    @property
    def sequence_data(self):
        """numpy.ndarray | None: Observed content-sequence values.

        Assigned values must be finite, non-empty, one-dimensional, contained
        in :attr:`sequence_bounds`, and the same length as :attr:`time_data`.
        """
        return self._sequence_data

    @sequence_data.setter
    def sequence_data(self, sequence_data):
        """Set observations after validating shape, values, bounds, and length."""
        sequence_array = np.asarray(sequence_data, dtype=float)
        self._check_one_dimensional(sequence_array, "sequence_data")
        self._check_finite(sequence_array, "sequence_data")
        self._check_data_len(self.time_data, sequence_array)
        lower, upper = self.sequence_bounds
        if not np.all((sequence_array >= lower) & (sequence_array <= upper)):
            raise ValueError("`sequence_data` must lie within `sequence_bounds`.")
        self._sequence_data = sequence_array

    @property
    def sequence_range(self) -> float:
        """float: Width of ``sequence_bounds`` used to scale default priors."""
        return self.sequence_bounds[1] - self.sequence_bounds[0]

    @staticmethod
    def _validate_sequence_bounds(sequence_bounds) -> tuple[float, float]:
        """Return validated finite bounds with strictly positive width."""
        try:
            lower, upper = sequence_bounds
        except (TypeError, ValueError) as error:
            raise ValueError(
                "`sequence_bounds` must contain exactly two numeric values."
            ) from error
        try:
            lower = float(lower)
            upper = float(upper)
        except (TypeError, ValueError) as error:
            raise ValueError("`sequence_bounds` must be numeric.") from error
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError("`sequence_bounds` must be finite.")
        if lower >= upper:
            raise ValueError("The lower sequence bound must be smaller than the upper bound.")
        return lower, upper

    @staticmethod
    def _check_one_dimensional(data: np.ndarray, name: str) -> None:
        """Validate that an input is a non-empty one-dimensional array."""
        if data.ndim != 1 or data.size == 0:
            raise ValueError(f"`{name}` must be a non-empty one-dimensional array.")

    @staticmethod
    def _check_finite(data: np.ndarray, name: str) -> None:
        """Validate that an input contains only finite values."""
        if not np.all(np.isfinite(data)):
            raise ValueError(f"`{name}` must contain only finite values.")

    def _check_time_validity(self, time_data: np.ndarray) -> int:
        """Validate that time data is strictly increasing and between zero and one."""
        if not np.all(np.diff(time_data) > 0):
            raise ValueError("`time_data` must be strictly increasing.")
        if not np.all((time_data >= 0) & (time_data <= 1)):
            raise ValueError("`time_data` must be between zero and one.")
        return 0

    def _check_data_len(self, time_data: None | np.ndarray, sequence_data: None | np.ndarray) -> int:
        """Validate that time and sequence inputs have the same length."""
        if time_data is not None and sequence_data is not None:
            if len(time_data) != len(sequence_data):
                raise ValueError("Length of `time_data` and `sequence_data` does not match.")
        return 0

    def sample_prior_predictive(self, *args, **kwargs):
        """Sample from the prior predictive distribution.

        Parameters
        ----------
        *args, **kwargs
            Forwarded to :func:`pymc.sample_prior_predictive`.

        Returns
        -------
        arviz.InferenceData | dict
            PyMC prior predictive result, depending on backend options.
        """
        if self.model is None:
            self.create_model()
        with self.model:
            trace = pm.sample_prior_predictive(*args, **kwargs)
            self.idata.extend(trace)
        return trace

    def sample(self, *args, **kwargs):
        """Run posterior sampling for the current model.

        Parameters
        ----------
        *args, **kwargs
            Forwarded to :func:`pymc.sample`.

        Returns
        -------
        arviz.InferenceData | pymc.backends.base.MultiTrace
            Posterior sample result produced by PyMC.
        """
        if self.model is None:
            self.create_model()
        with self.model:
            trace = pm.sample(*args, **kwargs)
            self.idata.extend(trace)
        return trace

    def sample_posterior_predictive(self, trace=None, *args, **kwargs):
        """Sample from the posterior predictive distribution.

        Parameters
        ----------
        trace
            Posterior samples to condition on. Defaults to this model's accumulated
            :class:`arviz.InferenceData`.
        *args, **kwargs
            Forwarded to :func:`pymc.sample_posterior_predictive`.

        Returns
        -------
        arviz.InferenceData | dict
            Posterior predictive result, depending on backend options.
        """
        if self.model is None:
            self.create_model()
        if trace is None:
            if "posterior" not in self.idata.groups():
                raise ValueError("Posterior samples are required before posterior prediction.")
            trace = self.idata
        with self.model:
            predictive = pm.sample_posterior_predictive(trace, *args, **kwargs)
            self.idata.extend(predictive)
        return predictive

    def fit(
        self,
        prior_draws: int = 1000,
        tune: int = 1000,
        draws: int = 1000,
        target_accept: float = 0.95,
        random_seed=None,
        progressbar: bool | str = True,
        **sample_kwargs,
    ):
        """
        Run the full Bayesian workflow: prior predictive, posterior sampling, and posterior predictive.

        Parameters
        ----------
        prior_draws
            Number of prior-predictive samples.
        tune, draws, target_accept
            Posterior sampling controls forwarded to :func:`pymc.sample`.
        random_seed
            Seed used for every sampling stage.
        progressbar
            PyMC posterior progress display. The default ``True`` shows progress
            information. PyMC also accepts the ``"combined"`` and
            ``"split"`` progress-bar layouts. Posterior-predictive progress is
            enabled whenever this value is truthy.
        **sample_kwargs
            Additional keyword arguments forwarded to :func:`pymc.sample`.

        Returns
        -------
        arviz.InferenceData
            Aggregated inference results from prior, posterior, and predictive runs.
        """
        self.sample_prior_predictive(draws=prior_draws, random_seed=random_seed)
        self.sample(
            tune=tune,
            draws=draws,
            target_accept=target_accept,
            init="adapt_diag",
            random_seed=random_seed,
            progressbar=progressbar,
            idata_kwargs={"log_likelihood": True},
            **sample_kwargs,
        )
        self.sample_posterior_predictive(
            random_seed=random_seed,
            progressbar=bool(progressbar),
        )
        return self.idata

class LikelihoodModel(ABC):
    """Interface for likelihood mixins that extend a latent mean model.

    Likelihood implementations create their own parameters and the ``observed``
    random variable inside the active PyMC model context.
    """

    @abstractmethod
    def _create_likelihood(self, *, mu, observed, lower, upper):
        """Create likelihood parameters and the observed random variable."""


class NormalLikelihoodModel(LikelihoodModel):
    """Mixin that adds a bounded Normal observation likelihood.

    The default scale prior is a half-Normal with standard deviation equal to
    one tenth of ``sequence_range``.

    Attributes
    ----------
    likelihood_priors : NormalLikelihoodPrior
        Configurable prior for the observation standard deviation.
    """

    def __init__(self, *args, likelihood_priors=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.likelihood_priors = (
            likelihood_priors
            if likelihood_priors is not None
            else NormalLikelihoodPrior(
                sigma=LocationScaleScalarPrior(
                    mu=0.0,
                    sigma=self.sequence_range / 10,
                    lower=0.0,
                ),
            )
        )

    def _create_likelihood(self, *, mu, observed, lower, upper):
        priors = self.likelihood_priors
        sigma = pm.TruncatedNormal(
            "sigma",
            mu=priors.sigma.mu,
            sigma=priors.sigma.sigma,
            lower=priors.sigma.lower,
            upper=priors.sigma.upper,
        )
        return pm.Truncated(
            "observed",
            pm.Normal.dist(mu=mu, sigma=sigma),
            lower=lower,
            upper=upper,
            observed=observed,
        )


class StudentTLikelihoodModel(LikelihoodModel):
    """Mixin that adds a bounded Student-t observation likelihood.

    The default observation-scale prior is a half-Normal with standard deviation
    equal to one tenth of ``sequence_range``. Degrees of freedom follow
    ``2 + Exponential(scale=10)``, which has mean 12 and keeps support above 2.

    Attributes
    ----------
    likelihood_priors : StudentTLikelihoodPrior
        Configurable priors for observation scale and degrees of freedom.
    """

    def __init__(self, *args, likelihood_priors=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.likelihood_priors = (
            likelihood_priors
            if likelihood_priors is not None
            else StudentTLikelihoodPrior(
                sigma=LocationScaleScalarPrior(
                    mu=0.0,
                    sigma=self.sequence_range / 10,
                    lower=0.0,
                ),
                dof=ScaleScalarPrior(scale=10.0, lower=2.0),
            )
        )

    def _create_likelihood(self, *, mu, observed, lower, upper):
        priors = self.likelihood_priors
        sigma = pm.TruncatedNormal(
            "sigma",
            mu=priors.sigma.mu,
            sigma=priors.sigma.sigma,
            lower=priors.sigma.lower,
            upper=priors.sigma.upper,
        )
        if priors.dof.scale <= 0:
            raise ValueError("The `dof` prior scale must be positive.")
        dof_lower = 0.0 if priors.dof.lower is None else priors.dof.lower
        if priors.dof.upper is None:
            dof_unconstrained = pm.Exponential(
                "dof_unconstrained",
                lam=1 / priors.dof.scale,
            )
        else:
            dof_width = priors.dof.upper - dof_lower
            if dof_width <= 0:
                raise ValueError("The `dof` upper bound must exceed its lower bound.")
            dof_unconstrained = pm.Truncated(
                "dof_unconstrained",
                pm.Exponential.dist(lam=1 / priors.dof.scale),
                lower=0.0,
                upper=dof_width,
            )
        dof = pm.Deterministic(
            "dof",
            dof_lower + dof_unconstrained,
        )
        return pm.Truncated(
            "observed",
            pm.StudentT.dist(nu=dof, mu=mu, sigma=sigma),
            lower=lower,
            upper=upper,
            observed=observed,
        )


class GenericPolynomialSinusoidalBaseModel(BaseModel):
    """Shared polynomial-sinusoidal mean model without a fixed likelihood.

    ``N`` is the polynomial degree, so the trend has ``N + 1`` coefficients.
    ``M`` is the number of sinusoidal harmonics. Default priors scale to
    ``sequence_bounds``: the linear intercept is centered on the range midpoint, while
    polynomial changes and sinusoidal amplitudes are expressed as fractions of
    the range.

    Parameters
    ----------
    N
        Non-negative polynomial degree.
    M
        Non-negative number of sinusoidal harmonics.
    sequence_bounds
        Finite lower and upper observation bounds.
    """

    def __init__(self, N, M, sequence_bounds: Tuple[float, float] = (0.0, 100.0)):
        if not isinstance(N, int) or N < 0:
            raise ValueError("`N` must be a non-negative integer polynomial degree.")
        if not isinstance(M, int) or M < 0:
            raise ValueError("`M` must be a non-negative integer harmonic count.")
        super().__init__(sequence_bounds=sequence_bounds)
        self.N = N
        self.M = M
        n_polynomial_coeffs = N + 1
        sequence_midpoint = sum(self.sequence_bounds) / 2
        polynomial_mu = [sequence_midpoint] + [0.0] * N
        polynomial_sigma = [self.sequence_range / 4] + [
            self.sequence_range / 2
        ] * N
        self.priors = PolynomialSinusoidalPrior(
            polynomial_coeffs=LocationScaleVectorPrior(
                mu=polynomial_mu,
                sigma=polynomial_sigma,
            ),
            sinusoidal_amplitude=LocationScaleVectorPrior(
                mu=[self.sequence_range / 10] * M,
                sigma=[self.sequence_range / 10] * M,
                lower=[0.0] * M,
            ),
            sinusoidal_frequency=LocationScaleVectorPrior(
                mu=[1.5] * M,
                sigma=[0.75] * M,
                lower=[0.0] * M,
            ),
            sinusoidal_phase=VonMisesVectorPrior(
                mu=[0.0] * M,
                kappa=[1.0] * M,
            ),
        )

    @abstractmethod
    def _create_likelihood(self, *, mu, observed, lower, upper):
        """Add likelihood-specific parameters and the observed variable."""

    def create_model(self):
        """Build and return the PyMC model graph for the configured data."""
        super().create_model()  # Validate required state before building the model
        priors = self.priors
        n_polynomial_coeffs = self.N + 1
        time_data = pt.as_tensor_variable(pm.floatX(self.time_data))

        def location_scale_rv(name, prior, shape):
            lower_unbounded = prior.lower is None or np.all(
                np.isneginf(np.asarray(prior.lower, dtype=float))
            )
            upper_unbounded = prior.upper is None or np.all(
                np.isposinf(np.asarray(prior.upper, dtype=float))
            )
            if lower_unbounded and upper_unbounded:
                return pm.Normal(
                    name,
                    mu=prior.mu,
                    sigma=prior.sigma,
                    shape=shape,
                )
            return pm.TruncatedNormal(
                name,
                mu=prior.mu,
                sigma=prior.sigma,
                lower=prior.lower,
                upper=prior.upper,
                shape=shape,
            )

        with pm.Model() as model:
            polynomial_coeffs = location_scale_rv(
                "polynomial_coeffs",
                priors.polynomial_coeffs,
                n_polynomial_coeffs,
            )
            polynomial_basis = pt.stack(
                [time_data**degree for degree in range(n_polynomial_coeffs)],
                axis=0,
            )
            expected_mu = pt.dot(polynomial_coeffs, polynomial_basis)

            if self.M > 0:
                sinusoidal_amplitude = location_scale_rv(
                    "sinusoidal_amplitude",
                    priors.sinusoidal_amplitude,
                    self.M,
                )
                sinusoidal_frequency = location_scale_rv(
                    "sinusoidal_frequency",
                    priors.sinusoidal_frequency,
                    self.M,
                )
                sinusoidal_phase = pm.VonMises(
                    "sinusoidal_phase",
                    mu=priors.sinusoidal_phase.mu,
                    kappa=priors.sinusoidal_phase.kappa,
                    shape=self.M,
                )
                angles = (
                    2 * np.pi * sinusoidal_frequency[:, None] * time_data[None, :]
                    - sinusoidal_phase[:, None]
                )
                expected_mu = expected_mu + pt.dot(
                    sinusoidal_amplitude,
                    pt.sin(angles),
                )

            mu = pm.Deterministic("mu", expected_mu)
            self._create_likelihood(
                mu=mu,
                lower=self.sequence_bounds[0],
                upper=self.sequence_bounds[1],
                observed=self.sequence_data,
            )

        self.model = model
        return model


class GenericPolynomialSinusoidalModel(
    StudentTLikelihoodModel,
    GenericPolynomialSinusoidalBaseModel,
):
    """Polynomial-sinusoidal mean with a bounded Student-t likelihood.

    Parameters
    ----------
    N, M
        Polynomial degree and number of sinusoidal harmonics.
    sequence_bounds
        Finite lower and upper observation bounds.
    likelihood_priors
        Optional replacement for the sequence-scaled Student-t defaults.
    """

    def __init__(
        self,
        N: int,
        M: int,
        sequence_bounds: Tuple[float, float] = (0.0, 100.0),
        likelihood_priors: StudentTLikelihoodPrior | None = None,
    ):
        super().__init__(
            N=N,
            M=M,
            sequence_bounds=sequence_bounds,
            likelihood_priors=likelihood_priors,
        )


class GenericPolynomialSinusoidalNormalModel(
    NormalLikelihoodModel,
    GenericPolynomialSinusoidalBaseModel,
):
    """Polynomial-sinusoidal mean with a bounded Normal likelihood.

    Parameters
    ----------
    N, M
        Polynomial degree and number of sinusoidal harmonics.
    sequence_bounds
        Finite lower and upper observation bounds.
    likelihood_priors
        Optional replacement for the sequence-scaled Normal defaults.
    """

    def __init__(
        self,
        N: int,
        M: int,
        sequence_bounds: Tuple[float, float] = (0.0, 100.0),
        likelihood_priors: NormalLikelihoodPrior | None = None,
    ):
        super().__init__(
            N=N,
            M=M,
            sequence_bounds=sequence_bounds,
            likelihood_priors=likelihood_priors,
        )
