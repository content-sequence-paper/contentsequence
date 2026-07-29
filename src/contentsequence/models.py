"""Interpretable models for bounded content sequences."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from typing import Tuple

from .priors import *
from .base_models import GenericPolynomialSinusoidalModel, GenericPolynomialSinusoidalNormalModel
from .utils import _global_extrema_locations

class LinearSinusoidalModel(GenericPolynomialSinusoidalModel):
    """Fit the paper's linear-sinusoidal content-sequence model.

    The latent mean is

    ``c + m*t + a*sin(2*pi*f*t - phase)``.

    Default priors scale with ``sequence_bounds``. For bounds ``(0, 100)``, the
    linear intercept has mean 50 and standard deviation 25, the linear trend
    has standard deviation 25,
    amplitude is ``TruncatedNormal(10, 5)``, and observation scale is
    ``HalfNormal(10)``. Frequency and phase retain the paper-oriented defaults of
    ``TruncatedNormal(1.5, 0.3)`` and ``VonMises(0, 5)``.

    Parameters
    ----------
    sequence_bounds
        Finite lower and upper bounds for the observed sequence measure.

    Notes
    -----
    The model records ``opening_position``, ``opening_direction``,
    ``linear_trend``, ``minimum_location``, and ``maximum_location`` as
    deterministic posterior variables.
    """

    def __init__(self, sequence_bounds: Tuple[float, float] = (0.0, 100.0)):
        """Initialize the linear-sinusoidal model and its default priors."""
        super().__init__(N=1, M=1, sequence_bounds=sequence_bounds)
        sequence_midpoint = sum(self.sequence_bounds) / 2
        self.priors = PolynomialSinusoidalPrior(
            polynomial_coeffs=LocationScaleVectorPrior(
                mu=[sequence_midpoint, 0.0],
                sigma=[self.sequence_range / 4, self.sequence_range / 4],
            ),
            sinusoidal_amplitude=LocationScaleVectorPrior(
                mu=self.sequence_range / 10,
                sigma=self.sequence_range / 20,
                lower=0.0,
            ),
            sinusoidal_frequency=LocationScaleVectorPrior(
                mu=1.5,
                sigma=0.3,
                lower=0.0,
            ),
            sinusoidal_phase=VonMisesVectorPrior(
                mu=0.0,
                kappa=5.0,
            ),
        )
        self.likelihood_priors = StudentTLikelihoodPrior(
            sigma=LocationScaleScalarPrior(
                mu=0.0,
                sigma=self.sequence_range / 10,
                lower=0.0,
            ),
            dof=ScaleScalarPrior(
                scale=10.0,
                lower=2.0,
            ),
        )

    def create_model(self):
        """Build the PyMC graph and add interpretation variables.

        Returns
        -------
        pymc.Model
            Configured model with the Student-t likelihood and deterministic
            interpretation variables.
        """
        model = super().create_model()
        # with the model, add additional deterministic variables in line with the six dimensions.
        with model:
            # Add opening position.
            pm.Deterministic(
                "opening_position",
                model["polynomial_coeffs"][0]
                - pt.sum(
                    model["sinusoidal_amplitude"]
                    * pt.sin(model["sinusoidal_phase"])
                ),
            )
            # Add opening_direction (slope at t=0)
            pm.Deterministic(
                "opening_direction",
                model["polynomial_coeffs"][1]
                + pt.sum(
                    2
                    * np.pi
                    * model["sinusoidal_frequency"]
                    * model["sinusoidal_amplitude"]
                    * pt.cos(model["sinusoidal_phase"])
                ),
            )

            # Find the global minimum and maximum locations analytically on [0, 1].
            linear_intercept = model["polynomial_coeffs"][0]
            slope = model["polynomial_coeffs"][1]
            amplitude = model["sinusoidal_amplitude"][0]
            frequency = model["sinusoidal_frequency"][0]
            phase = model["sinusoidal_phase"][0]
            angular_frequency = 2 * np.pi * frequency
            slope_amplitude = angular_frequency * amplitude
            safe_slope_amplitude = pt.maximum(
                slope_amplitude,
                np.finfo(float).eps,
            )
            stationary_ratio = -slope / safe_slope_amplitude
            has_stationary_points = pt.and_(
                pt.gt(slope_amplitude, 0),
                pt.lt(pt.abs(stationary_ratio), 1),
            )
            stationary_angle = pt.arccos(pt.clip(stationary_ratio, -1, 1))
            safe_angular_frequency = pt.maximum(
                angular_frequency,
                np.finfo(float).eps,
            )

            def stationary_branch_candidates(angle):
                branch_offset = phase + angle
                first_cycle = pt.ceil(-branch_offset / (2 * np.pi))
                last_cycle = pt.floor(
                    (angular_frequency - branch_offset) / (2 * np.pi)
                )
                branch_is_present = pt.and_(
                    has_stationary_points,
                    pt.le(first_cycle, last_cycle),
                )
                first_location = (
                    branch_offset + 2 * np.pi * first_cycle
                ) / safe_angular_frequency
                last_location = (
                    branch_offset + 2 * np.pi * last_cycle
                ) / safe_angular_frequency
                return (
                    pt.stack([first_location, last_location]),
                    pt.stack([branch_is_present, branch_is_present]),
                )

            positive_branch, positive_valid = stationary_branch_candidates(
                stationary_angle
            )
            negative_branch, negative_valid = stationary_branch_candidates(
                -stationary_angle
            )
            candidate_locations = pt.concatenate(
                [
                    pt.as_tensor_variable(pm.floatX([0, 1])),
                    positive_branch,
                    negative_branch,
                ]
            )
            candidate_valid = pt.concatenate(
                [pt.ones(2, dtype="bool"), positive_valid, negative_valid]
            )
            candidate_positions = (
                linear_intercept
                + slope * candidate_locations
                + amplitude
                * pt.sin(angular_frequency * candidate_locations - phase)
            )

            minimum_location, maximum_location = _global_extrema_locations(
                candidate_locations,
                candidate_positions,
                candidate_valid,
            )

            pm.Deterministic("minimum_location", minimum_location)
            pm.Deterministic("maximum_location", maximum_location)

            #For convenience
            pm.Deterministic("linear_trend", model["polynomial_coeffs"][1])

        return model


class QuadraticModel(GenericPolynomialSinusoidalNormalModel):
    """Fit a quadratic trend with a bounded Normal likelihood.

    The latent mean is ``c + m*t + q*t**2``. Default coefficient and likelihood
    priors scale to ``sequence_bounds``.

    Parameters
    ----------
    sequence_bounds
        Finite lower and upper bounds for the observed sequence measure.

    Notes
    -----
    The model records opening position and direction plus global minimum and
    maximum locations on ``[0, 1]``. Endpoints and an in-range vertex are
    considered. ``CurvilinearModel`` is an alias for this class.
    """

    def __init__(self, sequence_bounds: Tuple[float, float] = (0.0, 100.0)):
        """Initialize a quadratic model with sequence-scaled priors."""
        super().__init__(N=2, M=0, sequence_bounds=sequence_bounds)

    def create_model(self):
        """Create the quadratic model and its interpretation variables."""
        model = super().create_model()
        with model:
            coefficients = model["polynomial_coeffs"]
            linear_intercept = coefficients[0]
            slope = coefficients[1]
            curvature = coefficients[2]

            pm.Deterministic("opening_position", linear_intercept)
            pm.Deterministic("opening_direction", slope)

            safe_curvature = pt.switch(pt.eq(curvature, 0), 1, curvature)
            vertex_location = -slope / (2 * safe_curvature)
            vertex_is_valid = pt.and_(
                pt.neq(curvature, 0),
                pt.and_(pt.ge(vertex_location, 0), pt.le(vertex_location, 1)),
            )
            candidate_locations = pt.stack([0.0, 1.0, vertex_location])
            candidate_valid = pt.concatenate(
                [pt.ones(2, dtype="bool"), vertex_is_valid[None]]
            )
            candidate_positions = (
                linear_intercept
                + slope * candidate_locations
                + curvature * candidate_locations**2
            )
            minimum_location, maximum_location = _global_extrema_locations(
                candidate_locations,
                candidate_positions,
                candidate_valid,
            )
            pm.Deterministic("minimum_location", minimum_location)
            pm.Deterministic("maximum_location", maximum_location)

        return model

class LinearModel(GenericPolynomialSinusoidalNormalModel):
    """Fit a linear trend with a bounded Normal likelihood.

    The latent mean is ``c + m*t``. Default coefficient and likelihood priors
    scale to ``sequence_bounds``.

    Parameters
    ----------
    sequence_bounds
        Finite lower and upper bounds for the observed sequence measure.

    Notes
    -----
    The model records opening position and direction. Its global extrema occur
    at the endpoints; when the slope is exactly zero, both locations are zero.
    """

    def __init__(self, sequence_bounds: Tuple[float, float] = (0.0, 100.0)):
        """Initialize a linear model with sequence-scaled priors."""
        super().__init__(N=1, M=0, sequence_bounds=sequence_bounds)

    def create_model(self):
        """Create the linear model and its interpretation variables."""
        model = super().create_model()
        with model:
            coefficients = model["polynomial_coeffs"]
            linear_intercept = coefficients[0]
            slope = coefficients[1]
            zero = pt.as_tensor_variable(pm.floatX(0))
            one = pt.as_tensor_variable(pm.floatX(1))

            pm.Deterministic("opening_position", linear_intercept)
            pm.Deterministic("opening_direction", slope)
            pm.Deterministic(
                "minimum_location",
                pt.switch(pt.lt(slope, 0), one, zero),
            )
            pm.Deterministic(
                "maximum_location",
                pt.switch(pt.gt(slope, 0), one, zero),
            )

        return model

class ConstantModel(GenericPolynomialSinusoidalNormalModel):
    """Fit a constant mean with a bounded Normal likelihood.

    The latent mean is ``c``. The default linear intercept and likelihood priors scale
    to ``sequence_bounds``.

    Parameters
    ----------
    sequence_bounds
        Finite lower and upper bounds for the observed sequence measure.

    Notes
    -----
    The model records opening position and a zero opening direction. Minimum and
    maximum locations are undefined for a constant sequence and are not stored.
    """

    def __init__(self, sequence_bounds: Tuple[float, float] = (0.0, 100.0)):
        """Initialize a constant model with sequence-scaled priors."""
        super().__init__(N=0, M=0, sequence_bounds=sequence_bounds)

    def create_model(self):
        """Create the constant model and its opening interpretation variables."""
        model = super().create_model()
        with model:
            linear_intercept = model["polynomial_coeffs"][0]
            pm.Deterministic("opening_position", linear_intercept)
            pm.Deterministic(
                "opening_direction",
                pt.zeros_like(linear_intercept),
            )

        return model
