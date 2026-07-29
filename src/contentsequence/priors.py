"""Prior configuration objects for content-sequence models."""

from dataclasses import dataclass


class _VectorElementPrior:
    """Mutable scalar view of one element in a vector prior."""

    def __init__(self, prior, index):
        self._prior = prior
        self._index = index

    def _get(self, name):
        value = getattr(self._prior, name)
        if isinstance(value, (list, tuple)):
            return value[self._index]
        return value

    def _set(self, name, value):
        current = getattr(self._prior, name)
        if isinstance(current, list):
            current[self._index] = value
        elif isinstance(current, tuple):
            updated = list(current)
            updated[self._index] = value
            setattr(self._prior, name, updated)
        elif self._index == 0:
            setattr(self._prior, name, value)
        else:
            raise ValueError(f"`{name}` must be a list to edit one vector element.")

    @property
    def mu(self):
        """Location of this prior element."""
        return self._get("mu")

    @mu.setter
    def mu(self, value):
        self._set("mu", value)

    @property
    def sigma(self):
        """Standard deviation of this prior element."""
        return self._get("sigma")

    @sigma.setter
    def sigma(self, value):
        self._set("sigma", value)


@dataclass
class LocationScaleScalarPrior:
    """Parameters for a scalar Normal or truncated-Normal prior.

    Parameters
    ----------
    mu
        Location of the underlying Normal distribution.
    sigma
        Positive standard deviation of the underlying Normal distribution.
    lower, upper
        Optional support limits. If either limit is supplied, the package uses a
        truncated-Normal distribution.
    """

    mu: float
    sigma: float
    lower: float | None = None
    upper: float | None = None


@dataclass
class ScaleScalarPrior:
    """Parameters for a shifted, optionally bounded Exponential prior.

    This prior is used for the Student-t degrees of freedom. ``scale`` is the
    mean Exponential excess above ``lower``; equivalently, its rate is
    ``1 / scale``.

    Parameters
    ----------
    scale
        Positive Exponential scale.
    lower
        Optional shift applied to the Exponential draw.
    upper
        Optional upper bound on the shifted draw.
    """

    scale: float
    lower: float | None = None
    upper: float | None = None


@dataclass
class LocationScaleVectorPrior:
    """Parameters for vector Normal or truncated-Normal priors.

    Scalar values are broadcast across the requested vector. Lists specify one
    value per coefficient or harmonic.

    Parameters
    ----------
    mu, sigma
        Location and positive standard deviation values.
    lower, upper
        Optional support limits. Lists must be compatible with the vector size.
    """

    mu: list[float] | float
    sigma: list[float] | float
    lower: list[float] | float | None = None
    upper: list[float] | float | None = None


@dataclass
class VonMisesVectorPrior:
    """Parameters for circular phase priors.

    Parameters
    ----------
    mu
        Circular location in radians.
    kappa
        Positive concentration. Values near zero are diffuse; larger values
        concentrate phase around ``mu``.
    """

    mu: list[float] | float
    kappa: list[float] | float


@dataclass
class PolynomialSinusoidalPrior:
    """Prior configuration for the polynomial-sinusoidal mean function.

    Likelihood-specific priors live in :class:`NormalLikelihoodPrior` or
    :class:`StudentTLikelihoodPrior` so that the structural model does not
    acquire parameters that its likelihood does not use.

    Parameters
    ----------
    polynomial_coeffs
        Priors ordered from the linear intercept through polynomial degree ``N``.
    sinusoidal_amplitude
        Non-negative amplitude priors, one per harmonic.
    sinusoidal_frequency
        Non-negative frequency priors in cycles per normalized sequence.
    sinusoidal_phase
        Circular phase priors in radians, one per harmonic.
    """

    polynomial_coeffs: LocationScaleVectorPrior
    sinusoidal_amplitude: LocationScaleVectorPrior
    sinusoidal_frequency: LocationScaleVectorPrior
    sinusoidal_phase: VonMisesVectorPrior

    @property
    def linear_intercept(self):
        """Mutable prior for the linear intercept coefficient.

        This convenience view supports direct changes such as
        ``model.priors.linear_intercept.mu = 10``.
        """
        return _VectorElementPrior(self.polynomial_coeffs, 0)


@dataclass
class NormalLikelihoodPrior:
    """Prior configuration for a Normal observation likelihood.

    Parameters
    ----------
    sigma
        Prior for the positive observation standard deviation.
    """

    sigma: LocationScaleScalarPrior


@dataclass
class StudentTLikelihoodPrior(NormalLikelihoodPrior):
    """Prior configuration for a Student-t observation likelihood.

    Parameters
    ----------
    sigma
        Prior for the positive observation scale.
    dof
        Shifted Exponential prior for the degrees of freedom.
    """

    dof: ScaleScalarPrior
