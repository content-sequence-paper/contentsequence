"""Bayesian estimation of interpretable content-sequence shapes."""

from importlib.metadata import PackageNotFoundError, version

from .base_models import (
    BaseModel,
    GenericPolynomialSinusoidalBaseModel,
    GenericPolynomialSinusoidalModel,
    GenericPolynomialSinusoidalNormalModel,
    LikelihoodModel,
    NormalLikelihoodModel,
    StudentTLikelihoodModel,
)
from .models import ConstantModel, LinearModel, LinearSinusoidalModel, QuadraticModel
from .priors import (
    LocationScaleScalarPrior,
    LocationScaleVectorPrior,
    NormalLikelihoodPrior,
    PolynomialSinusoidalPrior,
    ScaleScalarPrior,
    StudentTLikelihoodPrior,
    VonMisesVectorPrior,
)

CurvilinearModel = QuadraticModel

try:
    __version__ = version("contentsequence")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "BaseModel",
    "ConstantModel",
    "CurvilinearModel",
    "GenericPolynomialSinusoidalBaseModel",
    "GenericPolynomialSinusoidalModel",
    "GenericPolynomialSinusoidalNormalModel",
    "LikelihoodModel",
    "LinearModel",
    "LinearSinusoidalModel",
    "LocationScaleScalarPrior",
    "LocationScaleVectorPrior",
    "NormalLikelihoodModel",
    "NormalLikelihoodPrior",
    "PolynomialSinusoidalPrior",
    "QuadraticModel",
    "ScaleScalarPrior",
    "StudentTLikelihoodModel",
    "StudentTLikelihoodPrior",
    "VonMisesVectorPrior",
    "__version__",
]
