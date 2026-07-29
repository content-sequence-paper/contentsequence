import inspect

import contentsequence as cs


def test_public_api_exports_paper_models_and_prior_types():
    assert cs.__version__ == "0.1.0"
    assert cs.CurvilinearModel is cs.QuadraticModel
    assert issubclass(cs.LinearSinusoidalModel, cs.GenericPolynomialSinusoidalModel)
    assert issubclass(
        cs.GenericPolynomialSinusoidalModel,
        cs.StudentTLikelihoodModel,
    )
    assert issubclass(
        cs.GenericPolynomialSinusoidalModel,
        cs.GenericPolynomialSinusoidalBaseModel,
    )
    assert issubclass(
        cs.LinearModel,
        cs.GenericPolynomialSinusoidalNormalModel,
    )
    assert issubclass(
        cs.GenericPolynomialSinusoidalNormalModel,
        cs.NormalLikelihoodModel,
    )
    assert issubclass(
        cs.GenericPolynomialSinusoidalNormalModel,
        cs.GenericPolynomialSinusoidalBaseModel,
    )
    assert not issubclass(
        cs.GenericPolynomialSinusoidalNormalModel,
        cs.GenericPolynomialSinusoidalModel,
    )
    assert cs.LocationScaleVectorPrior.__module__ == "contentsequence.priors"
    assert cs.NormalLikelihoodPrior.__module__ == "contentsequence.priors"
    assert cs.StudentTLikelihoodPrior.__module__ == "contentsequence.priors"


def test_generic_model_signatures_expose_structural_arguments():
    student_t_parameters = inspect.signature(
        cs.GenericPolynomialSinusoidalModel
    ).parameters
    normal_parameters = inspect.signature(
        cs.GenericPolynomialSinusoidalNormalModel
    ).parameters

    assert list(student_t_parameters) == [
        "N",
        "M",
        "sequence_bounds",
        "likelihood_priors",
    ]
    assert list(normal_parameters) == [
        "N",
        "M",
        "sequence_bounds",
        "likelihood_priors",
    ]


def test_linear_intercept_prior_can_be_changed_directly():
    model = cs.LinearSinusoidalModel()

    model.priors.linear_intercept.mu = 10
    model.priors.linear_intercept.sigma = 1

    assert model.priors.linear_intercept.mu == 10
    assert model.priors.linear_intercept.sigma == 1
    assert model.priors.polynomial_coeffs.mu[0] == 10
    assert model.priors.polynomial_coeffs.sigma[0] == 1
