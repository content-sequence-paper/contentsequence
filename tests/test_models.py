import arviz as az
import numpy as np
import pymc as pm
import pytest

import contentsequence as cs


@pytest.mark.parametrize(
    ("model_class", "degree", "harmonics"),
    [
        (cs.LinearSinusoidalModel, 1, 1),
        (cs.QuadraticModel, 2, 0),
        (cs.LinearModel, 1, 0),
        (cs.ConstantModel, 0, 0),
    ],
)
def test_simplified_models_build_with_expected_shapes(model_class, degree, harmonics):
    model = model_class()
    model.time_data = np.linspace(0.0, 1.0, 5)
    model.sequence_data = np.linspace(40.0, 60.0, 5)

    pymc_model = model.create_model()

    assert pymc_model.named_vars["polynomial_coeffs"].type.shape == (degree + 1,)
    assert pymc_model.named_vars["mu"].type.shape == (5,)
    if harmonics:
        assert pymc_model.named_vars["sinusoidal_amplitude"].type.shape == (harmonics,)
        assert pymc_model.named_vars["sinusoidal_frequency"].type.shape == (harmonics,)
        assert pymc_model.named_vars["sinusoidal_phase"].type.shape == (harmonics,)
    else:
        assert "sinusoidal_amplitude" not in pymc_model.named_vars


def test_student_t_model_owns_student_t_likelihood_parameters():
    model = cs.LinearSinusoidalModel()
    model.time_data = np.linspace(0.0, 1.0, 5)
    model.sequence_data = np.linspace(40.0, 60.0, 5)

    pymc_model = model.create_model()

    assert isinstance(model, cs.StudentTLikelihoodModel)
    assert isinstance(model.likelihood_priors, cs.StudentTLikelihoodPrior)
    assert "sigma" in pymc_model.named_vars
    assert "dof" in pymc_model.named_vars
    assert "dof_unconstrained" in pymc_model.named_vars
    assert not hasattr(model, "likelihood_generator")


@pytest.mark.parametrize(
    "model_class",
    [cs.ConstantModel, cs.LinearModel, cs.QuadraticModel],
)
def test_normal_models_do_not_create_student_t_parameters(model_class):
    model = model_class()
    model.time_data = np.linspace(0.0, 1.0, 5)
    model.sequence_data = np.linspace(40.0, 60.0, 5)

    pymc_model = model.create_model()

    assert isinstance(model, cs.NormalLikelihoodModel)
    assert isinstance(model.likelihood_priors, cs.NormalLikelihoodPrior)
    assert not hasattr(model.likelihood_priors, "dof")
    assert "sigma" in pymc_model.named_vars
    assert "dof" not in pymc_model.named_vars
    assert "dof_unconstrained" not in pymc_model.named_vars
    assert not hasattr(model, "likelihood_generator")


def test_default_priors_scale_to_sequence_bounds():
    model = cs.LinearModel(sequence_bounds=(-20.0, 80.0))

    assert model.priors.polynomial_coeffs.mu == [30.0, 0.0]
    assert model.priors.polynomial_coeffs.sigma == [25.0, 50.0]
    assert model.likelihood_priors.sigma.mu == 0.0
    assert model.likelihood_priors.sigma.sigma == 10.0


def test_linear_sinusoidal_defaults_cover_sequence_range():
    model = cs.LinearSinusoidalModel(sequence_bounds=(0.0, 100.0))

    assert model.priors.polynomial_coeffs.mu == [50.0, 0.0]
    assert model.priors.polynomial_coeffs.sigma == [25.0, 25.0]
    assert model.priors.sinusoidal_amplitude.mu == 10.0
    assert model.priors.sinusoidal_amplitude.sigma == 5.0
    assert model.likelihood_priors.sigma.sigma == 10.0
    assert model.likelihood_priors.dof.scale == 10.0
    assert model.likelihood_priors.dof.lower == 2.0


def test_student_t_dof_prior_respects_optional_upper_bound():
    model = cs.GenericPolynomialSinusoidalModel(
        N=0,
        M=0,
        likelihood_priors=cs.StudentTLikelihoodPrior(
            sigma=cs.LocationScaleScalarPrior(mu=0.0, sigma=10.0, lower=0.0),
            dof=cs.ScaleScalarPrior(scale=10.0, lower=2.0, upper=8.0),
        ),
    )
    model.time_data = np.linspace(0.0, 1.0, 5)
    model.sequence_data = np.linspace(40.0, 60.0, 5)

    idata = model.sample_prior_predictive(
        draws=50,
        var_names=["dof"],
        random_seed=42,
    )

    assert np.all(idata.prior["dof"].values >= 2.0)
    assert np.all(idata.prior["dof"].values <= 8.0)


@pytest.mark.parametrize(
    "sequence_bounds",
    [(0.0, 0.0), (1.0, 0.0), (0.0, np.inf), (0.0, 1.0, 2.0)],
)
def test_sequence_bounds_must_be_finite_and_increasing(sequence_bounds):
    with pytest.raises(ValueError):
        cs.ConstantModel(sequence_bounds=sequence_bounds)


@pytest.mark.parametrize(
    ("attribute", "values", "message"),
    [
        ("time_data", [[0.0, 1.0]], "one-dimensional"),
        ("time_data", [0.0, np.nan], "finite"),
        ("sequence_data", [[40.0, 50.0]], "one-dimensional"),
        ("sequence_data", [40.0, np.inf], "finite"),
        ("sequence_data", [40.0, 101.0], "sequence_bounds"),
    ],
)
def test_sequence_inputs_validate_shape_values_and_bounds(attribute, values, message):
    model = cs.ConstantModel()

    with pytest.raises(ValueError, match=message):
        setattr(model, attribute, values)


def test_linear_sinusoidal_prior_predictive_matches_observation_length():
    model = cs.LinearSinusoidalModel()
    model.time_data = np.linspace(0.0, 1.0, 7)
    model.sequence_data = np.linspace(40.0, 60.0, 7)

    idata = model.sample_prior_predictive(draws=2, random_seed=42)

    assert idata.prior["mu"].shape == (1, 2, 7)
    assert idata.prior_predictive["observed"].shape == (1, 2, 7)


def test_linear_sinusoidal_opening_position_is_latent_mean_at_zero():
    model = cs.LinearSinusoidalModel()
    model.time_data = np.linspace(0.0, 1.0, 7)
    model.sequence_data = np.linspace(40.0, 60.0, 7)

    idata = model.sample_prior_predictive(draws=5, random_seed=42)

    expected = (
        idata.prior["polynomial_coeffs"].values[..., 0]
        - (
            idata.prior["sinusoidal_amplitude"].values
            * np.sin(idata.prior["sinusoidal_phase"].values)
        ).sum(axis=-1)
    )
    np.testing.assert_allclose(idata.prior["opening_position"].values, expected)
    np.testing.assert_allclose(
        idata.prior["opening_position"].values,
        idata.prior["mu"].values[..., 0],
    )


def test_linear_sinusoidal_opening_direction_is_slope_at_zero():
    model = cs.LinearSinusoidalModel()
    model.time_data = np.linspace(0.0, 1.0, 7)
    model.sequence_data = np.linspace(40.0, 60.0, 7)

    idata = model.sample_prior_predictive(draws=5, random_seed=42)

    expected = (
        idata.prior["polynomial_coeffs"].values[..., 1]
        + (
            2
            * np.pi
            * idata.prior["sinusoidal_frequency"].values
            * idata.prior["sinusoidal_amplitude"].values
            * np.cos(idata.prior["sinusoidal_phase"].values)
        ).sum(axis=-1)
    )
    np.testing.assert_allclose(idata.prior["opening_direction"].values, expected)


def test_linear_sinusoidal_extrema_locations_include_turning_points_and_endpoints():
    model = cs.LinearSinusoidalModel()
    model.time_data = np.linspace(0.0, 1.0, 7)
    model.sequence_data = np.linspace(40.0, 60.0, 7)

    idata = model.sample_prior_predictive(draws=10, random_seed=42)

    coefficients = idata.prior["polynomial_coeffs"].values[0]
    amplitudes = idata.prior["sinusoidal_amplitude"].values[0, :, 0]
    frequencies = idata.prior["sinusoidal_frequency"].values[0, :, 0]
    phases = idata.prior["sinusoidal_phase"].values[0, :, 0]
    expected_locations = []

    for coefficients_draw, amplitude, frequency, phase in zip(
        coefficients,
        amplitudes,
        frequencies,
        phases,
        strict=True,
    ):
        linear_intercept, slope = coefficients_draw
        angular_frequency = 2 * np.pi * frequency
        ratio = -slope / (angular_frequency * amplitude)
        candidates = [0.0, 1.0]

        if abs(ratio) < 1:
            stationary_angle = np.arccos(ratio)
            for angle in (stationary_angle, -stationary_angle):
                first_cycle = int(np.ceil((-phase - angle) / (2 * np.pi)))
                last_cycle = int(
                    np.floor((angular_frequency - phase - angle) / (2 * np.pi))
                )
                candidates.extend(
                    (phase + angle + 2 * np.pi * cycle) / angular_frequency
                    for cycle in range(first_cycle, last_cycle + 1)
                )

        candidates = np.asarray(candidates)
        positions = (
            linear_intercept
            + slope * candidates
            + amplitude * np.sin(angular_frequency * candidates - phase)
        )
        minimum = positions.min()
        maximum = positions.max()
        expected_locations.append(
            [
                candidates[positions == minimum].min(),
                candidates[positions == maximum].min(),
            ]
        )

    np.testing.assert_allclose(
        idata.prior["minimum_location"].values[0],
        np.asarray(expected_locations)[:, 0],
    )
    np.testing.assert_allclose(
        idata.prior["maximum_location"].values[0],
        np.asarray(expected_locations)[:, 1],
    )


@pytest.mark.parametrize(
    ("model_class", "opening_direction_index"),
    [
        (cs.ConstantModel, None),
        (cs.LinearModel, 1),
        (cs.QuadraticModel, 1),
    ],
)
def test_polynomial_models_expose_opening_interpretation_variables(
    model_class,
    opening_direction_index,
):
    model = model_class()
    model.time_data = np.linspace(0.0, 1.0, 7)
    model.sequence_data = np.linspace(40.0, 60.0, 7)

    idata = model.sample_prior_predictive(draws=5, random_seed=42)

    coefficients = idata.prior["polynomial_coeffs"].values
    np.testing.assert_allclose(
        idata.prior["opening_position"].values,
        coefficients[..., 0],
    )
    expected_direction = (
        np.zeros(coefficients.shape[:-1])
        if opening_direction_index is None
        else coefficients[..., opening_direction_index]
    )
    np.testing.assert_allclose(
        idata.prior["opening_direction"].values,
        expected_direction,
    )


def test_constant_model_does_not_record_extrema_locations():
    model = cs.ConstantModel()
    model.time_data = np.linspace(0.0, 1.0, 7)
    model.sequence_data = np.linspace(40.0, 60.0, 7)

    idata = model.sample_prior_predictive(draws=2, random_seed=42)

    assert "minimum_location" not in idata.prior
    assert "maximum_location" not in idata.prior


def test_linear_model_extrema_locations_follow_the_slope():
    model = cs.LinearModel()
    model.time_data = np.linspace(0.0, 1.0, 7)
    model.sequence_data = np.linspace(40.0, 60.0, 7)

    idata = model.sample_prior_predictive(draws=20, random_seed=42)

    slopes = idata.prior["polynomial_coeffs"].values[..., 1]
    np.testing.assert_allclose(
        idata.prior["minimum_location"].values,
        np.where(slopes < 0, 1.0, 0.0),
    )
    np.testing.assert_allclose(
        idata.prior["maximum_location"].values,
        np.where(slopes > 0, 1.0, 0.0),
    )


def test_quadratic_model_extrema_locations_include_vertex_and_endpoints():
    model = cs.QuadraticModel()
    model.time_data = np.linspace(0.0, 1.0, 7)
    model.sequence_data = np.linspace(40.0, 60.0, 7)

    idata = model.sample_prior_predictive(draws=20, random_seed=42)

    expected_minimum = []
    expected_maximum = []
    for linear_intercept, slope, curvature in idata.prior["polynomial_coeffs"].values[0]:
        candidates = [0.0, 1.0]
        if curvature != 0:
            vertex = -slope / (2 * curvature)
            if 0 <= vertex <= 1:
                candidates.append(vertex)
        candidates = np.asarray(candidates)
        positions = linear_intercept + slope * candidates + curvature * candidates**2
        expected_minimum.append(candidates[positions == positions.min()].min())
        expected_maximum.append(candidates[positions == positions.max()].min())

    np.testing.assert_allclose(
        idata.prior["minimum_location"].values[0],
        expected_minimum,
    )
    np.testing.assert_allclose(
        idata.prior["maximum_location"].values[0],
        expected_maximum,
    )


def test_unbounded_polynomial_prior_does_not_create_an_interval_transform():
    model = cs.LinearSinusoidalModel()
    model.time_data = np.linspace(0.0, 1.0, 5)
    model.sequence_data = np.linspace(40.0, 60.0, 5)

    pymc_model = model.create_model()

    initial_point = pymc_model.initial_point()
    assert "polynomial_coeffs" in initial_point
    assert "polynomial_coeffs_interval__" not in initial_point


def test_posterior_predictive_defaults_to_accumulated_idata(monkeypatch):
    model = cs.LinearModel()
    model.model = pm.Model()
    model.idata = az.from_dict(posterior={"x": np.zeros((1, 2))})
    captured = {}

    def fake_sample_posterior_predictive(trace, *args, **kwargs):
        captured["trace"] = trace
        return az.from_dict(posterior_predictive={"observed": np.zeros((1, 2, 3))})

    monkeypatch.setattr(pm, "sample_posterior_predictive", fake_sample_posterior_predictive)

    model.sample_posterior_predictive()

    assert captured["trace"] is model.idata
    assert "posterior_predictive" in model.idata.groups()


def test_fit_does_not_pass_a_draw_count_as_the_posterior_trace(monkeypatch):
    model = cs.LinearModel()
    calls = []

    monkeypatch.setattr(
        model,
        "sample_prior_predictive",
        lambda *args, **kwargs: calls.append(("prior", args, kwargs)),
    )
    monkeypatch.setattr(
        model,
        "sample",
        lambda *args, **kwargs: calls.append(("posterior", args, kwargs)),
    )
    monkeypatch.setattr(
        model,
        "sample_posterior_predictive",
        lambda *args, **kwargs: calls.append(("predictive", args, kwargs)),
    )

    assert model.fit(random_seed=42) is model.idata
    assert calls[-1] == (
        "predictive",
        (),
        {"random_seed": 42, "progressbar": True},
    )
    assert calls[-2][2]["progressbar"] is True
    assert calls[-2][2]["init"] == "adapt_diag"
