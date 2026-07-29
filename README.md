# contentsequence

`contentsequence` is a Python package for measuring content sequences in unstructured data. Developed alongside *The Shape of Content: Toward an Anatomy of Content Sequences in Unstructured Data Analysis*, it implements a Bayesian linear–sinusoidal model that estimates six dimensions of a sequence: opening position, opening direction, trend, frequency of peaks and valleys, amplitude of peaks and valleys, and the position of maxima and minima.

The package is modality-independent and uses [PyMC](https://www.pymc.io/) for
Bayesian inference and [ArviZ](https://python.arviz.org/) for diagnostics and model comparison.

A full tutorial and API documentation is available at [https://content-sequence-paper.github.io/contentsequence/](https://content-sequence-paper.github.io/contentsequence/)

## Installation
1. Set up `python` (minimum python 3.11). We recommend using an environment manager such as `conda` or `uv`.
2. Download or clone this repository:
    - To download: https://github.com/content-sequence-paper/contentsequence/archive/refs/heads/main.zip
    - To clone: from your terminal, run 
        ```bash
        git clone git@github.com:content-sequence-paper/contentsequence.git
        ```
3. Change directory into the downloaded folder: 
    ```bash
    cd contentsequence
    ```
4. Install the package. Two options:
  - If you intend to run the notebook example, run:
    ```bash
    python -m pip install -e ".[notebook]"
    ```
  - If you want a minimal install, e.g., to run the package with your own Python codes, run:
    ```bash
    python -m pip install -e .
    ```

## Quick start

An example notebook analyzing a content sequence is provided in `example/sequence-analysis.ipynb`.
Example data is also provided in `example/8883_emo_sequence.csv`.


An example analysis code is below:
```python
import pandas as pd
import contentsequence as cs

# Import the supplied example data
data = pd.read_csv("example/8883_emo_sequence.csv")

# Construct the model
model = cs.LinearSinusoidalModel()
model.time_data = data.time
model.sequence_data = data.emo

idata = model.fit()
```

`fit()` returns an `arviz.InferenceData` object containing prior predictive,
posterior, posterior predictive, sample statistics, pointwise log likelihood,
and observed-data groups.

## Data requirements

- `time_data` must be a non-empty, finite, one-dimensional array.
- Locations must be strictly increasing and contained in `[0, 1]`.
- `sequence_data` must be finite, one-dimensional, and the same length as `time_data`.
- Every observation must lie inside `sequence_bounds`.
- `sequence_bounds=(lower, upper)` bounds the lower and upper bounds of the sequence data.

Set both arrays before calling `create_model()`, a sampling method, or `fit()`.

## Models and outputs

| Model | Latent mean | Likelihood | Interpretation variables |
|---|---|---|---|
| `cs.ConstantModel` | `c` | Bounded Normal | `opening_position`, `opening_direction` |
| `cs.LinearModel` | `c + m t` | Bounded Normal | Opening variables, `minimum_location`, `maximum_location` |
| `cs.QuadraticModel` | `c + m t + q t²` | Bounded Normal | Opening variables, `minimum_location`, `maximum_location` |
| `cs.CurvilinearModel` | Alias for `QuadraticModel` | Bounded Normal | Same as `QuadraticModel` |
| `cs.LinearSinusoidalModel` | `c + m t + a sin(2π f t - u)` | Bounded Student-t | Opening variables, extrema locations, `linear_trend` |

All models record `mu`, the latent mean at each supplied location, and `sigma`.
The Student-t model additionally records `dof`. Constant sequences have no
unique extrema locations, so the constant model deliberately omits them.

The generic classes support other polynomial degrees and harmonic counts:

```python
student_model = cs.GenericPolynomialSinusoidalModel(N=2, M=2)
normal_model = cs.GenericPolynomialSinusoidalNormalModel(N=2, M=2)
```

Generic classes build the requested latent mean but do not add the paper-specific
interpretation variables.

## Sequence-scaled default priors

Defaults adapt to `sequence_bounds`. Let `R = upper - lower` and let `C` be the
range midpoint. The generic polynomial models use:

- Linear intercept: `Normal(C, R / 4)`
- Remaining polynomial coefficients: `Normal(0, R / 2)`
- Observation scale: `HalfNormal(R / 10)`

`LinearSinusoidalModel` uses `Normal(C, R / 4)` for its linear intercept,
`Normal(0, R / 4)` for its linear trend, a positive
`TruncatedNormal(R / 10, R / 20)` amplitude, and the paper-oriented frequency
and phase priors. Its degrees of freedom are `2 + Exponential(scale=10)`.

These priors are sensible starting points that cover the range of the data. They are not substitutes for a proper prior predictive check.

## Changing a prior

Change an individual prior value directly before fitting:

```python
model.priors.linear_intercept.mu = 10
model.priors.linear_intercept.sigma = 1
```

The effects of priors can be analyzed using `arviz`

```
import arviz as az

model.sample_prior_predictive()
az.plot_ppc(model.idata, group="prior")
```

## Sampling and diagnostics

`fit()` runs prior predictive sampling, posterior sampling, and posterior
predictive sampling. For individual stages, use:

```python
model.sample_prior_predictive()
model.sample()
model.sample_posterior_predictive()
```

Check trace plots, rank-normalized R-hat, effective sample sizes, divergences,
and posterior predictive fit before interpreting a model. Compare fitted models
with `arviz.compare` only after each model has satisfactory diagnostics and
plausible prior and posterior predictive behavior.

## Examples and full documentation

The analysis notebook is in [`example/sequence-analysis.ipynb`](example/sequence-analysis.ipynb). It covers
changing a prior, fitting, diagnostics, posterior predictive HDIs, and LOO model
comparison.

Sphinx documentation source is in `docs/`. Build it locally with:

```bash
python -m pip install -e ".[docs]"
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

The resulting `docs/_build/html/index.html` directory is ready to publish with GitHub Pages.

## Development

```bash
python -m pip install -e ".[dev,notebook,docs]"
python -m pytest
python -m build
```


## Citation

Pending review.


## License

MIT. See [`LICENSE`](LICENSE).
