# Gaussian Mixture Model with EM from Scratch

A from-scratch implementation of a Gaussian Mixture Model (GMM) trained with the Expectation-Maximization (EM) algorithm.

This project originated as a university team assignment using a two-dimensional FAA AEDT dataset. The core GMM and EM algorithm were implemented without using a machine-learning library for model fitting.

## Project Overview

The model estimates a mixture of multivariate Gaussian distributions by iteratively applying:

1. **E-step**
   Computes the posterior responsibility of each Gaussian component for every sample.

2. **M-step**
   Updates the mixture weights, component means, and full covariance matrices using the responsibilities.

3. **Convergence check**
   Stops when the absolute change in log-likelihood falls below a configurable threshold or when the maximum number of iterations is reached.

The implementation also computes the Bayesian Information Criterion (BIC) for model selection.

## Coursework Experiment

The original experiment evaluated models with 1 through 5 Gaussian components on a dataset with 1,000 samples and two features.

| Components | Iterations | BIC |
| ---: | ---: | ---: |
| 1 | 2 | 10813.20 |
| 2 | 20 | 10389.95 |
| 3 | 27 | **10028.97** |
| 4 | 21 | 10066.04 |
| 5 | 36 | 10105.23 |

Among the evaluated models, the 3-component model produced the lowest BIC.

The implementation uses a fixed default random seed (`random_state=42`) so the original experiment is reproducible.

## Implementation Details

`GaussianMixture` supports:

- arbitrary sample counts and feature dimensions
- full covariance matrices
- configurable convergence tolerance
- configurable maximum iteration count
- configurable covariance regularization
- deterministic initialization through a local random number generator
- NumPy arrays and pandas DataFrames
- log-likelihood evaluation
- BIC calculation
- cluster prediction after fitting

For numerical stability, Gaussian probabilities are evaluated in log space and normalized using a log-sum-exp calculation. Covariance regularization is also applied to reduce failures on degenerate or nearly singular data.

## Repository Structure

~~~text
.
├── src/
│   ├── __init__.py
│   ├── main.py
│   └── model.py
├── tests/
│   └── test_model.py
├── .gitignore
├── README.md
└── requirements.txt
~~~

- `src/model.py` — from-scratch GMM and EM implementation
- `src/main.py` — original K=1 through K=5 clustering experiment and visualization
- `tests/test_model.py` — self-contained regression tests

## Installation

Tested with Python 3.13.3.

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

## Running the Tests

The test suite does not require the coursework dataset.

~~~bash
python -m unittest discover -s tests -p 'test_*.py' -v
~~~

The regression tests cover:

- deterministic fitting with a fixed random seed
- isolation from NumPy's global random state
- equivalent DataFrame and ndarray behavior
- feature-dimension validation
- covariance regularization on degenerate data
- stable inference for extremely distant samples
- full-covariance BIC parameter counting
- inference-before-fit validation

## Reproducing the Coursework Experiment

The original FAA AEDT CSV is not included in this repository.

To reproduce the experiment, place the dataset at:

~~~text
FAA_AEDT_data.csv
~~~

in the project root. The expected columns are:

~~~text
x1,x2
~~~

Then run:

~~~bash
python -m src.main
~~~

This fits models with 1 through 5 components and displays the resulting clustering plots together with their BIC values.

## Project Contribution

My primary contribution to the original team project was implementing the Gaussian Mixture Model and EM algorithm from scratch.

I also proposed using both:

- a maximum iteration limit, and
- a log-likelihood convergence threshold

to provide a bounded and practical stopping criterion for EM.

During analysis of the iterative optimization process, I examined how the log-likelihood evolved across iterations. Additional robustness and reproducibility checks were later added while preparing the implementation as a standalone software project.

## Notes on EM Optimization

For the default 3-component experiment, the observed log-likelihood increased monotonically during training, including a slower-improvement region followed by larger gains later in the optimization.

Separate initialization experiments also showed that EM can converge to different local optima depending on the initial component means. This is an expected property of EM and is one reason initialization can materially affect GMM solutions.

## Limitations

- The coursework experiment uses a two-dimensional dataset even though the model implementation supports higher-dimensional inputs.
- The default implementation performs a single initialization rather than multiple restarts.
- The EM implementation prioritizes clarity and a direct from-scratch formulation rather than maximum computational performance.
- The original coursework dataset is not redistributed with this repository.
