import scipy
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import ks_1samp, norm
from functools import partial
from inspect import signature, Parameter

EXAMPLE_SLOPE_VALUE = np.sqrt(2.0)
EXAMPLE_CONSTANT_VALUE = 3.14159
EXAMPLE_SCALE_VALUE = 3.0
EXAMPLE_TRANSLATION_VALUE = 3.0

LOWEST = -2.0
LOW = 0.0
HIGH = 10.0
HIGHEST = 12.0

N = 25
MEAS_NOISE = 0.5
CHEAT = False

x_gridded = np.linspace(LOWEST, HIGHEST, 301)


def reshape_inputs(x):
    if x.ndim == 1:
        return x[:, None]
    if x.ndim == 2:
        return x
    raise ValueError("Unexpected shape")


def distance_matrix(x_i, x_j):
    x_i = np.expand_dims(reshape_inputs(x_i), 1)
    x_j = np.expand_dims(reshape_inputs(x_j), 0)
    return np.linalg.norm(x_i - x_j, axis=-1)


def example_sample_from(mean, cov, size):
    chol = scipy.linalg.cholesky(cov, lower=True)
    return mean[:, None] + np.dot(chol, np.random.normal(size=(cov.shape[0], size)))


def sinc(xs):
    return np.where(xs == 0, np.ones(xs.size), np.sin(xs) / xs)


def truth(xs):
    return EXAMPLE_SCALE_VALUE * sinc(xs - EXAMPLE_TRANSLATION_VALUE)


def generate_training_data(n=N):
    np.random.seed(2012)
    X = np.random.uniform(LOW, HIGH, size=n)
    y = truth(X) + MEAS_NOISE * np.random.normal(size=n)
    return X, y


def plot_smooth_examples(x):
    dists = -np.square(x[:, None] - x[None, :])
    for i in range(50):
        ell = np.random.uniform(1.0, 10.0)
        example_cov = np.exp(dists / np.square(ell)) + 1e-8 * np.eye(x.size)
        s = example_sample_from(np.zeros(example_cov.shape[0]), example_cov, 1)
        plt.plot(x, s, alpha=0.5, color="steelblue")


def example_squared_exponential(x_i, x_j, sigma=1.0, ell=1.0):
    return (
        sigma * sigma * np.exp(-np.square(distance_matrix(x_i, x_j)) / np.square(ell))
    )


def TEST_SQUARED_EXPONENTIAL(f):
    xs = np.linspace(0.0, 1.0, 6)
    expected = example_squared_exponential(xs, xs, sigma=np.pi, ell=np.sqrt(2))
    actual = f(xs, xs, sigma=np.pi, ell=np.sqrt(2))
    diff = np.linalg.norm(expected - actual)
    if diff > 1e-8:
        print("Expected f(xs, xs, np.pi, np.sqrt(2.)) to look like this:")
        print(expected)
        print("But got:")
        print(actual)
        raise ValueError(f"Mismatch between expected and actual : {diff}")


def normalization_test(samples, mean, cov):
    chol = np.linalg.cholesky(cov)
    normalized = np.linalg.solve(chol, samples - mean[:, None])
    ks_statistic = ks_1samp(normalized.reshape(-1), norm.cdf).statistic
    return ks_statistic < 0.05


def TEST_SAMPLE_FROM(f):
    xs = np.linspace(0.0, 10.0, 21)
    mean = np.arange(xs.size)
    cov = example_squared_exponential(
        xs, xs, sigma=np.pi, ell=np.sqrt(2)
    ) + 1e-8 * np.eye(xs.size)

    sample_size = 3
    actual = f(mean, cov, size=sample_size)
    assert actual.shape[0] == mean.size  # output is the wrong shape
    assert actual.shape[1] == sample_size  # output is the wrong shape

    large_sample_size = 10000

    zero_mean = np.zeros(xs.size)
    zero_mean_samples = f(zero_mean, cov, size=large_sample_size)
    if not normalization_test(zero_mean_samples, zero_mean, cov):
        raise ValueError("The samples do not appear to be properly correlated")

    samples = f(mean, cov, size=large_sample_size)
    if not normalization_test(samples, mean, cov):
        raise ValueError(
            "Samples from the function are properly correlated, "
            "but you might have forgotten to add the mean"
        )


def example_fit_and_predict(cov_func, X, y, x_star, meas_noise):
    K_yy = cov_func(X, X) + meas_noise * meas_noise * np.eye(y.size)
    K_sy = cov_func(x_star, X)
    nugget = 1e-12
    n = x_star.shape[0]
    K_ss = cov_func(x_star, x_star) + nugget * np.eye(n)

    mean = np.dot(K_sy, np.linalg.solve(K_yy, y))
    cov = K_ss - np.dot(K_sy, np.linalg.solve(K_yy, K_sy.T))

    return mean, cov


def sinc(xs):
    non_zero = np.nonzero(xs)[0]
    output = np.ones(xs.shape)
    output[non_zero] = np.sin(xs[non_zero]) / xs[non_zero]
    return output


def truth(xs):
    return EXAMPLE_SCALE_VALUE * sinc(xs - EXAMPLE_TRANSLATION_VALUE)


def plot_truth(xs):
    plt.plot(xs, truth(xs), lw=5, color="firebrick", label="truth")


def plot_measurements(xs, ys, color="black", label="measurements"):
    plt.scatter(xs, ys, s=50, color=color, label=label)


def plot_spread(xs, mean, variances, ax=None):
    if ax is None:
        ax = plt.gca()
    xs = np.reshape(xs, -1)
    mean = np.reshape(mean, -1)
    variances = np.reshape(variances, -1)
    sd = np.sqrt(variances)
    ax.plot(xs, mean, lw=5, color="steelblue", label="prediction")
    ax.fill_between(
        xs,
        mean + 2 * sd,
        mean - 2 * sd,
        color="steelblue",
        alpha=0.2,
        label="uncertainty",
    )
    ax.fill_between(
        xs, mean + sd, mean - sd, color="steelblue", alpha=0.5, label="uncertainty"
    )


def TEST_FIT_AND_PREDICT(f):
    xs = np.linspace(0.0, 1.0, 6)
    x_test = np.array([0.3, 0.6])
    ys = np.random.normal(size=xs.size)
    cov_func = partial(example_squared_exponential, sigma=1.2, ell=2.3)
    expected_mean, expected_cov = example_fit_and_predict(
        cov_func, xs, ys, x_test, meas_noise=0.34
    )
    actual_mean, actual_cov = f(cov_func, xs, ys, x_test, meas_noise=0.34)

    if actual_mean.size != expected_mean.size:
        raise ValueError("returned mean was the wrong size")

    if actual_cov.shape != expected_cov.shape:
        raise ValueError("returned covariance was the wrong shape")

    if np.linalg.norm(actual_mean - expected_mean) > 1e-4:
        raise ValueError(
            f"Incorrect mean.\n Expected: f{expected_mean} \n Actual: f{actual_mean}"
        )

    if np.linalg.norm(actual_cov - expected_cov) > 1e-4:
        raise ValueError(
            f"Incorrect covariance [.\n Expected: f{expected_cov} \n Actual: f{actual_cov}"
        )


def example_fit(cov_func, X, y, meas_noise):
    K_yy = cov_func(X, X) + meas_noise * meas_noise * np.eye(y.size)
    L = np.linalg.cholesky(K_yy)
    v = scipy.linalg.cho_solve((L, True), y)

    return {"train_locations": X, "information": v, "cholesky": L, "cov_func": cov_func}


def example_predict(fit_model, x_star):
    cov_func = fit_model["cov_func"]
    v = fit_model["information"]
    L = fit_model["cholesky"]
    X = fit_model["train_locations"]

    K_sy = cov_func(x_star, X)
    K_ss = cov_func(x_star, x_star)
    mean = np.dot(K_sy, v)

    V_ys = scipy.linalg.solve_triangular(L, K_sy.T, lower=True)
    cov = K_ss - np.dot(V_ys.T, V_ys)
    return mean, cov


def TEST_FIT_THEN_PREDICT(fit, pred):
    def combined(cov_func, X, y, x_star, meas_noise):
        return pred(fit(cov_func, X, y, meas_noise), x_star)

    TEST_FIT_AND_PREDICT(combined)
