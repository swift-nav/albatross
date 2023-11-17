import scipy
import numpy as np
import matplotlib.pyplot as plt
import timeit

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
    x = np.array(x)
    if x.ndim == 0:
        return np.atleast_2d(x)
    if x.ndim == 1:
        return x[:, None]
    if x.ndim == 2:
        return x
    raise ValueError("Unexpected shape", x.shape)


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
        sigma
        * sigma
        * np.exp(-0.5 * np.square(distance_matrix(x_i, x_j)) / np.square(ell))
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


def plot_truth(xs, **kwdargs):
    return plt.plot(xs, truth(xs), lw=5, color="firebrick", label="truth", **kwdargs)[0]


def plot_measurements(xs, ys, color="black", label="measurements", **kwdargs):
    plt.scatter(xs, ys, s=50, color=color, label=label, **kwdargs)


def plot_spread(xs, mean, variances, ax=None, color="steelblue"):
    if ax is None:
        ax = plt.gca()
    xs = np.reshape(xs, -1)
    mean = np.reshape(mean, -1)
    variances = np.reshape(variances, -1)
    sd = np.sqrt(variances)
    line = ax.plot(xs, mean, lw=5, color=color, label="prediction")
    ax.fill_between(
        xs,
        mean + 2 * sd,
        mean - 2 * sd,
        color=color,
        alpha=0.2,
        label="uncertainty",
    )
    ax.fill_between(
        xs, mean + sd, mean - sd, color=color, alpha=0.5, label="uncertainty"
    )
    return line[0]


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


def assert_functions_close(reference, candidate, *args, threshold=1e-4):
    expected = reference(*args)
    actual = candidate(*args)
    if isinstance(expected, tuple):
        abs_diff = max(np.max(np.abs(a - b)) for a, b in zip(expected, actual))
    else:
        abs_diff = np.abs(expected - actual)
    if abs_diff > threshold:
        raise ValueError(f"Expected: {expected} but got {actual}")
    print("Good Job!")
    return abs_diff <= threshold


def example_compute_independent_likelihood(dist, data):
    return np.prod(dist.pdf(data))


def TEST_COMPUTE_INDEPENDENT_LIKELIHOOD(f):
    dist = scipy.stats.norm(loc=1.1, scale=2.1)
    assert_functions_close(
        example_compute_independent_likelihood, f, dist, dist.rvs(size=10)
    )


def example_compute_independent_log_likelihood(dist, data):
    return np.sum(dist.logpdf(data))


def TEST_COMPUTE_INDEPENDENT_LOG_LIKELIHOOD(f):
    dist = scipy.stats.norm(loc=1.1, scale=2.1)
    assert_functions_close(
        example_compute_independent_log_likelihood, f, dist, dist.rvs(size=10)
    )


def example_compute_independent_negative_log_likelihood(params, ys):
    mu, sigma = params
    return -example_compute_independent_log_likelihood(
        scipy.stats.norm(loc=mu, scale=sigma), ys
    )


def TEST_COMPUTE_INDEPENDENT_NEGATIVE_LOG_LIKELIHOOD(f):
    y = np.array(
        [
            -0.02415509423853975,
            -2.0601745974286185,
            -0.6431114465594998,
            -0.21516292936011427,
            -1.6847658704470119,
        ]
    )
    params = [-1.1, 2.1]

    assert_functions_close(
        example_compute_independent_negative_log_likelihood, f, params, y
    )


def example_compute_mvn_log_likelihood(S, y):
    L = np.linalg.cholesky(S + 1e-12 * np.eye(y.size))
    _, sqrt_log_det = np.linalg.slogdet(L)
    Li_y = scipy.linalg.solve_triangular(L, y, lower=True)
    decorrelated_ll = 0.5 * Li_y.T @ Li_y
    constant = y.size * 0.5 * np.log(2 * np.pi)
    return -decorrelated_ll - sqrt_log_det - constant


def TEST_COMPUTE_MVN_LOG_LIKELIHOOD(f):
    S = np.array(
        [
            [
                1.3856753684810348,
                -0.948119667263291,
                0.07339153686581486,
                -1.346752085609235,
                1.0588760336946892,
            ],
            [
                -0.948119667263291,
                4.118596313893358,
                -2.1950866239854387,
                1.3713638488445923,
                -0.8986971493288571,
            ],
            [
                0.07339153686581486,
                -2.1950866239854387,
                6.107252930579301,
                1.114190291633452,
                -1.0165191212603695,
            ],
            [
                -1.346752085609235,
                1.3713638488445923,
                1.114190291633452,
                4.778341927916547,
                -1.8550052285982568,
            ],
            [
                1.0588760336946892,
                -0.8986971493288571,
                -1.0165191212603695,
                -1.8550052285982568,
                1.9683792410128251,
            ],
        ]
    )
    S = 0.5 * (S + S.T)

    mvn_dist = scipy.stats.multivariate_normal(mean=np.zeros(S.shape[0]), cov=S)
    samp = np.array(
        [
            -0.02415509423853975,
            -2.0601745974286185,
            -0.6431114465594998,
            -0.21516292936011427,
            -1.6847658704470119,
        ]
    )

    ll = f(S, samp)
    expected = mvn_dist.logpdf(samp)
    abs_diff = np.abs(expected - ll)

    if np.abs(expected + ll) <= 1e-4:
        raise ValueError("You forgot the negative sign(s)!")

    ll_without_errors = 3.9992009800530663
    if np.abs(abs_diff - ll_without_errors) <= 1e-4:
        raise ValueError(
            f"It appears you've left out the squared error term:\n\n -0.5 * y.T S^-1 y\n\n"
        )

    ll_without_log_deg = 1.994231445949179
    if np.abs(abs_diff - ll_without_log_deg) <= 1e-4:
        raise ValueError(
            f"It appears you've left out the log det term:\n\n -0.5 * log(det(S))\n\n"
        )

    ll_without_constant = 4.594692666023871
    if np.abs(abs_diff - ll_without_constant) <= 1e-4:
        print(
            f"It appears you've left out the constant term:\n\n -n/2 log(2 pi)\n\n"
            "...which is actually OK! Just about anything you'd want to do "
            "with a likelihood can be down up to a constant."
        )
    elif abs_diff > 1e-4:
        print(f"Expected LL: {ll}  Actual {expected}")
        raise ValueError(
            "Ooops, your likelihood for an example doesn't match expectations"
        )

    if abs_diff == 0.0:
        print(f"You used scipy's multivariate_normal didn't you :)")


def example_compute_sqr_exp_negative_log_likelihood(params, X, y):
    sigma, ell, meas_noise = params
    cov_func = partial(example_squared_exponential, sigma=sigma, ell=ell)
    S = cov_func(X, X) + np.square(meas_noise + 1e-8) * np.eye(y.size)
    return -example_compute_mvn_log_likelihood(S, y)


def TEST_COMPUTE_SQR_EXP_NLL(f):
    X = np.arange(5)
    y = np.array(
        [
            -0.02415509423853975,
            -2.0601745974286185,
            -0.6431114465594998,
            -0.21516292936011427,
            -1.6847658704470119,
        ]
    )
    params = [-1.1, 2.1, 0.3]

    assert_functions_close(
        example_compute_sqr_exp_negative_log_likelihood, f, params, X, y
    )


def generate_timings(f):
    N = 6
    MIN_MEASUREMENTS = 1000
    MAX_MEASUREMENTS = 5000

    cov_func = example_squared_exponential
    for n in np.linspace(MIN_MEASUREMENTS, MAX_MEASUREMENTS, N):
        n = round(n)
        X, y = generate_training_data(n)
        start_time = timeit.default_timer()
        count = 0
        while timeit.default_timer() - start_time < 1.0 or count < 3:
            count = count + 1
            f(cov_func, X, y, np.array([4]), meas_noise=0.1)
        timing = (timeit.default_timer() - start_time) / count
        print(f"Timing with {n} measurements: {timing}")
        yield n, timing


def example_constant_covariance(x_i, x_j, sigma_constant=10.0):
    m = np.atleast_1d(x_i).shape[0]
    n = np.atleast_1d(x_j).shape[0]
    return sigma_constant * sigma_constant * np.ones((m, n))


def example_direct_fit_predict_constant(X, y, sigma_constant, meas_noise):
    var_constant = sigma_constant * sigma_constant
    inv_var_constant = 1.0 / var_constant
    var_noise = meas_noise * meas_noise
    inv_var_noise = 1.0 / var_noise
    n = y.size

    gamma = inv_var_noise / (inv_var_constant + n * inv_var_noise)
    ratio = var_constant / var_noise
    return ratio * np.sum(y - gamma * np.sum(y))


def TEST_DIRECT_FIT_PREDICT(f):
    X = np.arange(5)
    y = np.array(
        [
            -0.02415509423853975,
            -2.0601745974286185,
            -0.6431114465594998,
            -0.21516292936011427,
            -1.6847658704470119,
        ]
    )
    x_star = [0.0, 1.0]

    assert_functions_close(
        example_direct_fit_predict_constant, f, X, y, np.pi, np.log(2)
    )


# A helper to solve D^-1 b when D is a vector representing diagonal elements
def diagonal_solve(D, b):
    b = np.array(b)
    if b.ndim == 1:
        return b / D
    else:
        return b / D[:, None]


# A helper to evaluate a covariance function to obtain only the
# diagonal elements of a covariance matrix.
#
# diagonal_covariance(cov_func, X) == np.diag(cov_func(X, X))
def diagonal_variance(cov_func, X):
    return np.array([cov_func(X[i], X[i])[0, 0] for i in range(X.shape[0])])


def example_sparse_fit_and_predict(cov_func, X, y, u, x_star, meas_noise):
    K_uu = cov_func(u, u)
    K_uf = cov_func(u, X)
    K_ff_diag = diagonal_variance(cov_func, X)
    K_su = cov_func(x_star, u)
    K_ss = cov_func(x_star, x_star)

    Q_ff_diag = np.diag(K_uf.T @ np.linalg.solve(K_uu, K_uf))
    D = K_ff_diag - Q_ff_diag + meas_noise * meas_noise
    S = K_uu + K_uf @ diagonal_solve(D, K_uf.T)

    print(diagonal_solve(D, y).shape)
    mean = K_su @ np.linalg.solve(S, K_uf) @ diagonal_solve(D, y)
    cov = (
        K_ss - K_su @ np.linalg.solve(K_uu, K_su.T) + K_su @ np.linalg.solve(S, K_su.T)
    )
    return mean, cov


def TEST_SPARSE_FIT_AND_PREDICT(f):
    X = np.arange(5)
    y = np.array(
        [
            -0.02415509423853975,
            -2.0601745974286185,
            -0.6431114465594998,
            -0.21516292936011427,
            -1.6847658704470119,
        ]
    )
    U = np.array([0.0])
    x_star = np.array([0.0, 1.0])

    assert_functions_close(
        example_sparse_fit_and_predict,
        f,
        example_constant_covariance,
        X,
        y,
        U,
        x_star,
        MEAS_NOISE,
    )


def example_sparse_fit(cov_func, X, y, U, meas_noise):
    meas_var = meas_noise * meas_noise

    nugget = 1e-10
    K_yu = cov_func(X, U)
    K_uu = cov_func(U, U) + nugget * np.eye(U.shape[0])
    S = K_uu + (1.0 / meas_var) * np.dot(K_yu.T, K_yu)
    S = 0.5 * (S + S.T) + nugget * np.eye(S.shape[0])

    L_uu = np.linalg.cholesky(K_uu)
    L_s = np.linalg.cholesky(S)
    v = scipy.linalg.cho_solve((L_s, True), np.dot(K_yu.T, y) / meas_var)

    return {
        "train_locations": U,
        "information": v,
        "cholesky_S": L_s,
        "cholesky_uu": L_uu,
        "cov_func": cov_func,
    }


def example_sparse_predict(fit_model, x_star):
    cov_func = fit_model["cov_func"]
    v = fit_model["information"]
    L_s = fit_model["cholesky_S"]
    L_uu = fit_model["cholesky_uu"]
    U = fit_model["train_locations"]

    K_su = cov_func(x_star, U)
    K_ss = cov_func(x_star, x_star)
    mean = np.dot(K_su, v)

    V_ys = scipy.linalg.solve_triangular(L_uu, K_su.T, lower=True)
    Si_K_us = scipy.linalg.cho_solve((L_s, True), K_su.T)
    K_su_Si_K_us = np.dot(K_su, Si_K_us)
    cov = K_ss - np.dot(V_ys.T, V_ys) + K_su_Si_K_us
    return mean, cov


def example_sparse_fit_and_predict(cov_func, X, y, U, x_test, meas_noise):
    fit_model = example_sparse_fit(cov_func, X, y, U, meas_noise)
    return example_sparse_predict(fit_model, x_test)
