import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.special import dawsn, erf
from scipy.stats import gmean


def quadrature(f):
    """Integral of f from t to t_prime."""
    return lambda t, t_prime: np.array(
        [quad(f, t, t_prime)[0] for t, t_prime in zip(t, t_prime)]
    )


def inverse_sample(int_int_factor):
    """Sample from the density proportional to int_factor in [t, t + h]."""

    def inverse(t, h, y):
        """Return x such that int_int_factor(t, x) = y."""
        sol = root_scalar(
            lambda x: int_int_factor(t, x) - y,
            bracket=[t, t + h],
            method="brentq",
        )
        return sol.root

    return lambda t, h, u: np.array(
        [inverse(t, h, u * int_int_factor(t, t + h)) for t, u in zip(t, u)]
    )


def order_sample(int_int_factor, int_factor, scale_t):
    """Sample from the density proportional to int_factor in [t, t + h]."""

    def inverse(t, h, y):
        sol = root_scalar(
            lambda x: int_int_factor(t, x) - y,
            fprime=int_factor,
            fprime2=lambda t: -int_factor(t) * scale_t(t),
            bracket=[t, t + h],
            x0=t + h / 2,
            method="halley",
        )
        return sol.root

    return lambda t, h, u: np.array(
        [inverse(t, h, u * int_int_factor(t, t + h)) for t, u in zip(t, u)]
    )


def split_quad(f, a, b):
    return quad(f, a, b, points=[S_min, S_max])[0]


def check(
    rng,
    scale_t,
    int_factor,
    int_int_factor=None,
    sample_t=None,
    noise_t=None,
    sigma=None,
    s=None,
    a=0.0,
    b=5.0,
    num=int(1e3),
    h=0.1,
):
    t = np.linspace(a, b, num=num)
    # remove 0
    t += t[1]

    if scale_t is not None and int_factor is not None:
        g = np.exp(-np.array([split_quad(scale_t, t[0], t0) for t0 in t]))
        g *= gmean(int_factor(t) / g)
        assert np.allclose(int_factor(t), g), "int_factor wrong."

    if int_int_factor is not None:
        G = np.array([split_quad(int_factor, t[0], t0) for t0 in t])
        assert np.allclose(
            int_int_factor(t) - int_int_factor(t[0]), G
        ), "int_int_factor wrong."

    if noise_t is not None:
        assert (
            int_factor is not None and sigma is not None and s is not None
        ), "need int_factor and sigma and s."
        f = lambda t: np.square(int_factor(t) * sigma(t) * s(t))
        G = np.array([split_quad(f, t[0], t0) for t0 in t])
        assert np.allclose(noise_t(t) - noise_t(t[0]), G), "noise_t wrong."

    if (
        sample_t is None
        or scale_t is None
        or int_factor is None
        or int_int_factor is None
    ):
        return

    int_diff = lambda t, t_prime: int_int_factor(t_prime) - int_int_factor(t)
    # u = np.linspace(0, 1, num=num)
    u = rng.random((num,))
    sampler = inverse_sample(int_diff)
    assert np.allclose(sample_t(t, h, u), sampler(t, h, u)), "sample_t wrong."
    sampler = order_sample(int_diff, int_factor, scale_t)
    assert np.allclose(sample_t(t, h, u), sampler(t, h, u)), "sample_t wrong."


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    sigma_data = 0.5
    vp_beta_d = 19.9
    vp_beta_min = 0.1
    sigma = lambda t: np.sqrt(
        np.exp(0.5 * vp_beta_d * t**2 + vp_beta_min * t) - 1
    )
    sigma_max = 80
    sigma_min = 0.002
    sigma_stationary = 2
    S_churn = 40
    S_min = 0.05
    S_max = 50 / 20
    S_noise = 1.003

    beta_rate = S_churn / (sigma_max - sigma_min) * S_noise**2
    beta = lambda t: ((S_min <= t) & (t <= S_max)) * beta_rate

    scale_t = lambda t: 1 / t
    int_factor = lambda t: 1 / t
    int_int_factor = lambda t: np.log(t)
    sample_t = lambda t, h, u: t * np.pow(1 + h / t, u)

    check(rng, scale_t, int_factor, int_int_factor, sample_t)

    scale_t = lambda t: t / (t**2 + sigma_data**2)
    int_factor = lambda t: 1 / np.sqrt(t**2 + sigma_data**2)
    int_int_factor = lambda t: np.atanh(t / np.sqrt(t**2 + sigma_data**2))
    sample_t = (
        lambda t, h, u: sigma_data
        * np.tanh(x := (1 - u) * int_int_factor(t) + u * int_int_factor(t + h))
        * np.cosh(x)
    )

    check(rng, scale_t, int_factor, int_int_factor, sample_t)

    scale_t = lambda t: 1 / (2 * t)
    int_factor = lambda t: 1 / np.sqrt(t)
    int_int_factor = lambda t: 2 * np.sqrt(t)
    sample_t = lambda t, h, u: np.square(
        (1 - u) * np.sqrt(t) + u * np.sqrt(t + h)
    )

    check(rng, scale_t, int_factor, int_int_factor, sample_t)

    scale_t = lambda t: 1 / (2 * (t + sigma_data**2))
    int_factor = lambda t: 1 / np.sqrt(t + sigma_data**2)
    int_int_factor = lambda t: 2 * np.sqrt(t + sigma_data**2)
    sample_t = (
        lambda t, h, u: np.square(
            (1 - u) * np.sqrt(t + sigma_data**2)
            + u * np.sqrt(t + h + sigma_data**2)
        )
        - sigma_data**2
    )

    check(rng, scale_t, int_factor, int_int_factor, sample_t)

    scale_t = lambda t: 1
    int_factor = lambda t: np.exp(-t)
    int_int_factor = lambda t: -np.exp(-t)
    sample_t = lambda t, h, u: t - np.logaddexp(np.log1p(-u), np.log(u) - h)

    check(rng, scale_t, int_factor, int_int_factor, sample_t)

    scale_t = lambda t: 0
    int_factor = lambda t: 1
    int_int_factor = lambda t: t
    sample_t = lambda t, h, u: t + h * u

    check(rng, scale_t, int_factor, int_int_factor, sample_t)

    sigma_t = lambda t: t
    s = lambda t: 1
    noise_t = lambda t: t**3 / 3

    check(rng, scale_t, int_factor, noise_t=noise_t, sigma=sigma_t, s=s)

    sigma_t = lambda t: np.sqrt(t)
    s = lambda t: 1
    noise_t = lambda t: t**2 / 2

    check(rng, scale_t, int_factor, noise_t=noise_t, sigma=sigma_t, s=s)

    scale_t = lambda t: -(vp_beta_d * t + vp_beta_min) / 2
    int_factor = lambda t: np.sqrt(1 + sigma(t) ** 2)
    int_int_factor = lambda t: (
        2
        * np.sqrt(1 + sigma(t) ** 2)
        * dawsn((vp_beta_d * t + vp_beta_min) / (2 * np.sqrt(vp_beta_d)))
        / np.sqrt(vp_beta_d)
    )

    check(rng, scale_t, int_factor, int_int_factor, None)

    # not integrable
    # scale_t = lambda t: (vp_beta_d * t + vp_beta_min) / (2 * sigma(t) ** 2)
    #
    # check(rng, scale_t, int_factor, None, None)

    scale_t = (
        lambda t: (1 - sigma_data**2)
        * (vp_beta_d * t + vp_beta_min)
        / (2 * (sigma(t) ** 2 + sigma_data**2))
    )
    int_factor = lambda t: np.sqrt(
        (sigma(t) ** 2 + 1) / (sigma(t) ** 2 + sigma_data**2)
    )
    check(rng, scale_t, int_factor)

    scale_t = lambda t: (t + beta(t) * t**2) / sigma_stationary**2
    int_factor = lambda t: np.exp(
        -(t**2 / 2 + beta_rate * np.clip(t, S_min, S_max) ** 3 / 3)
        / sigma_stationary**2
    )

    check(rng, scale_t, int_factor)

    scale_t = lambda t: (0.5 + beta(t) * t) / sigma_stationary**2
    int_factor = lambda t: np.exp(
        -(t / 2 + beta_rate * np.clip(t, S_min, S_max) ** 2 / 2)
        / sigma_stationary**2
    )

    trunc_case = (
        lambda endpoint, t: -2
        * sigma_stationary**2
        * np.exp(-(endpoint**2 * beta_rate + t) / (2 * sigma_stationary**2))
    )
    int_factor_trunc = lambda t: np.exp(
        -(t / 2 + beta_rate * S_min**2 / 2) / sigma_stationary**2
    )
    check(rng, None, int_factor_trunc, lambda t: trunc_case(S_min, t))

    in_case = (
        lambda t: sigma_stationary
        * np.exp(1 / (8 * beta_rate * sigma_stationary**2))
        * np.sqrt(np.pi / (2 * beta_rate))
        * erf(
            (2 * beta_rate * t + 1)
            / (2 * np.sqrt(2 * beta_rate) * sigma_stationary)
        )
    )
    int_factor_untrunc = lambda t: np.exp(
        -(t / 2 + beta_rate * t**2 / 2) / sigma_stationary**2
    )
    check(rng, None, int_factor_untrunc, in_case)

    int_int_factor = lambda t: np.where(
        t <= S_min,
        trunc_case(S_min, t) - trunc_case(S_min, 0),
        np.where(
            t <= S_max,
            trunc_case(S_min, S_min)
            - trunc_case(S_min, 0)
            + in_case(t)
            - in_case(S_min),
            trunc_case(S_min, S_min)
            - trunc_case(S_min, 0)
            + in_case(S_max)
            - in_case(S_min)
            + trunc_case(S_max, t)
            - trunc_case(S_max, S_max),
        ),
    )

    check(rng, scale_t, int_factor, int_int_factor)

    scale_t = (
        lambda t: (
            (1 - sigma_stationary**2) * (vp_beta_d * t + vp_beta_min) / 2
            + sigma(t) ** 2 / (sigma(t) ** 2 + 1) * beta(t)
        )
        / sigma_stationary**2
    )
    scale_t_left = lambda t: (
        (1 - sigma_stationary**2)
        * (vp_beta_d * t + vp_beta_min)
        / (2 * sigma_stationary**2)
    )
    scale_t_right = (
        lambda t: sigma(t) ** 2
        / (sigma(t) ** 2 + 1)
        * beta(t)
        / sigma_stationary**2
    )

    int_left = (
        lambda t: (1 - sigma_stationary**2)
        * (vp_beta_d * t**2 / 2 + vp_beta_min * t)
        / (2 * sigma_stationary**2)
    )
    check(rng, None, scale_t_left, int_left)

    int_right_in = (
        lambda t: beta_rate
        * (
            t
            - np.sqrt(np.pi / (2 * vp_beta_d))
            * np.exp(vp_beta_min**2 / (2 * vp_beta_d))
            * erf((vp_beta_d * t + vp_beta_min) / np.sqrt(2 * vp_beta_d))
        )
        / sigma_stationary**2
    )
    check(
        rng,
        None,
        scale_t_right,
        lambda t: int_right_in(np.clip(t, S_min, S_max)),
    )

    int_factor = lambda t: np.exp(
        -(int_left(t) + int_right_in(np.clip(t, S_min, S_max)))
    )
    check(rng, scale_t, int_factor)

    int_factor = lambda t: np.exp(-t)
    int_diff = lambda t, t_prime: quad(int_factor, t, t_prime)[0]

    t = np.array([0.01])
    h = 2
    u = np.array([0.5])

    x = int_diff(t, h)

    sample = inverse_sample(int_diff)
    y = sample(t, h, u)
    print(y)
    print(int_diff(t, y), u * int_diff(t, t + h))

    sample = order_sample(int_diff, int_factor, scale_t)
    y = sample(t, h, u)
    print(y)
    print(int_diff(t, y), u * int_diff(t, t + h))
