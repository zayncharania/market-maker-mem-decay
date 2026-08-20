import numpy as np


T = 2500
TAU_MIN = 750
TAU_MAX = 1250
REGIMES = {
    1: (0.10, 0.20),
    2: (0.10, 0.40),
    3: (0.10, 0.70),
    4: (0.70, 0.60),
    5: (0.70, 0.40),
    6: (0.70, 0.10),
}


def p_buy(v, alpha):
    return (1 + v * alpha) / 2


def order_sim(v, alpha, n, rng):
    p = p_buy(v, alpha)
    draws = rng.random(n)
    buys = draws < p
    orders = np.where(buys, 1, -1)
    return orders


def episode_from_draws(regime, v, tau, uniforms):
    if regime not in REGIMES:
        raise ValueError("regime must be in {1, ..., 6}")
    if v not in (-1, 1):
        raise ValueError("v must be -1 or +1")
    if not TAU_MIN <= tau <= TAU_MAX:
        raise ValueError("tau must be in [750, 1250]")

    uniforms = np.asarray(uniforms)
    if uniforms.shape != (T,):
        raise ValueError("uniforms must have length 2500")

    alpha_pre, alpha_post = REGIMES[regime]
    alpha = np.empty(T)
    alpha[:tau] = alpha_pre
    alpha[tau:] = alpha_post

    probabilities = p_buy(v, alpha)
    orders = np.where(uniforms < probabilities, 1, -1)
    return v, tau, alpha, uniforms, orders


def episode_sim(regime, rng):
    v = rng.choice((-1, 1))
    tau = rng.integers(TAU_MIN, TAU_MAX + 1)
    uniforms = rng.random(T)
    return episode_from_draws(regime, v, tau, uniforms)
