import numpy as np


def p_buy(v, alpha):
    return (1 + v * alpha) / 2


def order_sim(v, alpha, n, rng):
    p = p_buy(v, alpha)
    draws = rng.random(n)
    buys = draws < p
    orders = np.where(buys, 1, -1)
    return orders

