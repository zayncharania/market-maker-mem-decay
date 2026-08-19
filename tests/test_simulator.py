import numpy as np

from adaptive_mm.simulator import order_sim


def test_order_f_match_p():
    rng = np.random.default_rng(67)
    orders = order_sim(v=1, alpha=0.40, n=1_000_000, rng=rng)
    f_buy = np.mean(orders == 1)

    assert abs(f_buy - 0.70) < 0.002