import numpy as np

from adaptive_mm.simulator import episode_from_draws, episode_sim, order_sim


def test_order_f_match_p():
    rng = np.random.default_rng(67)
    orders = order_sim(v=1, alpha=0.40, n=1_000_000, rng=rng)
    f_buy = np.mean(orders == 1)

    assert abs(f_buy - 0.70) < 0.002


def test_episode_regimes_and_tau_indexing():
    regimes = {
        1: (0.10, 0.20),
        2: (0.10, 0.40),
        3: (0.10, 0.70),
        4: (0.70, 0.60),
        5: (0.70, 0.40),
        6: (0.70, 0.10),
    }
    uniforms = np.full(2500, 0.50)

    for regime, (alpha_pre, alpha_post) in regimes.items():
        for tau in (750, 1250):
            _, _, alpha, _, _ = episode_from_draws(
                regime=regime, v=1, tau=tau, uniforms=uniforms
            )

            assert np.all(alpha[:tau] == alpha_pre)
            assert np.all(alpha[tau:] == alpha_post)


def test_episode_orders_use_supplied_uniforms_and_strict_threshold():
    tau = 750
    uniforms = np.zeros(2500)
    uniforms[0] = 0.55
    uniforms[tau] = 0.60

    v, returned_tau, alpha, returned_uniforms, orders = episode_from_draws(
        regime=1, v=1, tau=tau, uniforms=uniforms
    )
    expected_orders = np.where(returned_uniforms < (1 + v * alpha) / 2, 1, -1)

    assert v == 1
    assert returned_tau == tau
    assert np.array_equal(returned_uniforms, uniforms)
    assert np.array_equal(orders, expected_orders)
    assert orders[0] == -1
    assert orders[tau] == -1


def test_episode_sim_returns_complete_episode():
    rng = np.random.default_rng(123)
    v, tau, alpha, uniforms, orders = episode_sim(regime=3, rng=rng)

    assert v in (-1, 1)
    assert 750 <= tau <= 1250
    assert alpha.shape == (2500,)
    assert uniforms.shape == (2500,)
    assert orders.shape == (2500,)
    assert np.all((uniforms >= 0) & (uniforms < 1))
    assert set(np.unique(orders)) <= {-1, 1}
    expected_orders = np.where(uniforms < (1 + v * alpha) / 2, 1, -1)
    assert np.array_equal(orders, expected_orders)
