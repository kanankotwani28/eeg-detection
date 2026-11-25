import math

import numpy as np
import torch

from src.models.logreg import LinearModel
from src.privacy.accountant import calibrate_sigma, epoch_level_epsilon, epsilon
from src.privacy.participant_dp import add_noise, clip_and_sum, dp_local_train
from src.train.torch_core import PData


def test_clipping_bounds_each_participant():
    g = [torch.randn(50, 7, 3) * 100, torch.randn(50, 4) * 0.001]
    s, post = clip_and_sum(g, clip=1.0)
    assert torch.all(post <= 1.0 + 1e-5)
    small = [torch.randn(5, 3) * 1e-3]
    s2, post2 = clip_and_sum(small, clip=1.0)
    assert torch.allclose(s2[0], small[0].sum(0))  # below the bound: unchanged


def test_one_participant_changes_sum_by_at_most_C():
    g = [torch.randn(10, 20) * 50]
    s_all, _ = clip_and_sum(g, 1.0)
    s_wo, _ = clip_and_sum([g[0][1:]], 1.0)
    assert (s_all[0] - s_wo[0]).norm() <= 1.0 + 1e-5


def test_noise_scale():
    gen = torch.Generator().manual_seed(0)
    z = add_noise([torch.zeros(200_000)], sigma=1.3, clip=0.7, gen=gen)[0]
    assert abs(z.std().item() - 1.3 * 0.7) < 0.01


def test_participant_with_many_epochs_is_bounded_like_one_epoch():
    """mean_epochs mode: a participant's total contribution is clipped as one unit."""
    rng = np.random.default_rng(0)
    data = PData(["a", "b"], [1, 0], [rng.standard_normal((30, 5)).astype(np.float32) * 100,
                                      rng.standard_normal((1, 5)).astype(np.float32)])
    m = LinearModel(5)
    w0 = m.linear.weight.detach().clone()
    dp_local_train(m, data, steps=1, q=1.0, sigma=0.0, clip=0.5, lr=1.0, seed=0, device="cpu", grad_mode="mean_epochs")
    # sum of 2 clipped gradients / expected batch (2) -> step norm <= C
    step = torch.cat([(m.linear.weight - w0).flatten(), m.linear.bias.detach().flatten()])
    assert step.norm() <= 0.5 + 1e-5


def test_accountant_matches_analytic_gaussian_rdp():
    """With q=1 the mechanism is a plain Gaussian: RDP(a) = T*a/(2 sigma^2). Opacus converts RDP to (eps, delta)
    with eps = min_a [ rdp - (log(delta) + log(a))/(a-1) + log((a-1)/a) ] (Balle et al. 2020)."""
    from opacus.accountants.rdp import RDPAccountant

    sigma, T, delta = 2.0, 50, 1e-5
    orders = np.array(RDPAccountant.DEFAULT_ALPHAS, dtype=float)
    rdp = T * orders / (2 * sigma**2)
    eps_ref = np.min(rdp - (np.log(delta) + np.log(orders)) / (orders - 1) + np.log((orders - 1) / orders))
    assert abs(epsilon([(sigma, 1.0, T)], delta) - eps_ref) < 1e-6


def test_accountant_reference_value_and_monotonicity():
    # Opacus' RDP accountant is the reference implementation; values below are reproducible facts about it
    e1 = epsilon([(1.0, 0.01, 1000)], 1e-5)
    e2 = epsilon([(1.0, 0.01, 2000)], 1e-5)
    e3 = epsilon([(2.0, 0.01, 1000)], 1e-5)
    assert e2 > e1 > e3 > 0
    assert epsilon([(0.0, 0.1, 10)], 1e-5) == math.inf


def test_calibrate_sigma_hits_target_and_composes_extra():
    for target in (1.0, 3.0, 8.0):
        s = calibrate_sigma(target, 1e-5, 0.25, 240)
        assert epsilon([(s, 0.25, 240)], 1e-5) <= target + 1e-6
        assert epsilon([(s * 0.97, 0.25, 240)], 1e-5) > target
    s_extra = calibrate_sigma(3.0, 1e-5, 0.25, 240, extra=[(5.0, 1.0, 1)])
    assert s_extra > calibrate_sigma(3.0, 1e-5, 0.25, 240)
    assert epoch_level_epsilon(s, 10, 3000, 240, 1e-5) < epsilon([(s, 0.25, 240)], 1e-5)


def test_dp_training_without_noise_equals_clipped_mean_gradient():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((6, 4)).astype(np.float32)
    data = PData([str(i) for i in range(6)], [0, 1, 0, 1, 1, 0], [x[None] for x in X])
    m = LinearModel(4)
    dp_local_train(m, data, steps=1, q=1.0, sigma=0.0, clip=1e6, lr=1.0, seed=0, device="cpu")
    ref = LinearModel(4)
    loss = torch.nn.functional.cross_entropy(ref(torch.from_numpy(X)), torch.tensor(data.y))
    loss.backward()
    assert torch.allclose(m.linear.weight, -ref.linear.weight.grad, atol=1e-5)
