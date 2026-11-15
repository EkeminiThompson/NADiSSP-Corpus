import math

import numpy as np
import torch

from src.models.nadip import (GradientReversal, WeibullPHHead, ganin_lambda, masked_bce, masked_mse, nt_xent_loss,
                              load_compatible, TimeSeriesTransformer)


def test_grl_identity_forward_negated_scaled_backward():
    g = GradientReversal(0.7)
    x = torch.randn(4, 3, requires_grad=True)
    y = g(x)
    assert torch.equal(y, x)
    (y * torch.arange(3.0)).sum().backward()
    assert torch.allclose(x.grad, -0.7 * torch.arange(3.0).expand(4, 3))


def test_ganin_schedule_eq3():
    assert ganin_lambda(0.0) == 0.0
    assert abs(ganin_lambda(1.0) - (2 / (1 + math.exp(-10)) - 1)) < 1e-12
    assert 0.9999 < ganin_lambda(1.0) <= 1.0
    assert ganin_lambda(0.5, 0.4) == 0.4 * (2 / (1 + math.exp(-5)) - 1)


def test_weibull_beta_gt_1_and_hazard_increasing():
    h = WeibullPHHead(8)
    assert h.beta.item() > 1.0
    z = torch.randn(1, 8)
    t = torch.linspace(1, 200, 50)
    r = h.risk(z)
    hazard = (h.beta / h.eta) * (t / h.eta) ** (h.beta - 1) * torch.exp(r)
    assert (hazard[1:] > hazard[:-1]).all()


def test_weibull_nll_matches_closed_form_with_censoring():
    h = WeibullPHHead(4)
    z = torch.randn(6, 4)
    t = torch.tensor([10., 40., 80., 120., 5., 60.])
    ev = torch.tensor([1., 1., 0., 0., 1., 1.])
    beta, eta, r = h.beta.item(), h.eta.item(), h.risk(z).detach().numpy()
    expect = 0.0
    for ti, ei, ri in zip(t.numpy(), ev.numpy(), r):
        logS = -((ti / eta) ** beta) * math.exp(ri)
        logh = math.log(beta / eta) + (beta - 1) * math.log(ti / eta) + ri
        expect -= logS + ei * logh
    assert abs(h.nll(t, ev, z).item() - expect / 6) < 1e-4


def test_weibull_median_satisfies_S_half():
    h = WeibullPHHead(4)
    z = torch.randn(5, 4)
    t = h.rul_median(z)
    S = torch.exp(h.log_survival(t, z))
    assert torch.allclose(S, torch.full_like(S, 0.5), atol=1e-4)


def test_weibull_head_can_learn_shorter_rul_for_higher_risk():
    """Fit on synthetic (z, t) where t = 100 * exp(-z0): the head must rank-order RUL correctly."""
    torch.manual_seed(0)
    z = torch.randn(512, 4)
    t = (100 * torch.exp(-0.5 * z[:, 0]) * torch.empty(512).uniform_(0.8, 1.2)).clamp(min=2)
    h = WeibullPHHead(4, eta_init=100)
    opt = torch.optim.Adam(h.parameters(), lr=0.05)
    for _ in range(400):
        loss = h.nll(t, torch.ones(512), z)
        opt.zero_grad(); loss.backward(); opt.step()
    pred = h.rul_median(z)
    corr = np.corrcoef(pred.detach().numpy(), t.numpy())[0, 1]
    assert corr > 0.8


def test_cox_partial_likelihood_matches_manual():
    h = WeibullPHHead(3)
    z = torch.randn(5, 3)
    t = torch.tensor([5., 3., 9., 1., 7.])
    ev = torch.tensor([1., 1., 0., 1., 1.])
    r = h.risk(z).detach().numpy()
    manual = 0.0
    for i in range(5):
        if ev[i] == 1:
            risk_set = [j for j in range(5) if t[j] >= t[i]]
            manual -= r[i] - np.log(np.exp(r[risk_set]).sum())
    assert abs(h.cox_partial_nll(t, ev, z).item() - manual / ev.sum().item()) < 1e-5


def test_nt_xent_prefers_matching_views():
    torch.manual_seed(0)
    z = torch.randn(16, 8)
    assert nt_xent_loss(z, z + 0.01 * torch.randn_like(z)).item() < nt_xent_loss(z, torch.randn(16, 8)).item()


def test_masked_losses_ignore_masked_rows_and_survive_empty_mask():
    p, y = torch.tensor([1., 5.]), torch.tensor([1., 0.])
    assert masked_mse(p, y, torch.tensor([True, False])).item() == 0.0
    assert masked_mse(p, y, torch.tensor([False, False])).item() == 0.0
    lg = torch.tensor([10., -10.], requires_grad=True)
    assert masked_bce(lg, torch.tensor([1., 1.]), torch.tensor([True, False])).item() < 1e-3
    masked_bce(lg, torch.tensor([1., 1.]), torch.tensor([False, False])).backward()     # must not error


def test_load_compatible_reports_what_it_loaded():
    a = TimeSeriesTransformer(5, 16, 2, 2, 32)
    b = TimeSeriesTransformer(5, 16, 2, 2, 32)
    rep = load_compatible(a, b.state_dict())
    assert rep["loaded"] > 0 and rep["shape_mismatch"] == 0
    c = TimeSeriesTransformer(5, 24, 2, 2, 32)
    rep2 = load_compatible(a, c.state_dict())
    assert rep2["loaded"] < rep["loaded"] and rep2["shape_mismatch"] > 0


def test_default_architecture_matches_table_3_15b():
    from src.config import load_config
    from src.models.nadip import NADIP
    c = load_config()["model"]
    assert (c["d_model"], c["nhead"], c["num_layers"]) == (192, 8, 12)
    m = NADIP(41, c)
    assert len(m.backbone.layers) == 12 and m.backbone.d_model == 192
    assert sum(isinstance(l, torch.nn.Linear) for l in m.domain_disc) == 3          # 3-layer MLP discriminator
    assert sum(isinstance(l, torch.nn.Linear) for l in m.projector) == 3            # 3-layer MLP projector
