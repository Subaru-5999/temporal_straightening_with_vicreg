"""Tests for the combined "World Model + VICReg + SIGReg" objective.

VICReg (Bardes, Ponce & LeCun, ICLR 2022, arXiv:2105.04906) supplies explicit
second-order regularisation -- per-dim variance hinge + pairwise covariance --
while SIGReg (LeJEPA / LeWorldModel, research_papers/le-wm) pins the full
latent distribution via the sketched characteristic function. These tests pin
the semantics of running both on the same world model:

  * both terms are off by default and opt-in via config,
  * when enabled, every component is reported and the total loss is exactly
    base + lambda_var * std + lambda_cov * cov + lambda_SIG * SIGReg,
  * a collapsed (constant) encoder is penalised by BOTH terms,
  * vcreg_apply_to="visual" regularises exactly the tokens SIGReg sees and
    differs from the historical "enc" (visual+proprio) path,
  * gradients from the combined objective reach the encoder,
  * a VICReg run gets its own checkpoint directory (run_naming).

Run:  pytest tests/test_vicreg_sigreg.py -q
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.visual_world_model import VWorldModel
from run_naming import variant_tag


class _Enc(nn.Module):
    """Stand-in DINOv2 patch encoder: (n, 3, H, W) -> (n, num_patches, emb_dim)."""

    def __init__(self, emb_dim=4, patch_size=14, num_patches=9):
        super().__init__()
        self.name = "dinov2_vits14"
        self.emb_dim = emb_dim
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.latent_ndim = 2
        self.lin = nn.Linear(3 * 4 * 4, emb_dim)

    def forward(self, x):
        n = x.shape[0]
        pooled = nn.functional.adaptive_avg_pool2d(x, 4).reshape(n, -1)
        return self.lin(pooled).unsqueeze(1).expand(n, self.num_patches, self.emb_dim)


class _ConstEnc(_Enc):
    """Collapsed encoder: identical output for every input."""

    def forward(self, x):
        n = x.shape[0]
        return self.lin.weight.sum() * torch.ones(
            n, self.num_patches, self.emb_dim, device=x.device
        ) + self.lin.bias.reshape(1, 1, -1) * 0.0


class _Prop(nn.Module):
    def __init__(self, in_chans=4, emb_dim=10):
        super().__init__()
        self.in_chans = in_chans
        self.emb_dim = emb_dim
        self.lin = nn.Linear(in_chans, emb_dim)

    def forward(self, x):
        return self.lin(x)


class _Act(_Prop):
    pass


def _model(encoder=None, image_size=48, **kwargs):
    """Small end-to-end model. num_patches = (48 // 16)**2 = 9."""
    num_patches = (image_size // 16) ** 2
    return VWorldModel(
        image_size=image_size,
        num_hist=2,
        num_pred=1,
        encoder=encoder or _Enc(emb_dim=4, num_patches=num_patches),
        proprio_encoder=_Prop(),
        action_encoder=_Act(in_chans=2, emb_dim=10),
        decoder=None,
        predictor=nn.Identity(),
        proprio_dim=10,
        action_dim=10,
        concat_dim=1,
        num_action_repeat=1,
        num_proprio_repeat=1,
        **kwargs,
    )


def _batch(b=6, t=3):
    torch.manual_seed(0)
    obs = {
        "visual": torch.rand(b, t, 3, 48, 48),
        "proprio": torch.randn(b, t, 4),
    }
    act = torch.randn(b, t, 2)
    return obs, act


COMBINED = dict(
    vcreg=True,
    vcreg_std_coeff=25.0,
    vcreg_cov_coeff=1.0,
    vcreg_apply_to="visual",
    sigreg=True,
    sigreg_coeff=0.1,
    sigreg_num_proj=64,
    sigreg_apply_to="patch",
)


# ------------------------------------------------------------------ defaults
def test_off_by_default():
    m = _model()
    assert not m.vcreg
    assert not m.sigreg_enabled
    obs, act = _batch()
    _, _, _, loss, comp = m.forward(obs, act)
    assert torch.isfinite(loss)
    for key in ("z_vicreg_std_loss", "z_vicreg_cov_loss", "z_vcreg_loss_scaled",
                "sigreg_loss", "sigreg_loss_scaled"):
        assert key not in comp


def test_invalid_vcreg_apply_to_rejected():
    with pytest.raises(ValueError, match="vcreg_apply_to"):
        _model(vcreg=True, vcreg_apply_to="nope")


# ---------------------------------------------------------- combined objective
def test_combined_terms_reported_and_loss_is_exactly_additive():
    """L = L_pred(+decoder) + lam_var*std + lam_cov*cov + lam_SIG*SIGReg.

    The same seeded encoder in a plain model gives the base loss; the combined
    model's total must equal base + the two scaled regularisers exactly.
    """
    obs, act = _batch()

    torch.manual_seed(1)
    base = _model()
    _, _, _, loss_base, _ = base.forward(obs, act)

    torch.manual_seed(1)
    both = _model(**COMBINED)
    _, _, _, loss_both, comp = both.forward(obs, act)

    for key in ("z_vicreg_std_loss", "z_vicreg_cov_loss", "z_vcreg_loss_scaled",
                "sigreg_loss", "sigreg_loss_scaled"):
        assert key in comp, f"missing component {key}"
    torch.testing.assert_close(
        comp["z_vcreg_loss_scaled"],
        comp["z_vicreg_std_loss"] * 25.0 + comp["z_vicreg_cov_loss"] * 1.0,
    )
    torch.testing.assert_close(comp["sigreg_loss_scaled"], comp["sigreg_loss"] * 0.1)
    torch.testing.assert_close(
        loss_both,
        loss_base + comp["z_vcreg_loss_scaled"] + comp["sigreg_loss_scaled"],
    )


def test_collapse_is_penalised_by_both_terms():
    """A constant encoder is the degenerate solution the combination exists to
    forbid: the variance hinge must be ~1 (std ~ 0) and SIGReg strictly > 0,
    while the covariance term is exactly 0 (no spread to correlate)."""
    m = _model(encoder=_ConstEnc(emb_dim=4), **COMBINED)
    obs, act = _batch()
    _, _, _, loss, comp = m.forward(obs, act)
    # sqrt(0 + 1e-4) = 0.01 -> relu(1 - 0.01) = 0.99 on every dim
    assert comp["z_vicreg_std_loss"].item() == pytest.approx(0.99, abs=1e-5)
    assert comp["z_vicreg_cov_loss"].item() == pytest.approx(0.0, abs=1e-12)
    assert comp["sigreg_loss"].item() > 0.0
    assert torch.isfinite(loss)


def test_non_collapsed_latents_pay_less_variance_than_collapsed():
    torch.manual_seed(2)
    good = _model(**COMBINED)
    collapsed = _model(encoder=_ConstEnc(emb_dim=4), **COMBINED)
    obs, act = _batch()
    _, _, _, _, comp_good = good.forward(obs, act)
    _, _, _, _, comp_bad = collapsed.forward(obs, act)
    assert comp_good["z_vicreg_std_loss"].item() < comp_bad["z_vicreg_std_loss"].item()


# ------------------------------------------------- VICReg paper fidelity
# Cross-checks against Bardes et al. (ICLR 2022, arXiv:2105.04906), computed
# through independent code paths (torch.std / torch.cov / whitening) rather
# than the model's own algebra.
def _whiten(x):
    """Affinely transform (n, d) samples so their sample covariance is I."""
    x = x - x.mean(0)
    cov = torch.cov(x.T)
    L = torch.linalg.cholesky(cov)
    return x @ torch.linalg.inv(L).T


def test_std_loss_matches_the_paper_formula():
    """V(z) = (1/D) sum_i relu(1 - sqrt(Var(z_i) + eps)), eps = 1e-4."""
    m = _model()
    torch.manual_seed(5)
    z = torch.randn(256, 8) * torch.tensor([0.2, 0.5, 1.0, 2.0, 0.3, 1.5, 0.8, 4.0])
    std = z.std(dim=0)                       # independent path: torch.std
    expected = torch.relu(1 - torch.sqrt(std.pow(2) + 1e-4)).mean()
    torch.testing.assert_close(m.vcreg_std_loss(z), expected)


def test_cov_loss_matches_the_paper_formula():
    """C(z) = (1/D) sum_{i!=j} Cov(z)_ij^2, covariance with the n-1 divisor."""
    m = _model()
    torch.manual_seed(6)
    a = torch.randn(256)
    z = torch.stack([a, a + 0.5 * torch.randn(256), torch.randn(256),
                     a - torch.randn(256)], dim=1)   # correlated dims
    cov = torch.cov(z.T)                   # independent path: torch.cov
    off = cov - torch.diag(torch.diagonal(cov))
    expected = (off ** 2).sum() / z.shape[1]
    torch.testing.assert_close(m.vcreg_cov_loss(z), expected, rtol=1e-5, atol=1e-6)
    assert expected.item() > 0, "the probe tensor must actually be correlated"


def test_both_losses_vanish_on_whitened_unit_variance_latents():
    """Whitened latents have cov = I and std = 1: both VICReg terms must be ~0,
    i.e. the regulariser's unique minimum is exactly the paper's target."""
    m = _model()
    torch.manual_seed(7)
    z = _whiten(torch.randn(512, 8))
    assert m.vcreg_cov_loss(z).item() == pytest.approx(0.0, abs=1e-6)
    # std = 1 -> relu(1 - sqrt(1 + 1e-4)) = 0
    assert m.vcreg_std_loss(z).item() == pytest.approx(0.0, abs=1e-6)


def test_std_loss_ignores_dims_above_the_margin():
    """The hinge is one-sided: dims with std >= 1 contribute nothing, so the
    term cannot shrink an already-informative latent (paper property)."""
    m = _model()
    torch.manual_seed(8)
    z = _whiten(torch.randn(512, 4)) * 3.0   # std = 3 on every dim
    assert m.vcreg_std_loss(z).item() == 0.0


def test_cov_loss_detects_correlation_and_whitening_removes_it():
    m = _model()
    torch.manual_seed(9)
    a = torch.randn(512)
    # A hair of noise on the doubled column: exact collinearity makes the
    # sample covariance singular, and whether Cholesky tolerates that then
    # depends on last-bit rounding that differs across LAPACK builds (seen
    # failing on the B200 pod). Correlation stays ~0.9999, but the matrix is
    # strictly positive-definite on every platform.
    z = torch.stack([a, 2 * a + torch.randn(512) * 0.01,
                     a + torch.randn(512) * 0.1, torch.randn(512)], dim=1)
    assert m.vcreg_cov_loss(z).item() > 1.0, "strongly correlated dims must pay"
    assert m.vcreg_cov_loss(_whiten(z)).item() == pytest.approx(0.0, abs=1e-6)


def test_epsilon_guard_constant_input_is_finite_not_nan():
    """With eps = 0 the sqrt would still be fine at var = 0, but the gradient
    would blow up; the value must be relu(1 - sqrt(1e-4)) = 0.99."""
    m = _model()
    z = torch.ones(64, 8)
    loss = m.vcreg_std_loss(z)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.99, abs=1e-5)


# ------------------------------------------------------------ apply_to options
def test_visual_and_enc_regularise_different_tensors():
    """With proprio channels present, 'enc' (visual+proprio) and 'visual' must
    give different statistics -- they are distinct objective variants."""
    torch.manual_seed(3)
    obs, act = _batch()
    m_vis = _model(vcreg=True, vcreg_std_coeff=1.0, vcreg_cov_coeff=1.0,
                   vcreg_apply_to="visual")
    m_enc = _model(vcreg=True, vcreg_std_coeff=1.0, vcreg_cov_coeff=1.0,
                   vcreg_apply_to="enc")
    _, _, _, _, comp_vis = m_vis.forward(obs, act)
    _, _, _, _, comp_enc = m_enc.forward(obs, act)
    assert not torch.allclose(
        comp_vis["z_vicreg_cov_loss"], comp_enc["z_vicreg_cov_loss"]
    )


def test_gradients_from_the_combined_objective_reach_the_encoder():
    torch.manual_seed(4)
    m = _model(**COMBINED)
    obs, act = _batch()
    _, _, _, loss, _ = m.forward(obs, act)
    loss.backward()
    assert m.encoder.lin.weight.grad is not None
    assert m.encoder.lin.weight.grad.abs().sum() > 0, (
        "the combined regularisers must push gradient into the encoder, or "
        "they cannot prevent collapse"
    )


# ------------------------------------------------------------------ run naming
def test_vcreg_run_gets_its_own_directory():
    base = variant_tag(False, 0.0, True, "features")
    vic = variant_tag(False, 0.0, True, "features", 0.0, 0.0, 0.0,
                      True, 25.0, 1.0)
    assert base == ""
    assert vic == "_vic_s2e1_c1e0"
    combined = variant_tag(True, 0.1, False, "features", 0.0, 0.0, 0.0,
                           True, 1.0, 0.04)
    assert combined == "_sig1e-1_e2e_vic_s1e0_c4e-2"
    assert combined != variant_tag(True, 0.1, False, "features"), (
        "a WM+VICReg+SIGReg run must not resume the SIGReg-only checkpoint"
    )


def test_vcreg_flag_with_zero_coefficients_does_not_rename():
    """Zero weights contribute nothing, so the directory must stay unchanged."""
    assert variant_tag(False, 0.0, True, "features", 0.0, 0.0, 0.0,
                       True, 0.0, 0.0) == ""


def test_vic_tag_is_filesystem_safe():
    tag = variant_tag(True, 0.1, False, "features", 0.0, 0.0, 0.0,
                      True, 25.0, 1.0)
    assert all(c.isalnum() or c in "_-" for c in tag)
