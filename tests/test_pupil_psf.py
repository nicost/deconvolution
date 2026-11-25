"""Tests for the differentiable pupil -> PSF model (pupil_psf.PupilPSF)."""
import numpy as np
import pytest
import torch

import pupil_psf as pp
from make_airy_psf import airy_psf_3d

# Common imaging settings (100x / 1.40 oil, Cy3), small enough to be fast.
XY, Z, WL, NA, NI = 0.0651, 0.26, 0.605, 1.40, 1.515
NXY, NZ = 61, 31
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def model():
    return pp.PupilPSF(NXY, NZ, XY, Z, WL, NA, NI, device=CPU)


def test_pupil_radius_oversampled(model):
    # The pupil must be sampled over many pixels or Zernike phases are useless.
    assert model.R_pix >= 40
    assert model.N >= NXY


def test_ideal_psf_shape_sum_nonneg(model):
    psf = model.render_numpy()                 # all coefficients zero
    assert psf.shape == (NZ, NXY, NXY)
    assert abs(psf.sum() - 1.0) < 1e-4
    assert psf.min() >= 0.0


def test_ideal_psf_peak_centered(model):
    psf = model.render_numpy()
    peak = np.unravel_index(np.argmax(psf[NZ // 2]), (NXY, NXY))
    assert peak == (NXY // 2, NXY // 2)


def test_ideal_psf_matches_airy(model):
    """The zero-aberration pupil PSF must reproduce make_airy_psf's ideal PSF."""
    pupil = model.render_numpy()
    airy = airy_psf_3d(WL, NA, NI, XY, Z, NXY, NZ)
    airy = airy / airy.sum()
    cos = (pupil * airy).sum() / (np.linalg.norm(pupil) * np.linalg.norm(airy))
    nrmse = np.linalg.norm(pupil - airy) / np.linalg.norm(airy)
    assert cos > 0.95, f"cosine similarity too low: {cos:.3f}"
    assert nrmse < 0.15, f"NRMSE too high: {nrmse:.3f}"


def test_render_is_differentiable(model):
    coeffs = torch.zeros(model.n_terms, device=CPU, requires_grad=True)
    psf = model.render(coeffs)
    (psf ** 2).sum().backward()
    assert coeffs.grad is not None
    assert torch.isfinite(coeffs.grad).all()
    # Spherical aberration should have a non-zero effect on the PSF.
    sph = model.names.index("spherical")
    assert coeffs.grad[sph].abs() > 0


def test_spherical_breaks_axial_symmetry(model):
    """Spherical aberration makes the through-focus PSF asymmetric in z; the
    ideal PSF is symmetric."""
    def axial_asymmetry(psf):
        prof = psf[:, NXY // 2, NXY // 2]
        return np.linalg.norm(prof - prof[::-1]) / np.linalg.norm(prof)

    ideal = axial_asymmetry(model.render_numpy())
    coeffs = np.zeros(model.n_terms)
    coeffs[model.names.index("spherical")] = 0.5
    aberrated = axial_asymmetry(model.render_numpy(coeffs))
    assert ideal < 0.02
    assert aberrated > ideal + 0.02
