"""End-to-end test for semi-blind deconvolution.

The method fits the PSF by *model selection*: it deconvolves the image with
several candidate PSFs and keeps the one that gives the sharpest result (see
deconvolve_semiblind.py).  This test injects a known spherical aberration and
checks that the search recovers it.  It is marked 'slow' (it runs several small
deconvolutions); run the quick tests only with:  pytest -m "not slow".
"""
import numpy as np
import pytest
import torch

import pupil_psf as pp
import rl_core as rc
from deconvolve_semiblind import fit_coefficients, sharpness

CPU = torch.device("cpu")
XY, Z, WL, NA, NI = 0.0651, 0.26, 0.605, 1.40, 1.515


def _sparse_object(nz, nxy, n_points, photons, seed=0):
    rng = np.random.default_rng(seed)
    obj = np.zeros((nz, nxy, nxy), dtype=np.float32)
    obj[rng.integers(2, nz - 2, n_points),
        rng.integers(nxy // 4, 3 * nxy // 4, n_points),
        rng.integers(nxy // 4, 3 * nxy // 4, n_points)] = photons
    return obj


def _make_image(model, nz, nxy, true_coeffs, seed=0):
    obj = _sparse_object(nz, nxy, n_points=25, photons=400.0, seed=seed)
    obj_t = torch.tensor(obj, device=CPU)
    otf_t, _, _ = rc.otf_from_psf(model.render_numpy(true_coeffs), obj_t.shape, CPU)
    rate = rc.fftconv(obj_t, otf_t, obj_t.shape).clamp(min=0)
    return torch.poisson(rate, generator=torch.Generator(device=CPU).manual_seed(seed))


def test_sharpness_prefers_the_true_psf():
    """The sharpness score (used by the fit) is highest for the correct PSF."""
    nz, nxy, nxy_psf = 9, 40, 25
    model = pp.PupilPSF(nxy_psf, nz, XY, Z, WL, NA, NI, device=CPU)
    sph = model.names.index("spherical")

    true_coeffs = np.zeros(model.n_terms)
    true_coeffs[sph] = 0.30
    image = _make_image(model, nz, nxy, true_coeffs)

    def sharp_for(c):
        coeffs = np.zeros(model.n_terms)
        coeffs[sph] = c
        psf = model.render_numpy(coeffs)
        otf, otfT, otfotfT = rc.otf_from_psf(psf, image.shape, CPU)
        gen = torch.Generator(device=CPU).manual_seed(0)
        u = rc.gradient_consensus_deconvolve(image, otf, otfT, otfotfT, gen,
                                             n_steps=50, verbose=False)
        return sharpness(u.cpu().numpy())

    # The true value should beat clearly-wrong ones on both sides.
    assert sharp_for(0.30) > sharp_for(0.0)
    assert sharp_for(0.30) > sharp_for(0.60)


@pytest.mark.slow
def test_semiblind_recovers_spherical():
    """The full search recovers an injected spherical aberration."""
    nz, nxy, nxy_psf = 9, 40, 25
    true_sph = 0.30

    model = pp.PupilPSF(nxy_psf, nz, XY, Z, WL, NA, NI, device=CPU)
    sph = model.names.index("spherical")

    true_coeffs = np.zeros(model.n_terms)
    true_coeffs[sph] = true_sph
    image = _make_image(model, nz, nxy, true_coeffs)

    # refine=False: spherical scan only (fast and deterministic).
    coeffs = fit_coefficients(model, image, CPU, refine=False, verbose=False)
    assert abs(coeffs[sph] - true_sph) < 0.1, \
        f"spherical not recovered: {coeffs[sph]:.3f}"
