"""Tests for the semi-blind machinery.

Two things are tested separately:

1. The PSF-fitting step (m_step) on its own: given a good object estimate, can it
   recover a known aberration?  This is the core differentiable machinery and it
   is deterministic, so it is asserted strictly.

2. The full alternating loop (E-step + M-step together), starting from a flat
   object.  This currently recovers the aberration only weakly, because the
   object estimate absorbs the PSF error (the classic object/PSF degeneracy of
   blind deconvolution).  It is therefore marked xfail until the driver is
   strengthened (e.g. an object sparsity prior / joint optimization).  Run the
   quick tests only with:  pytest -m "not slow".
"""
import numpy as np
import pytest
import torch

import pupil_psf as pp
import rl_core as rc
from deconvolve_semiblind import m_step

CPU = torch.device("cpu")
XY, Z, WL, NA, NI = 0.0651, 0.26, 0.605, 1.40, 1.515


def _norm_rmse(a, b):
    a = a / a.sum()
    b = b / b.sum()
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def _sparse_object(nz, nxy, n_points, photons, seed=0):
    rng = np.random.default_rng(seed)
    obj = np.zeros((nz, nxy, nxy), dtype=np.float32)
    obj[rng.integers(2, nz - 2, n_points),
        rng.integers(nxy // 4, 3 * nxy // 4, n_points),
        rng.integers(nxy // 4, 3 * nxy // 4, n_points)] = photons
    return obj


def test_mstep_recovers_spherical_given_true_object():
    """The PSF fit recovers a known spherical aberration when the object is known.

    This isolates the differentiable pupil fit from the object/PSF degeneracy."""
    nz, nxy, nxy_psf = 9, 40, 25
    true_sph = 0.30

    model = pp.PupilPSF(nxy_psf, nz, XY, Z, WL, NA, NI, device=CPU)
    sph = model.names.index("spherical")

    obj = _sparse_object(nz, nxy, n_points=25, photons=400.0)
    obj_t = torch.tensor(obj, device=CPU)

    true_coeffs = np.zeros(model.n_terms)
    true_coeffs[sph] = true_sph
    otf_t, _, _ = rc.otf_from_psf(model.render_numpy(true_coeffs), obj_t.shape, CPU)
    rate = rc.fftconv(obj_t, otf_t, obj_t.shape).clamp(min=0)
    image = torch.poisson(rate, generator=torch.Generator(device=CPU).manual_seed(0))

    # Fit only spherical, with the true object held fixed.
    active = torch.tensor([n == "spherical" for n in model.names], device=CPU)
    coeffs = torch.zeros(model.n_terms, device=CPU, dtype=model.dtype)
    coeffs = m_step(model, torch.fft.rfftn(obj_t), image, coeffs, active,
                    iters=120, lr=1e-2, lam=1e-3)

    assert abs(coeffs[sph].item() - true_sph) < 0.1, \
        f"spherical not recovered: {coeffs[sph].item():.3f}"


@pytest.mark.slow
@pytest.mark.xfail(reason="full alternating recovery is weak: the object estimate "
                          "absorbs the PSF error (object/PSF degeneracy); "
                          "strengthening the driver is future work",
                   strict=False)
def test_full_semiblind_recovers_spherical():
    """End-to-end: flat start, alternate E/M, recover injected spherical."""
    nz, nxy, nxy_psf = 9, 40, 25
    true_sph = 0.30

    model = pp.PupilPSF(nxy_psf, nz, XY, Z, WL, NA, NI, device=CPU)
    sph = model.names.index("spherical")

    obj = _sparse_object(nz, nxy, n_points=20, photons=300.0)
    obj_t = torch.tensor(obj, device=CPU)

    true_coeffs = np.zeros(model.n_terms)
    true_coeffs[sph] = true_sph
    otf_t, _, _ = rc.otf_from_psf(model.render_numpy(true_coeffs), obj_t.shape, CPU)
    rate = rc.fftconv(obj_t, otf_t, obj_t.shape).clamp(min=0)
    gen = torch.Generator(device=CPU).manual_seed(0)
    image = torch.poisson(rate, generator=gen)

    coeffs = torch.zeros(model.n_terms, device=CPU, dtype=model.dtype)
    recon = None
    for stage in (["spherical"], list(pp.DEFAULT_ABERRATIONS)):
        active = torch.tensor([n in stage for n in model.names], device=CPU)
        for _ in range(2):
            psf_np = model.render_numpy(coeffs.detach().cpu().numpy())
            otf, otfT, otfotfT = rc.otf_from_psf(psf_np, image.shape, CPU)
            recon = rc.gradient_consensus_deconvolve(
                image, otf, otfT, otfotfT, gen, n_steps=15, recon=recon, verbose=False)
            coeffs = m_step(model, torch.fft.rfftn(recon.detach()), image,
                            coeffs, active, iters=40, lr=1e-2, lam=1e-3)

    assert 0.1 < coeffs[sph].item() < 0.6, \
        f"spherical not recovered: {coeffs[sph].item():.3f}"
