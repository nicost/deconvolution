"""Tests for the shared deconvolution engine (rl_core.py)."""
import numpy as np
import torch

import rl_core as rc

CPU = torch.device("cpu")


def _norm_rmse(a, b):
    """RMSE after scaling each volume to sum 1 (brightness-blind)."""
    a = a / a.sum()
    b = b / b.sum()
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def _gaussian_psf(shape, sigma):
    """A small normalized 3D Gaussian blur kernel."""
    coords = [np.arange(n) - n // 2 for n in shape]
    zz, yy, xx = np.meshgrid(*coords, indexing="ij")
    g = np.exp(-(zz ** 2 + yy ** 2 + xx ** 2) / (2 * sigma ** 2))
    return (g / g.sum()).astype(np.float32)


def test_psf_as_image_normalized_and_centered():
    psf = _gaussian_psf((5, 7, 7), sigma=1.0)
    full = rc.psf_as_image(psf, (9, 16, 16))
    assert abs(full.sum() - 1.0) < 1e-5
    # The PSF center must end up at the array origin (for FFT convolution).
    assert np.unravel_index(np.argmax(full), full.shape) == (0, 0, 0)


def test_kldiv_zero_for_identical():
    x = torch.rand(8, 8) + 0.1
    assert abs(rc.kldiv(x, x)) < 1e-6


def test_fftconv_with_delta_is_identity():
    shape = (9, 16, 16)
    delta = np.zeros((5, 5, 5), dtype=np.float32)
    delta[2, 2, 2] = 1.0                       # a point source at the center
    otf, _, _ = rc.otf_from_psf(delta, shape, CPU)
    image = torch.rand(shape, device=CPU) + 0.5
    out = rc.fftconv(image, otf, shape)
    assert torch.allclose(out, image, atol=1e-4)


def test_otf_paths_agree():
    """otf_from_psf (numpy input) and psf_to_otf_differentiable (torch input)
    must build the same OTF."""
    shape = (9, 16, 16)
    psf = _gaussian_psf((5, 9, 9), sigma=1.3)
    otf_np, _, _ = rc.otf_from_psf(psf, shape, CPU)
    otf_torch = rc.psf_to_otf_differentiable(torch.tensor(psf, device=CPU), shape)
    assert torch.allclose(otf_np, otf_torch, atol=1e-5)


def test_deconvolution_sharpens():
    """Deconvolving a blurred image should bring it closer to the true object."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    shape = (9, 32, 32)

    obj = np.zeros(shape, dtype=np.float32)
    zs = rng.integers(2, 7, 20)
    ys = rng.integers(6, 26, 20)
    xs = rng.integers(6, 26, 20)
    obj[zs, ys, xs] = 300.0
    obj_t = torch.tensor(obj, device=CPU)

    psf = _gaussian_psf((5, 11, 11), sigma=1.6)
    otf, otfT, otfotfT = rc.otf_from_psf(psf, shape, CPU)
    blurred = rc.fftconv(obj_t, otf, shape).clamp(min=0)

    gen = torch.Generator(device=CPU).manual_seed(0)
    recon = rc.gradient_consensus_deconvolve(
        blurred, otf, otfT, otfotfT, gen, n_steps=25, verbose=False)

    assert recon.shape == obj_t.shape
    assert recon.min().item() >= 0.0
    rmse_blur = _norm_rmse(blurred.numpy(), obj)
    rmse_recon = _norm_rmse(recon.numpy(), obj)
    assert rmse_recon < rmse_blur, f"did not sharpen: {rmse_recon:.3f} vs {rmse_blur:.3f}"


def test_deconvolve_runs_fixed_number_of_steps():
    """With n_steps set, the loop returns without consulting the stop criterion."""
    shape = (5, 16, 16)
    psf = _gaussian_psf((3, 7, 7), sigma=1.0)
    otf, otfT, otfotfT = rc.otf_from_psf(psf, shape, CPU)
    image = (torch.rand(shape, device=CPU) * 50).clamp(min=0)
    gen = torch.Generator(device=CPU).manual_seed(1)
    recon = rc.gradient_consensus_deconvolve(
        image, otf, otfT, otfotfT, gen, n_steps=3, verbose=False)
    assert torch.isfinite(recon).all()
