"""Self-test for the semi-blind deconvolution: can it recover a known aberration?

This makes its own data, so it needs no microscope files.  For two kinds of
sample -- sparse bright points, and dense smooth "blobs" that look more like
real cell structures -- it:

  1. blurs the sample with a PSF that has a *known* amount of spherical (and
     astigmatism) aberration, and adds photon (Poisson) noise -- this is the
     "measurement",
  2. runs the same model-selection search as deconvolve_semiblind.py (try
     candidate PSFs, keep the one that best predicts held-out photons),
  3. checks the result:
       - sparse points: the injected aberration is recovered, and the
         deconvolved object is closer to the truth than with the ideal PSF;
       - dense blobs: dense, smooth samples carry little information about the
         aberration, so the fit may fall back towards the ideal PSF.  What must
         not happen is *overshooting*: fitting more aberration than is there,
         which over-sharpens the image and makes it worse than the ideal PSF.

The blob sample matters: the older "sharpest result" criterion passed on sparse
points, but on dense samples it fits far too much aberration and does worse
than the ideal PSF -- exactly what happened on real cells.

Run it with:

    uv run src/validate_semiblind_synthetic.py

It prints the injected vs recovered coefficients and a PASS/FAIL summary.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from pupil_psf import DEFAULT_ABERRATIONS, PupilPSF
from deconvolve_semiblind import deconvolve_with, fit_coefficients, render_psf
from rl_core import fftconv, otf_from_psf, pick_device, torch

# Small volume so it runs in a reasonable time, even on a CPU.
NZ, NXY, NXY_PSF = 15, 64, 45
XY_UM, Z_UM = 0.0651, 0.26
WAVELENGTH_UM, NA, N_IMMERSION = 0.605, 1.40, 1.515
N_POINTS = 40
PHOTONS = 400.0
SEED = 0

# The aberration we secretly put into the data (waves RMS).
TRUE_ABERRATION = {"spherical": 0.40, "astig_vertical": 0.15}


def normalized_rmse(a, b):
    """RMSE between two volumes after scaling each to sum 1 (brightness-blind)."""
    a = a / a.sum()
    b = b / b.sum()
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def sparse_points(rng):
    """Random bright points in the middle of the volume."""
    obj = np.zeros((NZ, NXY, NXY), dtype=np.float32)
    obj[rng.integers(NZ // 4, 3 * NZ // 4, N_POINTS),
        rng.integers(NXY // 4, 3 * NXY // 4, N_POINTS),
        rng.integers(NXY // 4, 3 * NXY // 4, N_POINTS)] = PHOTONS
    return obj


def dense_blobs(rng):
    """Smooth, densely packed structures on a dim background (cell-like)."""
    noise = gaussian_filter(rng.random((NZ, NXY, NXY)), sigma=(1.0, 2.5, 2.5))
    obj = np.clip(noise - np.percentile(noise, 70), 0, None)
    return (obj * (30.0 / obj.mean()) + 2.0).astype(np.float32)


def run_case(name, obj, model, true_coeffs, device, must_recover=True):
    """Make noisy data from 'obj', fit the PSF, and report. Returns True if OK.

    must_recover=False only requires no overshoot (see the module docstring)."""
    obj_t = torch.tensor(obj, device=device)
    otf_t, _, _ = otf_from_psf(render_psf(model, true_coeffs), obj_t.shape, device)
    blurred = fftconv(obj_t, otf_t, obj_t.shape).clamp(min=0)
    image = torch.poisson(blurred, generator=torch.Generator(device=device).manual_seed(SEED))
    print(f"\n=== {name}: image total {image.sum().item():.0f} photons ===")

    print("Fitting the PSF aberrations by model selection...")
    fitted = fit_coefficients(model, image, device, refine=True)
    recon_sb = deconvolve_with(model, fitted, image, device, n_steps=None)
    # Baseline: deconvolve with the WRONG (ideal) PSF.
    recon_ideal = deconvolve_with(model, np.zeros(model.n_terms), image, device,
                                  n_steps=None)

    print("\nAberration coefficients (waves RMS):")
    print(f"  {'name':24s} {'injected':>10s} {'recovered':>10s}")
    for term, inj, rec in zip(model.names, true_coeffs, fitted):
        mark = "  <--" if abs(inj) > 1e-6 else ""
        print(f"  {term:24s} {inj:>+10.3f} {rec:>+10.3f}{mark}")

    rmse_sb = normalized_rmse(recon_sb.cpu().numpy(), obj)
    rmse_ideal = normalized_rmse(recon_ideal.cpu().numpy(), obj)
    print("\nObject error (normalized RMSE vs truth):")
    print(f"  ideal-PSF deconvolution : {rmse_ideal:.4f}")
    print(f"  semi-blind deconvolution: {rmse_sb:.4f}")

    sph = model.names.index("spherical")
    sph_err = abs(fitted[sph] - true_coeffs[sph])
    if must_recover:
        coeff_ok = sph_err < 0.12
        print(f"Spherical recovered within {sph_err:.3f} waves (want < 0.12): "
              f"{'OK' if coeff_ok else 'FAIL'}")
    else:
        over = fitted[sph] - true_coeffs[sph]
        coeff_ok = over < 0.12
        print(f"Spherical overshoot {max(over, 0):.3f} waves (want < 0.12): "
              f"{'OK' if coeff_ok else 'FAIL'}")
    rmse_ok = rmse_sb <= rmse_ideal + 1e-6
    print(f"Semi-blind beats ideal-PSF object error: {'OK' if rmse_ok else 'FAIL'}")
    return coeff_ok and rmse_ok


if __name__ == "__main__":
    device = pick_device()
    print(f"Calculating on: {device}")
    rng = np.random.default_rng(SEED)

    model = PupilPSF(NXY_PSF, NZ, XY_UM, Z_UM, WAVELENGTH_UM, NA, N_IMMERSION,
                     aberrations=DEFAULT_ABERRATIONS, device=device)
    true_coeffs = np.zeros(model.n_terms)
    for term, val in TRUE_ABERRATION.items():
        true_coeffs[model.names.index(term)] = val

    results = {name: run_case(name, make(rng), model, true_coeffs, device, strict)
               for name, make, strict in (("sparse points", sparse_points, True),
                                          ("dense blobs", dense_blobs, False))}

    print("\nSUMMARY:")
    for name, ok in results.items():
        print(f"  {name:14s} {'PASS' if ok else 'FAIL'}")
    print("OVERALL:", "PASS" if all(results.values()) else "FAIL")
