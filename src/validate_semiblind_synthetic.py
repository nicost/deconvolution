"""Self-test for the semi-blind deconvolution: can it recover a known aberration?

This makes its own data, so it needs no microscope files.  It:

  1. builds a sparse 3D object (random bright points),
  2. blurs it with a PSF that has a *known* amount of spherical (and astigmatism)
     aberration, and adds photon (Poisson) noise -- this is the "measurement",
  3. runs the same model-selection search as deconvolve_semiblind.py (try
     candidate PSFs, keep the one that deconvolves sharpest) and checks that it
     recovers the aberration we put in, and that the deconvolved object is closer
     to the truth than deconvolving with the wrong (ideal) PSF.

Run it with:

    uv run src/validate_semiblind_synthetic.py

It prints the injected vs recovered coefficients and a PASS/FAIL summary.
"""

import numpy as np

from pupil_psf import DEFAULT_ABERRATIONS, PupilPSF
from deconvolve_semiblind import deconvolve_with, fit_coefficients
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


if __name__ == "__main__":
    device = pick_device()
    print(f"Calculating on: {device}")
    rng = np.random.default_rng(SEED)

    # --- A sparse 3D object of bright points ------------------------------------
    obj = np.zeros((NZ, NXY, NXY), dtype=np.float32)
    obj[rng.integers(NZ // 4, 3 * NZ // 4, N_POINTS),
        rng.integers(NXY // 4, 3 * NXY // 4, N_POINTS),
        rng.integers(NXY // 4, 3 * NXY // 4, N_POINTS)] = PHOTONS
    obj_t = torch.tensor(obj, device=device)

    # --- Build the true (aberrated) PSF and make noisy data ---------------------
    model = PupilPSF(NXY_PSF, NZ, XY_UM, Z_UM, WAVELENGTH_UM, NA, N_IMMERSION,
                     aberrations=DEFAULT_ABERRATIONS, device=device)
    true_coeffs = np.zeros(model.n_terms)
    for name, val in TRUE_ABERRATION.items():
        true_coeffs[model.names.index(name)] = val
    otf_t, _, _ = otf_from_psf(model.render_numpy(true_coeffs), obj_t.shape, device)
    blurred = fftconv(obj_t, otf_t, obj_t.shape).clamp(min=0)
    gen = torch.Generator(device=device).manual_seed(SEED)
    image = torch.poisson(blurred, generator=gen)
    print(f"Object: {N_POINTS} points, {PHOTONS:.0f} photons each; "
          f"image total {image.sum().item():.0f} photons")

    # --- Semi-blind fit + deconvolution -----------------------------------------
    print("Fitting the PSF aberrations by model selection...")
    fitted = fit_coefficients(model, image, device, refine=True)
    recon_sb = deconvolve_with(model, fitted, image, device, n_steps=None)

    # --- Baseline: deconvolve with the WRONG (ideal) PSF ------------------------
    recon_ideal = deconvolve_with(model, np.zeros(model.n_terms), image, device,
                                  n_steps=None)

    # --- Report -----------------------------------------------------------------
    print("\nAberration coefficients (waves RMS):")
    print(f"  {'name':24s} {'injected':>10s} {'recovered':>10s}")
    for name, inj, rec in zip(model.names, true_coeffs, fitted):
        mark = "  <--" if abs(inj) > 1e-6 else ""
        print(f"  {name:24s} {inj:>+10.3f} {rec:>+10.3f}{mark}")

    rmse_sb = normalized_rmse(recon_sb.cpu().numpy(), obj)
    rmse_ideal = normalized_rmse(recon_ideal.cpu().numpy(), obj)
    print(f"\nObject error (normalized RMSE vs truth):")
    print(f"  ideal-PSF deconvolution : {rmse_ideal:.4f}")
    print(f"  semi-blind deconvolution: {rmse_sb:.4f}")

    # --- Pass/fail --------------------------------------------------------------
    sph = model.names.index("spherical")
    sph_err = abs(fitted[sph] - true_coeffs[sph])
    coeff_ok = sph_err < 0.12
    rmse_ok = rmse_sb <= rmse_ideal + 1e-6
    print(f"\nSpherical recovered within {sph_err:.3f} waves (want < 0.12): "
          f"{'OK' if coeff_ok else 'FAIL'}")
    print(f"Semi-blind beats ideal-PSF object error: {'OK' if rmse_ok else 'FAIL'}")
    print("\nOVERALL:", "PASS" if (coeff_ok and rmse_ok) else "FAIL")
