"""Self-test for the semi-blind deconvolution: can it recover a known aberration?

This makes its own data, so it needs no microscope files.  It:

  1. builds a sparse 3D object (random bright points),
  2. blurs it with a PSF that has a *known* amount of spherical (and astigmatism)
     aberration, and adds photon (Poisson) noise -- this is the "measurement",
  3. runs the same alternating E/M loop as deconvolve_semiblind.py, starting from
     the ideal (no-aberration) PSF, and checks that it recovers the aberration
     coefficients we put in, and that the deconvolved object is closer to the
     truth than deconvolving with the wrong (ideal) PSF.

Run it with:

    uv run src/deconvolution/validate_semiblind_synthetic.py

It prints the injected vs recovered coefficients and a PASS/FAIL summary.
"""

import numpy as np

from pupil_psf import DEFAULT_ABERRATIONS, PupilPSF
from deconvolve_semiblind import m_step
from rl_core import (fftconv, gradient_consensus_deconvolve, otf_from_psf,
                     pick_device, torch)

# Small volume so it runs quickly, even on a CPU.
NZ, NXY = 25, 64
NXY_PSF = 45
XY_UM, Z_UM = 0.0651, 0.26
WAVELENGTH_UM, NA, N_IMMERSION = 0.605, 1.40, 1.515
N_POINTS = 40
PHOTONS = 400.0                   # brightness of each point (more photons = easier)
SEED = 0

# The aberration we secretly put into the data (waves RMS).
TRUE_ABERRATION = {"spherical": 0.40, "astig_vertical": 0.15}

# Coarse-to-fine fitting schedule (same idea as the real driver).
FIT_SCHEDULE = [["spherical"], ["spherical", "secondary_spherical"],
                list(DEFAULT_ABERRATIONS)]
ROUNDS_PER_STAGE, E_STEPS, M_ITERS, ADAM_LR, L2 = 2, 15, 50, 1e-2, 1e-3


def normalized_rmse(a, b):
    """RMSE between two volumes after scaling each to sum 1 (brightness-blind)."""
    a = a / a.sum()
    b = b / b.sum()
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def run_semiblind(model, image, generator, device):
    """The alternating loop, returning (recon, fitted coeffs)."""
    coeffs = torch.zeros(model.n_terms, device=device, dtype=model.dtype)
    recon = None
    for stage_names in FIT_SCHEDULE:
        active = torch.tensor([n in stage_names for n in model.names], device=device)
        for _ in range(ROUNDS_PER_STAGE):
            psf_np = model.render_numpy(coeffs.detach().cpu().numpy())
            otf, otfT, otfotfT = otf_from_psf(psf_np, image.shape, device)
            recon = gradient_consensus_deconvolve(
                image, otf, otfT, otfotfT, generator,
                n_steps=E_STEPS, recon=recon, verbose=False)
            recon_fft = torch.fft.rfftn(recon.detach())
            coeffs = m_step(model, recon_fft, image, coeffs, active,
                            M_ITERS, ADAM_LR, L2)
    # Final run to convergence with the fitted PSF.
    psf_np = model.render_numpy(coeffs.detach().cpu().numpy())
    otf, otfT, otfotfT = otf_from_psf(psf_np, image.shape, device)
    recon = gradient_consensus_deconvolve(
        image, otf, otfT, otfotfT, generator, n_steps=None, recon=recon, verbose=False)
    return recon, coeffs


if __name__ == "__main__":
    device = pick_device()
    print(f"Calculating on: {device}")
    rng = np.random.default_rng(SEED)

    # --- A sparse 3D object of bright points ------------------------------------
    obj = np.zeros((NZ, NXY, NXY), dtype=np.float32)
    zs = rng.integers(NZ // 4, 3 * NZ // 4, N_POINTS)
    ys = rng.integers(NXY // 4, 3 * NXY // 4, N_POINTS)
    xs = rng.integers(NXY // 4, 3 * NXY // 4, N_POINTS)
    obj[zs, ys, xs] = PHOTONS
    obj_t = torch.tensor(obj, device=device)

    # --- Build the true (aberrated) PSF and make noisy data ---------------------
    model = PupilPSF(NXY_PSF, NZ, XY_UM, Z_UM, WAVELENGTH_UM, NA, N_IMMERSION,
                     aberrations=DEFAULT_ABERRATIONS, device=device)
    true_coeffs = np.zeros(model.n_terms)
    for name, val in TRUE_ABERRATION.items():
        true_coeffs[model.names.index(name)] = val
    psf_true = model.render_numpy(true_coeffs)
    otf_t, _, _ = otf_from_psf(psf_true, obj_t.shape, device)
    blurred = fftconv(obj_t, otf_t, obj_t.shape).clamp(min=0)
    generator = torch.Generator(device=device).manual_seed(SEED)
    image = torch.poisson(blurred, generator=generator)
    print(f"Object: {N_POINTS} points, {PHOTONS:.0f} photons each; "
          f"image total {image.sum().item():.0f} photons")

    # --- Semi-blind deconvolution (fits the PSF) --------------------------------
    print("Running semi-blind deconvolution...")
    recon_sb, fitted = run_semiblind(model, image, generator, device)

    # --- Baseline: deconvolve with the WRONG (ideal) PSF ------------------------
    psf_ideal = model.render_numpy(np.zeros(model.n_terms))
    otf, otfT, otfotfT = otf_from_psf(psf_ideal, image.shape, device)
    gen2 = torch.Generator(device=device).manual_seed(SEED)
    recon_fixed = gradient_consensus_deconvolve(
        image, otf, otfT, otfotfT, gen2, n_steps=None, verbose=False)

    # --- Report -----------------------------------------------------------------
    print("\nAberration coefficients (waves RMS):")
    print(f"  {'name':24s} {'injected':>10s} {'recovered':>10s}")
    fitted_list = fitted.detach().cpu().tolist()
    for name, inj, rec in zip(model.names, true_coeffs, fitted_list):
        mark = "  <--" if abs(inj) > 1e-6 else ""
        print(f"  {name:24s} {inj:>+10.3f} {rec:>+10.3f}{mark}")

    rmse_sb = normalized_rmse(recon_sb.cpu().numpy(), obj)
    rmse_fixed = normalized_rmse(recon_fixed.cpu().numpy(), obj)
    print(f"\nObject error (normalized RMSE vs truth):")
    print(f"  ideal-PSF deconvolution : {rmse_fixed:.4f}")
    print(f"  semi-blind deconvolution: {rmse_sb:.4f}")

    # --- Pass/fail --------------------------------------------------------------
    sph_i = model.names.index("spherical")
    sph_err = abs(fitted_list[sph_i] - true_coeffs[sph_i])
    coeff_ok = sph_err < 0.1
    rmse_ok = rmse_sb < rmse_fixed
    print(f"\nSpherical recovered within {sph_err:.3f} waves (want < 0.1): "
          f"{'OK' if coeff_ok else 'FAIL'}")
    print(f"Semi-blind beats ideal-PSF object error: {'OK' if rmse_ok else 'FAIL'}")
    print("\nOVERALL:", "PASS" if (coeff_ok and rmse_ok) else "FAIL")
