"""Semi-blind deconvolution: deconvolve the image *and* fit the PSF at once.

deconvolve_gradient_consensus.py needs a PSF, and make_airy_psf.py gives it the
PSF of a *perfect* objective.  But real objectives are not perfect: an index
mismatch between the immersion medium and the sample (very common) adds
spherical aberration, so the ideal PSF is a bit wrong and the deconvolution
suffers.  We usually do not know the exact aberration.

"Semi-blind" means: we do not let the PSF be anything at all (that would be
hopeless), but we do let it vary within the physically possible family described
by a pupil with a few Zernike aberrations (see pupil_psf.py), and fit those few
numbers to the data.

How we fit them.  A naive approach -- deconvolve, then nudge the PSF to fit the
same image better -- does not work: the deconvolved image simply absorbs the PSF
error, so there is nothing left to fit (the "object/PSF degeneracy" of blind
deconvolution).  Instead we use *model selection*: we try different aberrations,
deconvolve the image with each candidate PSF, and keep the one whose result is
**sharpest**.  A wrong PSF leaves rings and haze (a blurry, spread-out result); the
right PSF gives a crisp one.  "Sharpness" is measured as sum(u^2) / sum(u)^2,
which is large when the light is concentrated and small when it is spread out.

We search spherical aberration first (the big one, and just one number, so a
quick 1D scan finds it reliably), then refine the remaining aberrations together.

Run it (after the data is in place) with:

    uv run src/deconvolve_semiblind.py
    uv run src/deconvolve_semiblind.py ~/Images/BPAE_100x_ZStack_3Crop
    uv run src/deconvolve_semiblind.py ~/Images/BPAE_100x_ZStack_3Crop --output ~/Results

The fitted aberrations are printed and before/after projections are shown.  The
refined PSF and the deconvolved stack are saved in a folder next to the data
folder, "<data folder>_deconvolved" (or the folder given with --output).  They
are never saved inside the data folder: Micro-Manager would then fail to read
the dataset back.  Each candidate
PSF costs one (short) deconvolution, so without a GPU set CROP_XY below and/or
trim ABERRATIONS to keep the search quick.
"""

import argparse
import re
import timeit
from pathlib import Path

import numpy as np
import tifffile

from pupil_psf import (DEFAULT_ABERRATIONS, PupilPSF, _save_psf_tiff, _show_psf,
                       odd_size)
from rl_core import (gradient_consensus_deconvolve, otf_from_psf, output_dir_for,
                     pick_device, show_mips, torch)

DATA_DIR = Path("~/Images/BPAE_100x_ZStack_3Crop").expanduser()
IMAGE_NAME = "BPAE_100x_ZStack_3_MMStack_Pos0.ome.tif"
CHANNEL = 0                       # Cy3 is the first channel in the stack

# Imaging conditions (these are not in the image file, so set them here).
NA = 1.40                         # objective numerical aperture
N_IMMERSION = 1.515               # immersion oil (the sample is more like 1.33,
                                  # which is exactly what creates spherical aberration)
WAVELENGTH_NM = 605               # Cy3 emission

PSF_SIZE_XY_UM = 4.0              # width/height of the PSF (microns)

# Which aberrations may be fitted (waves RMS).  Spherical is always searched
# first with a 1D scan; the rest are refined together afterwards.  Trim this list
# to make the search faster.
ABERRATIONS = DEFAULT_ABERRATIONS

# Spherical-aberration values tried in the first (1D) scan.
SPHERICAL_SCAN = np.round(np.arange(-0.3, 0.61, 0.1), 3)
SCORE_STEPS = 50                  # deconvolution steps used when scoring a candidate
REFINE_OTHERS = True              # after spherical, refine the other aberrations
REFINE_MAXITER = 3               # Powell passes over the other aberrations
COEFF_BOUND = 1.0                 # keep |coefficient| <= this many waves

# Use only the center of the image (e.g. CROP_XY = 256) to try things out quickly.
CROP_XY = None
SEED = 0


def sharpness(u):
    """How concentrated (sharp) the deconvolved image is: sum(u^2) / sum(u)^2.

    Large when the light sits in a few bright pixels, small when it is smeared
    out.  The correct PSF gives the sharpest result."""
    u = u.astype(np.float64)
    s = u.sum()
    return float((u * u).sum() / (s * s + 1e-12))


def deconvolve_with(model, coeffs, image, device, n_steps=None):
    """Deconvolve 'image' using the PSF for these Zernike coefficients.

    A fresh, identically-seeded generator is used so that comparing two
    candidates is fair (same random photon splits)."""
    psf_np = model.render_numpy(np.asarray(coeffs))
    otf, otfT, otfotfT = otf_from_psf(psf_np, image.shape, device)
    gen = torch.Generator(device=device).manual_seed(SEED)
    return gradient_consensus_deconvolve(
        image, otf, otfT, otfotfT, gen, n_steps=n_steps, verbose=False)


def score(model, coeffs, image, device):
    """Lower is better: the negative sharpness of a fixed-length deconvolution.

    Candidates are penalized for straying outside +/- COEFF_BOUND waves."""
    coeffs = np.asarray(coeffs, dtype=float)
    if np.any(np.abs(coeffs) > COEFF_BOUND):
        return 1e9 + float(np.sum(np.abs(coeffs)))
    u = deconvolve_with(model, coeffs, image, device, n_steps=SCORE_STEPS)
    return -sharpness(u.cpu().numpy())


def _parabola_min(x, y):
    """Vertex of the parabola through three points (for sub-grid refinement)."""
    denom = y[0] - 2 * y[1] + y[2]
    if denom == 0:
        return x[1]
    return x[1] + 0.5 * (y[0] - y[2]) / denom * (x[1] - x[0])


def fit_coefficients(model, image, device, refine=REFINE_OTHERS, verbose=True):
    """Find the Zernike coefficients whose PSF deconvolves the image sharpest."""
    coeffs = np.zeros(model.n_terms)
    sph = model.names.index("spherical")

    # --- 1D scan over spherical aberration ---
    scores = []
    for c in SPHERICAL_SCAN:
        coeffs[sph] = c
        scores.append(score(model, coeffs, image, device))
        if verbose:
            print(f"  spherical = {c:+.2f}  score = {scores[-1]:.4e}")
    i = int(np.argmin(scores))
    best = SPHERICAL_SCAN[i]
    if 0 < i < len(SPHERICAL_SCAN) - 1:        # refine between grid points
        best = _parabola_min(SPHERICAL_SCAN[i - 1:i + 2], scores[i - 1:i + 2])
    coeffs[sph] = best
    if verbose:
        print(f"  -> spherical = {best:+.3f}")

    # --- Refine the remaining aberrations together ---
    if refine and model.n_terms > 1:
        from scipy.optimize import minimize
        if verbose:
            print("  refining the other aberrations (Powell)...")
        res = minimize(lambda v: score(model, v, image, device), coeffs,
                       method="Powell", bounds=[(-COEFF_BOUND, COEFF_BOUND)] * model.n_terms,
                       options={"maxiter": REFINE_MAXITER, "xtol": 1e-2, "ftol": 1e-3})
        if res.fun <= score(model, coeffs, image, device):
            coeffs = res.x
    return coeffs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Semi-blind deconvolution: fit PSF aberrations and deconvolve.")
    parser.add_argument("data_dir", nargs="?", type=Path, default=DATA_DIR,
                        help=f"Micro-Manager data folder (default: {DATA_DIR})")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="folder for the results (default: '<data_dir>_deconvolved' "
                             "next to the data folder; may not be inside the data folder)")
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser()
    out_dir = output_dir_for(data_dir, args.output)   # checked before the slow fit

    # --- Load the image ---------------------------------------------------------
    with tifffile.TiffFile(data_dir / IMAGE_NAME) as tif:
        image_np = tif.series[0].asarray()[CHANNEL]           # axes: z, y, x
        ome = tif.ome_metadata
        xy_um = float(re.search(r'PhysicalSizeX="([\d.eE+-]+)"', ome).group(1))
        z_um = float(re.search(r'PhysicalSizeZ="([\d.eE+-]+)"', ome).group(1))

    if CROP_XY:
        y0 = (image_np.shape[1] - CROP_XY) // 2
        x0 = (image_np.shape[2] - CROP_XY) // 2
        image_np = image_np[:, y0:y0 + CROP_XY, x0:x0 + CROP_XY]

    nz, ny, nx = image_np.shape
    device = pick_device()
    print(f"Image: {nz} x {ny} x {nx} pixels (z, y, x), calculating on: {device}")

    # --- Build the differentiable PSF model -------------------------------------
    nxy_psf = min(odd_size(PSF_SIZE_XY_UM, xy_um), nx if nx % 2 else nx - 1)
    model = PupilPSF(nxy_psf, nz, xy_um, z_um, WAVELENGTH_NM / 1000, NA, N_IMMERSION,
                     aberrations=ABERRATIONS, device=device)
    print(f"PSF model: {nz} x {nxy_psf} x {nxy_psf}, pupil {model.N} px "
          f"(radius {model.R_pix:.0f}), fitting {model.names}")

    image = torch.tensor(image_np.astype(np.float32), device=device)

    # --- Fit the PSF by model selection, then deconvolve to convergence ---------
    start = timeit.default_timer()
    print("Fitting the PSF aberrations...")
    coeffs = fit_coefficients(model, image, device)
    print("Deconvolving with the fitted PSF...")
    recon = deconvolve_with(model, coeffs, image, device, n_steps=None)
    print(f"Total time: {timeit.default_timer() - start:.1f} s")

    print("\nFitted aberrations (waves RMS):")
    for name, val in zip(model.names, coeffs):
        print(f"  {name:24s} {val:+.4f}")

    # --- Save the refined PSF and the deconvolved stack -------------------------
    stem = Path(IMAGE_NAME).name.removesuffix('.ome.tif')
    psf_np = model.render_numpy(coeffs)
    psf_out = out_dir / f"{stem}_Cy3_PSF_fitted.tif"
    _save_psf_tiff(psf_np, psf_out, xy_um, z_um)
    print(f"Saved {psf_out}")

    recon_np = recon.cpu().numpy().astype(np.float32)
    recon_out = out_dir / f"{stem}_Cy3_semiblind_deconvolved.tif"
    tifffile.imwrite(recon_out, recon_np, imagej=True, resolution=(1 / xy_um, 1 / xy_um),
                     metadata={"axes": "ZYX", "spacing": z_um, "unit": "micron"})
    print(f"Saved {recon_out}")

    _show_psf(psf_np, xy_um, z_um, "Fitted PSF", save_to=psf_out.with_suffix(".png"))
    show_mips(image_np, recon_np)
