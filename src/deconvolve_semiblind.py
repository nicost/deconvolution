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
deconvolution).  Instead we use *model selection*: we try different aberrations
and keep the PSF that best **predicts data it has not seen**:

  - Split the recorded image once into two half images (each photon goes to one
    half or the other).  Both show the same sample, with independent noise.
  - Deconvolve half A with the candidate PSF, blur the result with that PSF
    again, and measure how well this prediction matches half B (the Poisson
    likelihood).  Then swap A and B, and average.

A PSF that is too *broad* can only reproduce the fine detail of half A by also
reproducing its noise, which does not carry over to half B, so it scores badly.
A PSF that is too *narrow* can partly mimic the right one (the deconvolved
image just stays a bit blurrier), so when the data is ambiguous the fit leans
towards the ideal, unaberrated PSF: it should not do much worse than the
theoretical PSF, and only moves away from it when the data clearly asks.

(Earlier versions kept the PSF whose deconvolution was *sharpest*.  That works
for sparse points, but on real cells it picks PSFs that are far too broad, which
over-sharpen the image into speckles.  Set SCORE = "sharpness" to compare.)

By default only spherical aberration (the big one, from the index mismatch) is
fitted, with a quick 1D scan.  More aberrations can be added to ABERRATIONS and
refined together afterwards (REFINE_OTHERS), but every extra free number makes
the fit less reliable.

The fitted PSF is re-centered in z: spherical aberration moves the plane of
best focus, which would otherwise shift the whole deconvolved stack in z.

Run it (after the data is in place) with:

    uv run src/deconvolve_semiblind.py
    uv run src/deconvolve_semiblind.py ~/Images/BPAE_100x_ZStack_3Crop
    uv run src/deconvolve_semiblind.py ~/Images/BPAE_100x_ZStack_3Crop --output ~/Results

The fitted aberrations are printed and before/after projections are shown.  The
refined PSF and the deconvolved stack are saved in a folder next to the data
folder, "<data folder>_deconvolved" (or the folder given with --output).  They
are never saved inside the data folder: Micro-Manager would then fail to read
the dataset back.  Each candidate PSF costs two short deconvolutions (of the
half images), so without a GPU set CROP_XY below to keep the search quick.
"""

import argparse
import re
import timeit
from pathlib import Path

import numpy as np
import tifffile

from pupil_psf import (DEFAULT_ABERRATIONS, PupilPSF, _save_psf_tiff, _show_psf,
                       odd_size)
from rl_core import (fftconv, gradient_consensus_deconvolve, otf_from_psf,
                     output_dir_for, pick_device, show_mips, torch)

DATA_DIR = Path("~/Images/BPAE_100x_ZStack_3Crop").expanduser()
IMAGE_NAME = "BPAE_100x_ZStack_3_MMStack_Pos0.ome.tif"
CHANNEL = 0                       # Cy3 is the first channel in the stack

# Imaging conditions (these are not in the image file, so set them here).
NA = 1.40                         # objective numerical aperture
N_IMMERSION = 1.515               # immersion oil (the sample is more like 1.33,
                                  # which is exactly what creates spherical aberration)
WAVELENGTH_NM = 605               # Cy3 emission

PSF_SIZE_XY_UM = 4.0              # width/height of the PSF (microns)
PSF_SIZE_Z_UM = 8.0               # depth of the PSF (microns), as in make_airy_psf.py

# Which aberrations may be fitted (waves RMS).  Spherical is always searched
# first with a 1D scan; any others are refined together afterwards (if
# REFINE_OTHERS).  Spherical alone is the most reliable choice for real data;
# to fit more, use e.g. ABERRATIONS = DEFAULT_ABERRATIONS, REFINE_OTHERS = True.
ABERRATIONS = ["spherical"]

# How candidate PSFs are compared: "heldout" (how well the PSF predicts an
# independent half of the photons; recommended) or "sharpness" (the older
# criterion, which picks PSFs that are too broad on non-sparse samples).
SCORE = "heldout"

# Spherical-aberration values tried in the first (1D) scan.
SPHERICAL_SCAN = np.round(np.arange(-0.5, 0.51, 0.1), 3)
SCORE_STEPS = 50                  # deconvolution steps used when scoring a candidate
                                  # (fixed; with much fewer the held-out score
                                  # underestimates the aberration)
REFINE_OTHERS = False             # after spherical, refine the other aberrations
REFINE_MAXITER = 3                # Powell passes over the other aberrations
COEFF_BOUND = 0.5                 # keep |coefficient| <= this many waves

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


def center_psf_z(psf):
    """Move the PSF along z so that its brightest plane is the center plane.

    Spherical aberration shifts the plane of best focus.  The deconvolution
    treats the PSF's center plane as "no shift", so an off-center PSF would
    shift the whole deconvolved stack in z (and wrap planes around)."""
    axial = psf.sum(axis=(1, 2))
    return np.roll(psf, psf.shape[0] // 2 - int(np.argmax(axial)), axis=0)


def render_psf(model, coeffs):
    """The PSF for these Zernike coefficients, re-centered in z."""
    return center_psf_z(model.render_numpy(np.asarray(coeffs)))


def deconvolve_with(model, coeffs, image, device, n_steps=None):
    """Deconvolve 'image' using the PSF for these Zernike coefficients.

    A fresh, identically-seeded generator is used so that comparing two
    candidates is fair (same random photon splits)."""
    otf, otfT, otfotfT = otf_from_psf(render_psf(model, coeffs), image.shape, device)
    gen = torch.Generator(device=device).manual_seed(SEED)
    return gradient_consensus_deconvolve(
        image, otf, otfT, otfotfT, gen, n_steps=n_steps, verbose=False)


def split_halves(image, seed=SEED + 1):
    """Split the image into two half images with independent photon noise."""
    gen = torch.Generator(device=image.device).manual_seed(seed)
    a = torch.binomial(image, torch.full_like(image, 0.5), generator=gen)
    return a, image - a


def heldout_score(model, coeffs, halves, device):
    """Lower is better: how badly the PSF predicts photons it was not fitted to.

    Deconvolve one half image, blur the result with the same PSF, and compute
    the Poisson negative log-likelihood of the *other* half; both ways round."""
    otf, otfT, otfotfT = otf_from_psf(render_psf(model, coeffs), halves[0].shape, device)
    total = 0.0
    for fit, test in (halves, halves[::-1]):
        gen = torch.Generator(device=device).manual_seed(SEED)
        u = gradient_consensus_deconvolve(fit, otf, otfT, otfotfT, gen,
                                          n_steps=SCORE_STEPS, verbose=False)
        mu = fftconv(u, otf, fit.shape).clamp(min=1e-6)      # predicted photons
        total += (mu - test * torch.log(mu)).mean().item()
    return total / 2


def score(model, coeffs, image, device, halves=None):
    """Lower is better: the held-out likelihood (SCORE = "heldout") or the
    negative sharpness of a fixed-length deconvolution (SCORE = "sharpness").

    Candidates are penalized for straying outside +/- COEFF_BOUND waves.
    'halves' (from split_halves) lets all candidates share one split."""
    coeffs = np.asarray(coeffs, dtype=float)
    if np.any(np.abs(coeffs) > COEFF_BOUND + 1e-9):
        return 1e9 + float(np.sum(np.abs(coeffs)))
    if SCORE == "heldout":
        return heldout_score(model, coeffs, halves or split_halves(image), device)
    if SCORE == "sharpness":
        u = deconvolve_with(model, coeffs, image, device, n_steps=SCORE_STEPS)
        return -sharpness(u.cpu().numpy())
    raise ValueError(f"unknown SCORE {SCORE!r}; use 'heldout' or 'sharpness'")


def _parabola_min(x, y):
    """Vertex of the parabola through three points (for sub-grid refinement)."""
    denom = y[0] - 2 * y[1] + y[2]
    if denom == 0:
        return x[1]
    return x[1] + 0.5 * (y[0] - y[2]) / denom * (x[1] - x[0])


def fit_coefficients(model, image, device, refine=REFINE_OTHERS, verbose=True):
    """Find the Zernike coefficients whose PSF scores best (see score())."""
    coeffs = np.zeros(model.n_terms)
    sph = model.names.index("spherical")
    halves = split_halves(image) if SCORE == "heldout" else None  # one split for all

    # --- 1D scan over spherical aberration ---
    scores = []
    for c in SPHERICAL_SCAN:
        coeffs[sph] = c
        scores.append(score(model, coeffs, image, device, halves))
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
        res = minimize(lambda v: score(model, v, image, device, halves), coeffs,
                       method="Powell", bounds=[(-COEFF_BOUND, COEFF_BOUND)] * model.n_terms,
                       options={"maxiter": REFINE_MAXITER, "xtol": 1e-2, "ftol": 1e-3})
        if res.fun <= score(model, coeffs, image, device, halves):
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
    nz_psf = min(odd_size(PSF_SIZE_Z_UM, z_um), nz if nz % 2 else nz - 1)
    model = PupilPSF(nxy_psf, nz_psf, xy_um, z_um, WAVELENGTH_NM / 1000, NA, N_IMMERSION,
                     aberrations=ABERRATIONS, device=device)
    print(f"PSF model: {nz_psf} x {nxy_psf} x {nxy_psf}, pupil {model.N} px "
          f"(radius {model.R_pix:.0f}), fitting {model.names}")

    image = torch.tensor(image_np.astype(np.float32), device=device)

    # --- Fit the PSF by model selection, then deconvolve to convergence ---------
    start = timeit.default_timer()
    print(f"Fitting the PSF aberrations (score: {SCORE}, lower is better)...")
    coeffs = fit_coefficients(model, image, device)
    print("Deconvolving with the fitted PSF...")
    recon = deconvolve_with(model, coeffs, image, device, n_steps=None)
    print(f"Total time: {timeit.default_timer() - start:.1f} s")

    print("\nFitted aberrations (waves RMS):")
    for name, val in zip(model.names, coeffs):
        print(f"  {name:24s} {val:+.4f}")

    # --- Save the refined PSF and the deconvolved stack -------------------------
    stem = Path(IMAGE_NAME).name.removesuffix('.ome.tif')
    psf_np = render_psf(model, coeffs)                 # re-centered, as used
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
