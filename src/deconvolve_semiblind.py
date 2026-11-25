"""Semi-blind deconvolution: deconvolve the image *and* fit the PSF at once.

deconvolve_gradient_consensus.py needs a PSF, and make_airy_psf.py gives it the
PSF of a *perfect* objective.  But real objectives are not perfect: an index
mismatch between the immersion medium and the sample (very common) adds
spherical aberration, so the ideal PSF is a bit wrong and the deconvolution
suffers.  We usually do not know the exact aberration.

"Semi-blind" means: we do not let the PSF be anything at all (that would be
hopeless), but we do let it vary within the physically possible family described
by a pupil with a few Zernike aberrations (see pupil_psf.py).  We then fit those
few numbers to the data.  The recipe alternates two steps:

  E-step: with the current PSF, run gradient-consensus Richardson-Lucy to improve
          the deconvolved image (the same method as deconvolve_gradient_consensus).
  M-step: with the current image, nudge the Zernike coefficients so that
          (image * PSF) matches the recorded data better.  Because pupil_psf.py
          is differentiable, PyTorch finds the right nudge for us (Adam).

We go coarse to fine: fit spherical aberration first (the big one), then let the
other aberrations join in.  Run it (after the data is in place) with:

    uv run src/deconvolution/deconvolve_semiblind.py
    uv run src/deconvolution/deconvolve_semiblind.py ~/Images/BPAE_100x_ZStack_3Crop

The fitted aberrations are printed; the refined PSF and the deconvolved stack are
saved next to the data, and before/after projections are shown.  Without a GPU
set CROP_XY below to try the center of the image first.
"""

import re
import sys
import timeit
from pathlib import Path

import numpy as np
import tifffile

from pupil_psf import (DEFAULT_ABERRATIONS, PupilPSF, _save_psf_tiff, _show_psf,
                       odd_size)
from rl_core import (gradient_consensus_deconvolve, otf_from_psf, pick_device,
                     psf_to_otf_differentiable, show_mips, torch)

DATA_DIR = Path("~/Images/BPAE_100x_ZStack_3Crop").expanduser()
IMAGE_NAME = "BPAE_100x_ZStack_3_MMStack_Pos0.ome.tif"
CHANNEL = 0                       # Cy3 is the first channel in the stack

# Imaging conditions (these are not in the image file, so set them here).
NA = 1.40                         # objective numerical aperture
N_IMMERSION = 1.515               # immersion oil (the sample is more like 1.33,
                                  # which is exactly what creates spherical aberration)
WAVELENGTH_NM = 605               # Cy3 emission

PSF_SIZE_XY_UM = 4.0              # width/height of the PSF (microns)

# Which aberrations may be fitted, and in what order they are switched on
# (coarse to fine): spherical first, then both spherical terms, then everything.
ABERRATIONS = DEFAULT_ABERRATIONS
FIT_SCHEDULE = [
    ["spherical"],
    ["spherical", "secondary_spherical"],
    list(DEFAULT_ABERRATIONS),
]
ROUNDS_PER_STAGE = 2              # E/M rounds spent at each schedule stage
E_STEPS = 10                      # gradient-consensus steps per E-step
M_ITERS = 40                      # Adam steps per M-step
ADAM_LR = 1e-2                    # learning rate for the Zernike coefficients
L2_LAMBDA = 1e-3                  # small pull toward zero, to stay physical

# Use only the center of the image (e.g. CROP_XY = 256) to try things out quickly.
CROP_XY = None
SEED = 0


def poisson_nll(pred, image):
    """How badly 'pred' explains the recorded 'image', assuming photon (Poisson)
    noise.  Smaller is better.  (Constant terms that do not depend on pred are
    dropped.)"""
    pred = pred.clamp(min=1e-6)
    return (pred - image * torch.log(pred)).sum()


def m_step(model, image_fft_recon, image, coeffs, active, iters, lr, lam):
    """Adjust the Zernike coefficients to fit the data, with the image fixed.

    image_fft_recon : rfftn of the current deconvolved image (precomputed)
    active          : bool tensor, which coefficients are allowed to move now"""
    shape = image.shape
    c = coeffs.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([c], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        psf = model.render(c)                       # differentiable PSF, sums to 1
        otf = psf_to_otf_differentiable(psf, shape)
        pred = torch.fft.irfftn(image_fft_recon * otf, s=shape)
        loss = poisson_nll(pred, image) + lam * (c * c).sum()
        loss.backward()
        with torch.no_grad():
            c.grad[~active] = 0                     # freeze the switched-off aberrations
        opt.step()
    return c.detach()


if __name__ == "__main__":
    data_dir = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DATA_DIR

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
    # The PSF spans the whole z-stack; its width is PSF_SIZE_XY_UM (but never
    # wider than the image).
    nxy_psf = min(odd_size(PSF_SIZE_XY_UM, xy_um), nx if nx % 2 else nx - 1)
    model = PupilPSF(nxy_psf, nz, xy_um, z_um, WAVELENGTH_NM / 1000, NA, N_IMMERSION,
                     aberrations=ABERRATIONS, device=device)
    print(f"PSF model: {nz} x {nxy_psf} x {nxy_psf}, pupil {model.N} px "
          f"(radius {model.R_pix:.0f}), fitting {model.names}")

    image = torch.tensor(image_np.astype(np.float32), device=device)
    generator = torch.Generator(device=device).manual_seed(SEED)
    coeffs = torch.zeros(model.n_terms, device=device, dtype=model.dtype)

    def otf_for(coeffs):
        """Build the three Fourier kernels the E-step needs, for these coeffs."""
        psf_np = model.render_numpy(coeffs.detach().cpu().numpy())
        return otf_from_psf(psf_np, image.shape, device)

    # --- Alternating E (image) / M (PSF) optimization ---------------------------
    start = timeit.default_timer()
    recon = None
    for stage, stage_names in enumerate(FIT_SCHEDULE, 1):
        active = torch.tensor([n in stage_names for n in model.names], device=device)
        for r in range(ROUNDS_PER_STAGE):
            # E-step: improve the image with the current PSF.
            otf, otfT, otfotfT = otf_for(coeffs)
            recon = gradient_consensus_deconvolve(
                image, otf, otfT, otfotfT, generator,
                n_steps=E_STEPS, recon=recon, verbose=False)
            # M-step: improve the PSF with the current image.
            recon_fft = torch.fft.rfftn(recon.detach())
            coeffs = m_step(model, recon_fft, image, coeffs, active,
                            M_ITERS, ADAM_LR, L2_LAMBDA)
            shown = ", ".join(f"{n}={v:+.3f}" for n, v in zip(model.names, coeffs.tolist())
                              if n in stage_names)
            print(f"Stage {stage} (fitting {stage_names}), round {r + 1}: {shown}")

    # --- Final E-step: run the image to convergence with the fitted PSF ---------
    otf, otfT, otfotfT = otf_for(coeffs)
    recon = gradient_consensus_deconvolve(
        image, otf, otfT, otfotfT, generator, n_steps=None, recon=recon, verbose=True)
    print(f"Total time: {timeit.default_timer() - start:.1f} s")

    print("\nFitted aberrations (waves RMS):")
    for name, val in zip(model.names, coeffs.tolist()):
        print(f"  {name:24s} {val:+.4f}")

    # --- Save the refined PSF and the deconvolved stack -------------------------
    stem = Path(IMAGE_NAME).name.removesuffix('.ome.tif')
    psf_np = model.render_numpy(coeffs.detach().cpu().numpy())
    psf_out = data_dir / f"{stem}_Cy3_PSF_fitted.tif"
    _save_psf_tiff(psf_np, psf_out, xy_um, z_um)
    print(f"Saved {psf_out}")

    recon_np = recon.cpu().numpy().astype(np.float32)
    recon_out = data_dir / f"{stem}_Cy3_semiblind_deconvolved.tif"
    tifffile.imwrite(recon_out, recon_np, imagej=True, resolution=(1 / xy_um, 1 / xy_um),
                     metadata={"axes": "ZYX", "spacing": z_um, "unit": "micron"})
    print(f"Saved {recon_out}")

    _show_psf(psf_np, xy_um, z_um, "Fitted PSF", save_to=psf_out.with_suffix(".png"))
    show_mips(image_np, recon_np)
