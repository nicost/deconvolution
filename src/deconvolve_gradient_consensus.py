"""Deconvolve the Cy3 channel of the BPAE z-stack with the PSF from make_airy_psf.py.

Make the PSF first (once), then run this from the top folder of the repository:

    uv run src/deconvolution/make_airy_psf.py
    uv run src/deconvolution/deconvolve_gradient_consensus.py

The first command saves PSF_605nm.tif in the top folder, where this script
looks for it.  If your copy of the data is somewhere else, give the folder
(and, if you like, a different PSF file):

    uv run src/deconvolution/deconvolve_gradient_consensus.py "D:/my data/BPAE_100x_ZStack_3Crop"
    uv run src/deconvolution/deconvolve_gradient_consensus.py "D:/my data/BPAE_100x_ZStack_3Crop" PSF_605nm.tif

The method is Richardson-Lucy deconvolution, adapted from the Colab notebook by
James Manton (MRC LMB) and colleagues (deconvolution/gradientConsensusDeconvolution.py).
Richardson-Lucy improves a guess of the "true" image step by step:

  1. blur the guess with the PSF: this is what the microscope would have shown
  2. compare with the recorded image (divide recorded by predicted)
  3. use that ratio to correct the guess, and repeat

Plain Richardson-Lucy never says when to stop, and if you go on too long it
starts to "deconvolve" the noise.  This version decides by itself:

  - Every step, the recorded image is split at random into two half images
    (each photon goes to one half or the other).  Both halves show the same
    sample but have different noise.
  - The guess is only corrected where both halves agree on the direction of
    the correction ("gradient consensus").  Where they disagree, the
    correction is caused by noise and is skipped.
  - When the prediction stops getting closer to the half images, more steps
    would only add noise, so it stops.
  - "Biggs-Andrews acceleration" takes larger steps in the direction the guess
    was already moving, so fewer steps are needed.

The calculations run on the graphics card (GPU) if there is one, which is much
faster.  Without a GPU it works too, but slowly: set CROP_XY below to try it on
the center of the image first.

The result is saved next to the data as a 32-bit z-stack with the same pixel
size, and maximum intensity projections of before and after are shown.

The heart of the method (the forward model and the iteration loop) now lives in
rl_core.py, so that other scripts can reuse exactly the same code.
"""

import re
import sys
from pathlib import Path

import numpy as np
import tifffile

from rl_core import (gradient_consensus_deconvolve, otf_from_psf, pick_device,
                     show_mips, torch)

DATA_DIR = Path("Z:/ImageAnalysisCourse/ZStack_100x_BPAE/BPAE_100x_ZStack_3Crop")
IMAGE_NAME = "BPAE_100x_ZStack_3_MMStack_Pos0.ome.tif"
CHANNEL = 0                       # Cy3 is the first channel in the stack
PSF_FILE = Path("PSF_605nm.tif")  # made by make_airy_psf.py (for Cy3: 605 nm)

# Use only the center of the image (e.g. CROP_XY = 512) to try things out quickly.
CROP_XY = None

# Stop as soon as one of the half images gets worse (True), or only when both
# do (False).  True is the safer choice and gives fewer artifacts.
PLAY_IT_SAFE = True
SEED = 0                          # makes the random splits the same every run


if __name__ == "__main__":
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR
    psf_file = Path(sys.argv[2]) if len(sys.argv) > 2 else PSF_FILE

    # --- Load the image and the PSF ---------------------------------------------
    with tifffile.TiffFile(data_dir / IMAGE_NAME) as tif:
        image_np = tif.series[0].asarray()[CHANNEL]           # axes: z, y, x
        # Pixel size and z-step (in microns), to copy to the result.
        ome = tif.ome_metadata
        xy_um = float(re.search(r'PhysicalSizeX="([\d.eE+-]+)"', ome).group(1))
        z_um = float(re.search(r'PhysicalSizeZ="([\d.eE+-]+)"', ome).group(1))
    with tifffile.TiffFile(psf_file) as tif:
        psf_small = tif.asarray()
        # TIFF stores the resolution as pixels per micron, written as a fraction.
        pixels, microns = tif.pages[0].tags["XResolution"].value
        psf_xy_um = microns / pixels
        psf_z_um = tif.imagej_metadata.get("spacing", 1.0)

    # The PSF must have the same pixel size and z-step as the image, or the
    # deconvolution uses a PSF that is too large or too small.
    if not (np.isclose(psf_xy_um, xy_um, rtol=0.01) and np.isclose(psf_z_um, z_um, rtol=0.01)):
        print(f"Warning: the PSF has pixel size {psf_xy_um:.4f} um and z-step {psf_z_um} um, "
              f"but the image has {xy_um} um and {z_um} um.\n"
              f"Make a new PSF with: --pixel-size {xy_um} --z-step {z_um}")

    if CROP_XY:
        y0 = (image_np.shape[1] - CROP_XY) // 2
        x0 = (image_np.shape[2] - CROP_XY) // 2
        image_np = image_np[:, y0:y0 + CROP_XY, x0:x0 + CROP_XY]

    device = pick_device()
    print(f"Image: {image_np.shape[0]} x {image_np.shape[1]} x {image_np.shape[2]} pixels "
          f"(z, y, x), PSF: {psf_file.name}, calculating on: {device}")

    # --- Prepare the calculation ------------------------------------------------
    image = torch.tensor(image_np.astype(np.float32), device=device)
    otf, otfT, otfotfT = otf_from_psf(psf_small, image_np.shape, device)
    generator = torch.Generator(device=device).manual_seed(SEED)

    # --- Richardson-Lucy iterations ---------------------------------------------
    recon = gradient_consensus_deconvolve(image, otf, otfT, otfotfT, generator,
                                          n_steps=None, play_it_safe=PLAY_IT_SAFE)

    # --- Save the result with the same pixel size as the original ---------------
    recon_np = recon.cpu().numpy().astype(np.float32)
    output = data_dir / f"{Path(IMAGE_NAME).name.removesuffix('.ome.tif')}_Cy3_deconvolved.tif"
    tifffile.imwrite(output, recon_np, imagej=True, resolution=(1 / xy_um, 1 / xy_um),
                     metadata={"axes": "ZYX", "spacing": z_um, "unit": "micron"})
    print(f"Saved {output}")

    show_mips(image_np, recon_np)
