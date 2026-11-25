# Microscope deconvolution: PSF modelling, gradient-consensus, and semi-blind fitting

Tools for deconvolving 3D fluorescence microscopy z-stacks, and for building the
point spread function (PSF) they need — up to and including **semi-blind
deconvolution**, which fits the PSF's aberrations to the data while it
deconvolves.

Everything is plain Python using PyTorch, and runs on a GPU (NVIDIA CUDA or Apple
MPS) when one is available, or on the CPU otherwise.

## What's here

| Script | What it does |
|---|---|
| `make_airy_psf.py` | Generate the theoretical PSF of an **ideal** objective (a scalar Airy/Debye model) from the imaging conditions. |
| `pupil_psf.py` | Generate a PSF from a **pupil function** with Zernike aberrations. Differentiable, so aberrations can be fitted. Also validates itself against `make_airy_psf.py`. |
| `deconvolve_gradient_consensus.py` | Richardson-Lucy deconvolution with **gradient consensus** (auto-stopping, resists amplifying noise) and Biggs-Andrews acceleration. Needs a PSF file. |
| `deconvolve_semiblind.py` | **Semi-blind** deconvolution: deconvolves the image *and* fits a few Zernike aberrations of the PSF (e.g. spherical aberration from a refractive-index mismatch) at the same time. |
| `validate_semiblind_synthetic.py` | A self-test with simulated data: injects a known aberration and checks it is recovered. No microscope files needed. |
| `rl_core.py` | Shared engine (the forward model and the deconvolution loop) imported by the scripts above. Not run directly. |

## The idea in one paragraph

A microscope image is the true object **convolved** with the microscope's PSF,
plus photon noise. Deconvolution tries to undo that convolution. Richardson-Lucy
does it by repeatedly guessing the object, blurring the guess with the PSF, and
comparing to the recorded image. Plain Richardson-Lucy eventually "deconvolves"
the noise; the **gradient-consensus** variant (Andrew York; code by James Manton)
avoids this by splitting the photons into two halves each iteration and only
updating pixels where both halves agree. All of that needs a *correct* PSF —
and the ideal PSF is usually a bit wrong, mostly because of **spherical
aberration** when the immersion medium (e.g. oil, n≈1.515) does not match the
sample (e.g. a watery cell, n≈1.33). `deconvolve_semiblind.py` closes that gap by
fitting the PSF's aberrations within a physically valid pupil model. It does this
by **model selection**: it deconvolves the image with several candidate PSFs and
keeps the one whose result is sharpest (a wrong PSF leaves rings and haze). This
side-steps the object/PSF degeneracy that defeats naive "deconvolve, then nudge
the PSF" schemes, where the object estimate simply absorbs the PSF error.

## Requirements

- Python 3.12 or 3.13
- `torch`, `numpy`, `scipy`, `matplotlib`, `tifffile`

These are declared in `pyproject.toml`; the recommended way to run is with
[uv](https://docs.astral.sh/uv/), which creates the environment for you
automatically. (For viewing the 3D results, [napari](https://napari.org) is handy
but not required.)

## Running

The scripts live in `src/` and import each other as siblings, so run them **from
the repository root** with the `src/` path — uv (or Python) then puts `src/` on
the import path:

```bash
uv run src/make_airy_psf.py
```

or, if you have already installed the dependencies into your environment:

```bash
python src/make_airy_psf.py
```

Every script prints a short usage summary and its options with `--help`.

### 1. Make a theoretical PSF

```bash
# Default: Cy3 (605 nm) on a 100x / 1.40 oil objective
uv run src/make_airy_psf.py

# For other imaging conditions, e.g. NA 1.2 water, 500 nm, 0.08 um pixels, 0.3 um z-step
uv run src/make_airy_psf.py --na 1.2 --n-immersion 1.33 --wavelength 500 \
    --pixel-size 0.08 --z-step 0.3 --output PSF_water_500nm.tif
```

This writes `PSF_<wavelength>nm.tif` (a 32-bit z-stack with the pixel size and
z-step stored in the file) plus a `.png` preview (XY in focus and XZ side view).
**Use the same pixel size and z-step as the images you want to deconvolve.**

`pupil_psf.py` can make the same PSF (and is the one used for fitting). Check that
the two agree, and render an aberrated PSF:

```bash
uv run src/pupil_psf.py --validate                     # ideal PSF vs make_airy_psf.py
uv run src/pupil_psf.py --aberration spherical=0.3 --output PSF_spherical.tif
```

### 2. Deconvolve with a fixed PSF

```bash
# Give the folder with the data; optionally the PSF file (default PSF_605nm.tif)
uv run src/deconvolve_gradient_consensus.py /path/to/BPAE_100x_ZStack_3Crop
uv run src/deconvolve_gradient_consensus.py /path/to/data PSF_605nm.tif
uv run src/deconvolve_gradient_consensus.py /path/to/data --output /path/to/results
```

It deconvolves one channel of an OME-TIFF z-stack, saves the result as
`..._deconvolved.tif` in a folder next to the data folder (see
[Where results are saved](#where-results-are-saved)), and shows before/after maximum-intensity
projections. The data defaults, channel, and a `CROP_XY` option (use the center
of the image for a quick test on a CPU) are constants near the top of the file.

### 3. Semi-blind deconvolution (fit the PSF too)

```bash
uv run src/deconvolve_semiblind.py /path/to/BPAE_100x_ZStack_3Crop
uv run src/deconvolve_semiblind.py /path/to/BPAE_100x_ZStack_3Crop --output /path/to/results
```

No separate PSF file is needed — the imaging conditions (NA, immersion index,
wavelength) are set near the top of the file. It searches a small set of Zernike
aberrations by trying candidate PSFs and keeping the sharpest result (spherical
first via a 1D scan, then the rest refined together). It prints the fitted
coefficients (in waves RMS) and saves both the **fitted PSF**
(`..._PSF_fitted.tif`) and the **deconvolved stack**
(`..._semiblind_deconvolved.tif`). Each candidate costs one short deconvolution,
so on a CPU set `CROP_XY` near the top of the file to try a small center crop
first.

### 4. Check it works (no data required)

```bash
uv run src/validate_semiblind_synthetic.py
```

This simulates a sparse object, blurs it with a PSF that has a **known**
aberration, adds photon noise, runs the semi-blind method, and reports the
injected vs recovered coefficients plus a PASS/FAIL. It is the quickest way to
confirm the install is working end to end.

## Data format

The deconvolution scripts expect an **OME-TIFF** z-stack whose OME metadata
records `PhysicalSizeX` and `PhysicalSizeZ` (microns) — standard for
Micro-Manager and many acquisition tools. The PSF and all outputs are written as
ImageJ TIFFs that carry the pixel size and z-step, so Fiji/ImageJ and these
scripts read them back correctly. Edit the `IMAGE_NAME`, `CHANNEL`, and imaging
constants at the top of a script to match your data.

### Where results are saved

Results are **never written inside the data folder**: Micro-Manager treats every
TIFF in a dataset folder as part of the dataset, so extra files there break
reading the data back. By default the deconvolution scripts save to a sibling
folder named `<data folder>_deconvolved`, e.g. `/path/to/BPAE_100x_ZStack_3Crop`
→ `/path/to/BPAE_100x_ZStack_3Crop_deconvolved/`. Use `-o/--output` to choose
another folder; a folder inside the data folder is refused.

## Notes and limitations

- The semi-blind model uses **one PSF for the whole stack**. Real spherical
  aberration from an index mismatch grows with depth; this is a good
  approximation for thin samples but an approximation nonetheless.
- The aberration search is controlled by constants near the top of
  `deconvolve_semiblind.py`: `ABERRATIONS` (which aberrations to fit),
  `SPHERICAL_SCAN` (values tried for spherical), `SCORE_STEPS`, and
  `REFINE_OTHERS`. Each candidate PSF costs one short deconvolution, so fitting
  many aberrations on a large stack is slow — trim `ABERRATIONS` and/or set
  `CROP_XY` to keep it quick, especially without a GPU.
- The sharpness score rewards concentrated reconstructions, which suits
  point-like and filamentous samples well; for very dense, uniform samples it is
  less discriminating.
- A GPU makes this much faster. Without one it still works; use `CROP_XY` to keep
  the first runs small.

## Tests

Fast, CPU-only tests live in [`tests/`](tests/) and need no data. From the
repository root:

```bash
uv run pytest                 # run everything
uv run pytest -m "not slow"   # skip the few-second end-to-end test
```

See [`tests/README.md`](tests/README.md) for details and what each file covers.

## Credit and references

- Gradient-consensus deconvolution: Andrew G. York and colleagues
  (doi:10.5281/zenodo.10278918).
- Local, GPU-accelerated implementation adapted from James Manton's (MRC LMB)
  gradient-consensus Richardson-Lucy notebook.
- Biggs-Andrews acceleration: Biggs & Andrews, *Applied Optics* 36, 1766 (1997).
- Richardson-Lucy: en.wikipedia.org/wiki/Richardson–Lucy_deconvolution
