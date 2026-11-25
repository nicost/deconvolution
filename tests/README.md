# Tests

Fast, deterministic tests for the deconvolution code. They run on the CPU and
need no microscope data (the few that need an image make a tiny synthetic one).

## Running

From the repository root:

```bash
uv run pytest                 # run everything
uv run pytest -q              # quieter output
uv run pytest -m "not slow"   # skip the few-second end-to-end test
uv run pytest -v              # verbose: list every test
```

Run a single file or test:

```bash
uv run pytest tests/test_pupil_psf.py
uv run pytest tests/test_pupil_psf.py::test_ideal_psf_matches_airy -v
```

(If you have already installed the dependencies into an active environment, you
can drop the `uv run` and just use `pytest`.)

`pytest` is declared in the `dev` dependency group in `pyproject.toml`, which uv
installs automatically, and `pythonpath = ["src"]` lets the tests import the
scripts from `src/`.

## What is covered

| File | Checks |
|---|---|
| `test_zernike.py` | Zernike polynomials: Noll index mapping, RMS normalization, the closed form for spherical, and orthogonality. |
| `test_pupil_psf.py` | The pupil→PSF model: PSF sums to 1, is non-negative and centered; the zero-aberration PSF matches `make_airy_psf`; `render()` is differentiable; spherical aberration breaks axial symmetry. |
| `test_rl_core.py` | The deconvolution engine: PSF centering/normalization, KL divergence, FFT convolution with a delta is the identity, the two OTF code paths agree, and deconvolution actually sharpens a blurred image. |
| `test_semiblind.py` | The sharpness score prefers the correct PSF, and the full model-selection search recovers a known injected spherical aberration (the search is marked `slow`). |

## Notes

- The tolerances are deliberately loose enough to avoid flakiness but tight
  enough to catch real regressions. The `test_ideal_psf_matches_airy` bound
  (cosine similarity > 0.95, NRMSE < 0.15) absorbs the small difference between
  the pupil-FFT model and the Bessel-integral model.
- Everything is seeded for reproducibility.
- A GPU is not used; the tests force the CPU so results are deterministic.
