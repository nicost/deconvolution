"""A 3D PSF built from a *pupil function*, so its aberrations can be fitted.

make_airy_psf.py makes the PSF of a *perfect* objective.  Real objectives are
not perfect: light does not all arrive exactly in step, and the small timing
errors (the "wavefront aberration") blur and distort the PSF.  The biggest
offender in practice is **spherical aberration**, which appears whenever the
refractive index the objective was designed for (e.g. immersion oil, 1.515)
does not match the sample (e.g. a watery cell, 1.33).

This module builds the PSF a different way than make_airy_psf.py, so that we
*can* add aberrations and, later, fit them to real data:

  1. Describe the objective by its *pupil*: a disk in the back focal plane.
     Each point of the disk is one plane wave the objective collects.
  2. Give every point a phase.  Part of the phase is defocus (how the PSF
     changes above and below focus); the rest is the aberration, written as a
     sum of **Zernike polynomials** (the standard "alphabet" of aberrations:
     defocus, astigmatism, coma, spherical, ...).
  3. The light field in one image plane is the 2D Fourier transform of the
     pupil; the PSF is its intensity (|field|^2).  Stack the planes over z.

Everything is done with PyTorch, so the PSF is a *differentiable* function of
the Zernike coefficients.  That is the whole point: deconvolve_semiblind.py can
then ask PyTorch "which way should I nudge the coefficients to make this PSF fit
the data better?" and let gradient descent do the rest.

With all coefficients zero this reproduces make_airy_psf.py's ideal PSF; check
that with:

    uv run src/deconvolution/pupil_psf.py --validate

Render a PSF with some aberration (coefficients are in waves RMS):

    uv run src/deconvolution/pupil_psf.py --aberration spherical=0.3 --output PSF_sph.tif

Note: a single PSF is used for the whole z-stack.  Real spherical aberration
from an index mismatch grows with depth into the sample, so this is an
approximation (a good one for thin samples).
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np

# Let Apple GPUs hand calculations they don't support to the CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch  # noqa: E402

TWO_PI = 2.0 * np.pi

# Named aberrations and their Noll index.  Piston (1), tip/tilt (2, 3) and
# defocus (4) are left out on purpose: they only shift or rescale the PSF, which
# is indistinguishable from moving the sample, so there is nothing to fit.
ABERRATION_NOLL = {
    "astig_oblique": 5,         # 45-degree astigmatism
    "astig_vertical": 6,        # 0/90-degree astigmatism
    "coma_vertical": 7,
    "coma_horizontal": 8,
    "trefoil_oblique": 9,
    "trefoil_vertical": 10,
    "spherical": 11,            # primary spherical: the index-mismatch aberration
    "secondary_astig_oblique": 12,
    "secondary_astig_vertical": 13,
    "secondary_spherical": 22,
}

# A sensible broad default: the low-order aberrations plus the two spherical
# terms.  Fitting starts with spherical only and adds the rest (see the driver).
DEFAULT_ABERRATIONS = [
    "spherical", "secondary_spherical",
    "astig_oblique", "astig_vertical",
    "coma_vertical", "coma_horizontal",
    "trefoil_oblique", "trefoil_vertical",
]


# --- Zernike polynomials --------------------------------------------------------

def noll_to_nm(j):
    """Convert a Noll index j (1, 2, 3, ...) to the Zernike orders (n, m).

    n is the radial order, m the azimuthal order (sign: + = cosine, - = sine)."""
    if j < 1:
        raise ValueError("Noll indices start at 1")
    n = int((-1.0 + np.sqrt(8 * (j - 1) + 1)) / 2.0)
    p = (j - (n * (n + 1)) // 2) - 1
    if n % 2 == 0:
        m = 2 * ((p + 1) // 2)
    else:
        m = 2 * (p // 2) + 1
    if j % 2:          # odd Noll index -> sine term (negative m by convention)
        m = -m
    return n, m


def radial_poly(n, m, rho):
    """The radial part R_n^m(rho) of a Zernike polynomial (m here is |m|)."""
    m = abs(m)
    r = np.zeros_like(rho)
    for s in range((n - m) // 2 + 1):
        c = ((-1) ** s * math.factorial(n - s)
             / (math.factorial(s)
                * math.factorial((n + m) // 2 - s)
                * math.factorial((n - m) // 2 - s)))
        r = r + c * rho ** (n - 2 * s)
    return r


def zernike(j, rho, phi, mask):
    """Zernike polynomial j (Noll index) on the pupil, RMS-normalized to 1.

    rho, phi: polar coordinates of the pupil (rho in [0, 1]); mask: the disk."""
    n, m = noll_to_nm(j)
    norm = np.sqrt(n + 1) if m == 0 else np.sqrt(2 * (n + 1))
    rad = radial_poly(n, m, rho)
    if m >= 0:
        z = norm * rad * np.cos(m * phi)
    else:
        z = norm * rad * np.sin(-m * phi)
    return z * mask


# --- Smallest fast FFT size -----------------------------------------------------

def next_fast_len(n):
    """Smallest even number >= n whose only prime factors are 2, 3, 5, 7
    (so the FFT is fast)."""
    m = n if n % 2 == 0 else n + 1
    while True:
        x = m
        for p in (2, 3, 5, 7):
            while x % p == 0:
                x //= p
        if x == 1:
            return m
        m += 2


# --- The differentiable pupil -> PSF model --------------------------------------

class PupilPSF:
    """Turn Zernike coefficients into a 3D PSF, differentiably.

    Build it once with the imaging settings; then call .render(coeffs) as often
    as you like (the optimizer calls it many times).  Coefficients are in waves
    RMS (one wave = a full 2*pi of wavefront error, root-mean-square over the
    pupil).
    """

    def __init__(self, nxy, nz, xy_um, z_um, wavelength_um, na, n_immersion,
                 aberrations=None, device=None, oversample=5.0, min_pupil_radius=40.0):
        if na >= n_immersion:
            raise ValueError("the NA must be smaller than the immersion index")
        self.device = device or torch.device("cpu")
        self.nxy, self.nz = nxy, nz
        self.xy_um, self.z_um = xy_um, z_um
        self.wavelength_um = wavelength_um
        self.na, self.n_immersion = na, n_immersion

        names = list(DEFAULT_ABERRATIONS if aberrations is None else aberrations)
        self.names = names
        noll = [ABERRATION_NOLL[a] if a in ABERRATION_NOLL else int(a) for a in names]

        # The pupil lives in spatial-frequency space.  To draw a Zernike phase on
        # it we need the pupil disk to be many pixels wide, which forces an
        # oversampled FFT grid (bigger than the PSF we finally want).
        dx = xy_um
        nu_max = na / wavelength_um                     # pupil edge, cycles/micron
        n_from_oversample = int(np.ceil(oversample * nxy))
        n_from_radius = int(np.ceil(min_pupil_radius / (nu_max * dx)))
        self.N = next_fast_len(max(n_from_oversample, n_from_radius))
        self.R_pix = nu_max * self.N * dx               # pupil radius, pixels

        # Radial spatial frequency on the FFT grid (cycles/micron), origin at [0,0].
        f = np.fft.fftfreq(self.N, d=dx)
        FX, FY = np.meshgrid(f, f, indexing="ij")
        f_r = np.hypot(FX, FY)
        mask = (f_r <= nu_max).astype(np.float64)
        rho = np.where(mask > 0, f_r / nu_max, 0.0)     # normalized radius [0, 1]
        phi = np.arctan2(FY, FX)

        # s = n*sin(theta) = NA*rho; the defocus phase is k*z*sqrt(n^2 - s^2).
        s = na * rho
        sqrt_term = np.sqrt(np.clip(n_immersion ** 2 - s ** 2, 0.0, None)) * mask
        k = TWO_PI / wavelength_um
        z = (np.arange(nz) - nz // 2) * z_um
        defocus = k * z[:, None, None] * sqrt_term[None]   # [nz, N, N], radians

        # Zernike basis, precomputed once as [n_terms, N, N].
        basis = np.stack([zernike(j, rho, phi, mask) for j in noll]) if noll \
            else np.zeros((0, self.N, self.N))

        # Move the fixed parts onto the compute device.  float32 keeps it fast on
        # GPUs (including Apple MPS, which does not support float64).
        self.dtype = torch.float32
        self.mask = torch.tensor(mask, device=self.device, dtype=self.dtype)
        self.defocus = torch.tensor(defocus, device=self.device, dtype=self.dtype)
        self.basis = torch.tensor(basis, device=self.device, dtype=self.dtype)
        self.n_terms = len(noll)
        self.crop_start = self.N // 2 - nxy // 2           # keeps the peak centered

    def render(self, coeffs):
        """PSF for the given Zernike coefficients (torch tensor, waves RMS).

        Returns a [nz, nxy, nxy] tensor that sums to 1 and is differentiable in
        'coeffs'."""
        if self.n_terms:
            coeffs = coeffs.to(self.basis.dtype)
            aberration = TWO_PI * torch.tensordot(coeffs, self.basis, dims=1)
        else:
            aberration = torch.zeros_like(self.mask)
        phase = self.defocus + aberration[None]            # [nz, N, N]
        pupil = self.mask * torch.exp(1j * phase)
        planes = torch.fft.ifft2(pupil, dim=(-2, -1))
        inten = planes.real ** 2 + planes.imag ** 2
        inten = torch.fft.fftshift(inten, dim=(-2, -1))
        s = self.crop_start
        psf = inten[:, s:s + self.nxy, s:s + self.nxy]
        return psf / psf.sum()

    def render_numpy(self, coeffs=None):
        """Convenience: render to a plain numpy array (for saving or plotting)."""
        if coeffs is None:
            coeffs = np.zeros(self.n_terms)
        c = torch.tensor(np.asarray(coeffs), device=self.device, dtype=self.dtype)
        with torch.no_grad():
            return self.render(c).cpu().numpy().astype(np.float32)


def odd_size(length_um, step_um):
    """Number of pixels covering the length, odd so there is a center pixel."""
    n = int(round(length_um / step_um))
    return n + 1 if n % 2 == 0 else n


# --- Command line ---------------------------------------------------------------

def _save_psf_tiff(psf, output, xy_um, z_um):
    import tifffile
    tifffile.imwrite(output, psf.astype(np.float32), imagej=True,
                     resolution=(1 / xy_um, 1 / xy_um),
                     metadata={"axes": "ZYX", "spacing": z_um, "unit": "micron"})


def _show_psf(psf, xy_um, z_um, title, save_to=None):
    import matplotlib.pyplot as plt
    nz, nxy, _ = psf.shape
    half_xy, half_z = nxy * xy_um / 2, nz * z_um / 2
    fig, axes = plt.subplots(2, 1, figsize=(4, 7), height_ratios=[nxy * xy_um, nz * z_um])
    axes[0].imshow(psf[nz // 2], cmap="magma", extent=[-half_xy, half_xy, -half_xy, half_xy])
    axes[0].set_title(f"{title}: XY in focus")
    axes[0].set_ylabel("y (µm)")
    axes[1].imshow(np.sqrt(psf[:, nxy // 2, :]), cmap="magma", aspect="auto",
                   extent=[-half_xy, half_xy, -half_z, half_z])
    axes[1].set_title("XZ (square root of intensity)")
    axes[1].set_xlabel("x (µm)")
    axes[1].set_ylabel("z (µm)")
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=120)
        print(f"Saved {save_to}")
    plt.show()


def _validate(model, args):
    """Compare the zero-aberration pupil PSF with make_airy_psf's ideal PSF."""
    from make_airy_psf import airy_psf_3d
    import matplotlib.pyplot as plt

    pupil_psf = model.render_numpy()                       # coeffs all zero
    airy = airy_psf_3d(args.wavelength / 1000, args.na, args.n_immersion,
                       args.pixel_size, args.z_step, model.nxy, model.nz)
    airy = airy / airy.sum()

    nrmse = np.linalg.norm(pupil_psf - airy) / np.linalg.norm(airy)
    print(f"Pupil FFT grid: {model.N} x {model.N}, pupil radius {model.R_pix:.1f} pixels")
    print(f"Normalized RMSE between pupil PSF and make_airy_psf PSF: {nrmse:.4f}")
    print("(should be small, a few percent, limited by FFT sampling)")

    nz, nxy, _ = pupil_psf.shape
    mid = nxy // 2
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(airy[nz // 2, mid], label="make_airy_psf")
    axes[0].plot(pupil_psf[nz // 2, mid], "--", label="pupil PSF")
    axes[0].set_title("In-focus profile (XY)")
    axes[0].set_xlabel("pixel")
    axes[0].legend()
    axes[1].plot(airy[:, mid, mid], label="make_airy_psf")
    axes[1].plot(pupil_psf[:, mid, mid], "--", label="pupil PSF")
    axes[1].set_title("Axial profile (Z through center)")
    axes[1].set_xlabel("z plane")
    axes[1].legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D PSF from a Zernike pupil (ideal when no aberrations given).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--na", type=float, default=1.40)
    parser.add_argument("--n-immersion", type=float, default=1.515)
    parser.add_argument("--pixel-size", type=float, default=0.0651, help="microns")
    parser.add_argument("--z-step", type=float, default=0.26, help="microns")
    parser.add_argument("--wavelength", type=float, default=605, help="nanometers")
    parser.add_argument("--size-xy", type=float, default=4.0, help="microns")
    parser.add_argument("--size-z", type=float, default=8.0, help="microns")
    parser.add_argument("--aberration", action="append", default=[], metavar="NAME=WAVES",
                        help="add an aberration, e.g. --aberration spherical=0.3 "
                             "(repeatable); names: " + ", ".join(ABERRATION_NOLL))
    parser.add_argument("--oversample", type=float, default=5.0,
                        help="how much bigger the pupil FFT grid is than the PSF")
    parser.add_argument("--validate", action="store_true",
                        help="compare the ideal (no-aberration) PSF to make_airy_psf.py")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    wavelength_um = args.wavelength / 1000
    nxy = odd_size(args.size_xy, args.pixel_size)
    nz = odd_size(args.size_z, args.z_step)

    # Parse the requested aberrations into (names, values).
    names, values = [], []
    for item in args.aberration:
        name, _, val = item.partition("=")
        names.append(name.strip())
        values.append(float(val))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PupilPSF(nxy, nz, args.pixel_size, args.z_step, wavelength_um,
                     args.na, args.n_immersion,
                     aberrations=names or DEFAULT_ABERRATIONS, device=device,
                     oversample=args.oversample)

    if args.validate:
        _validate(model, args)
        raise SystemExit

    # Map the given values onto the model's coefficient vector (0 for the rest).
    coeffs = np.zeros(model.n_terms)
    for name, val in zip(names, values):
        coeffs[model.names.index(name)] = val
    psf = model.render_numpy(coeffs)

    output = args.output or Path(f"PSF_pupil_{args.wavelength:g}nm.tif")
    _save_psf_tiff(psf, output, args.pixel_size, args.z_step)
    print(f"Saved {output}")
    if names:
        print("Aberrations (waves RMS): "
              + ", ".join(f"{n}={v:+.3f}" for n, v in zip(names, values)))
    _show_psf(psf, args.pixel_size, args.z_step,
              f"{args.wavelength:g} nm, NA {args.na}", save_to=output.with_suffix(".png"))
