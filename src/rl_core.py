"""Shared building blocks for Richardson-Lucy gradient-consensus deconvolution.

These helpers used to live inside deconvolve_gradient_consensus.py.  They were
moved here so other scripts (for example the semi-blind deconvolution) can reuse
exactly the same forward model and the same iteration loop, instead of copying
the code.  Nothing about the algorithm changed; see the docstring of
deconvolve_gradient_consensus.py for the explanation of the method.

The most important function is gradient_consensus_deconvolve(): one call runs the
whole Richardson-Lucy gradient-consensus loop and returns the deconvolved image.
It can either run until it finds the optimum by itself (n_steps=None) or run a
fixed number of steps (used by the semi-blind code, which alternates between
improving the image and improving the PSF).
"""

import os
import timeit

import matplotlib.pyplot as plt
import numpy as np

# Let Apple GPUs hand calculations they don't support to the CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch  # noqa: E402


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def psf_as_image(psf_small, shape):
    """Put the small PSF in an image of zeros of the given size, with the PSF's
    center pixel at position (0, 0, 0), as the Fourier transform needs."""
    psf = np.zeros(shape, dtype=np.float32)
    psf[tuple(slice(0, n) for n in psf_small.shape)] = psf_small
    psf = np.roll(psf, [-(n // 2) for n in psf_small.shape], axis=(0, 1, 2))
    return psf / psf.sum()


def fftconv(x, H, shape):
    """Convolve x with a kernel, using its Fourier transform H."""
    return torch.fft.irfftn(torch.fft.rfftn(x) * H, s=shape)


def kldiv(p, q):
    """Kullback-Leibler divergence: how different two images are (0 = the same)."""
    p = p + 1e-4
    q = q + 1e-4
    p = p / p.sum()
    q = q / q.sum()
    kld = p * (torch.log(p) - torch.log(q))
    return torch.nan_to_num(kld).sum().item()


def otf_from_psf(psf_small, shape, device):
    """Turn a small real-space PSF into the three Fourier-space kernels the
    deconvolution needs: the OTF, its mirror image, and the two multiplied."""
    psf = torch.tensor(psf_as_image(psf_small, shape), device=device)
    otf = torch.fft.rfftn(psf)
    otfT = torch.conj(otf)
    otfotfT = otf * otfT
    return otf, otfT, otfotfT


def psf_to_otf_differentiable(psf_small, shape):
    """Like otf_from_psf, but for a PSF that is itself a torch tensor being
    optimized: the same zero-pad / center / normalize done with torch ops, so
    gradients flow back from the OTF to the PSF.  Returns just the OTF.

    psf_small : torch tensor [pz, py, px], the small PSF (must fit in 'shape')
    shape     : the full image shape (Z, Y, X)"""
    pz, py, px = psf_small.shape
    full = torch.zeros(shape, dtype=psf_small.dtype, device=psf_small.device)
    full[:pz, :py, :px] = psf_small
    full = torch.roll(full, shifts=(-(pz // 2), -(py // 2), -(px // 2)), dims=(0, 1, 2))
    full = full / full.sum()
    return torch.fft.rfftn(full)


def gradient_consensus_deconvolve(image, otf, otfT, otfotfT, generator,
                                  n_steps=None, play_it_safe=True,
                                  recon=None, verbose=True):
    """Richardson-Lucy deconvolution with gradient consensus and Biggs-Andrews
    acceleration (see deconvolve_gradient_consensus.py for the full explanation).

    image      : the recorded image, as a torch tensor on the compute device
    otf, otfT, otfotfT : Fourier-space PSF kernels from otf_from_psf()
    generator  : a torch.Generator, so the random photon splits are reproducible
    n_steps    : stop after this many iterations; None = run until the optimum
                 (one of the half images stops getting closer) is found
    play_it_safe : stop when one half image gets worse (True) or only when both
                 do (False).  Only used when n_steps is None.
    recon      : starting guess (default: a flat image at the average brightness)
    verbose    : print a line per iteration

    Returns the deconvolved image as a torch tensor.
    """
    shape = image.shape
    half = torch.full_like(image, 0.5)

    # Start with a flat guess: every pixel has the average brightness of the image.
    if recon is None:
        recon = torch.full_like(image, image.mean().item())
    previous_recon = recon

    num_iters = 0
    prev_kld1 = np.inf
    prev_kld2 = np.inf
    start_time = timeit.default_timer()

    # Biggs-Andrews acceleration: the last two changes of the guess
    g2 = torch.zeros_like(recon)
    g1 = torch.zeros_like(recon)

    while True:
        iter_start_time = timeit.default_timer()

        # Split recorded image into 50:50 images
        split1 = torch.binomial(image, half, generator=generator)
        split2 = image - split1

        # Biggs-Andrews acceleration: step further in the direction we were moving
        alpha = ((g1 * g2).sum() / (g2 * g2).sum()).item()
        alpha = 0.0 if np.isnan(alpha) else min(max(alpha, 0.0), 1.0)
        y = recon + alpha * (recon - previous_recon)

        # Calculate prediction: what the microscope would show for this guess
        Hu = fftconv(y, otf, shape)

        # Calculate KL divergences and stop iterations if they have increased
        kldim = kldiv(Hu, image)
        kld1 = kldiv(Hu, split1)
        kld2 = kldiv(Hu, split2)
        if n_steps is None:
            if play_it_safe:
                stop = kld1 > prev_kld1 or kld2 > prev_kld2
            else:
                stop = kld1 > prev_kld1 and kld2 > prev_kld2
            if stop:
                recon = previous_recon
                if verbose:
                    print(f"Optimum result obtained after {num_iters - 1} iterations "
                          f"with a total time of "
                          f"{timeit.default_timer() - start_time:.1f} seconds.")
                break
        prev_kld1 = kld1
        prev_kld2 = kld2

        # Biggs-Andrews acceleration
        previous_recon = recon
        recon = y

        # Calculate updates for split images and full images (H^T (d / Hu))
        HTratio1 = fftconv(split1 / (0.5 * (Hu + 1e-12)), otfT, shape)
        del split1
        HTratio2 = fftconv(split2 / (0.5 * (Hu + 1e-12)), otfT, shape)
        del split2
        # Average of the two (the Colab code adds them, which makes the result
        # twice as bright and stops the consensus test below from working).
        HTratio = 0.5 * (HTratio1 + HTratio2)
        del Hu

        # Calculate gradient consensus and blur with the interaction kernel, i.e. H^T H
        consensus_map = fftconv((HTratio1 - 1) * (HTratio2 - 1), otfotfT, shape)
        del HTratio1, HTratio2

        # Update our reconstruction only where the blurred consensus map says we
        # should (and don't allow negative brightness)
        recon = torch.where(consensus_map < 0, recon, recon * HTratio).clamp(min=0)
        del consensus_map

        # Biggs-Andrews acceleration
        g2 = g1
        g1 = recon - y
        del y

        min_HTratio = HTratio.min().item()
        max_HTratio = HTratio.max().item()
        max_relative_delta = ((recon - previous_recon) / recon.max()).max().item()
        del HTratio

        num_iters += 1
        if verbose:
            print(f"Iteration {num_iters:03d} completed in "
                  f"{timeit.default_timer() - iter_start_time:.3f} s. "
                  f"KLDs = {kldim:.4f} (image), {kld1:.4f} (split 1), {kld2:.4f} (split 2). "
                  f"Update range: {min_HTratio:.2f} to {max_HTratio:.2f}. "
                  f"Largest relative delta = {max_relative_delta:.5f}.")

        if n_steps is not None and num_iters >= n_steps:
            break

    return recon


def show_mips(image, recon):
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image.max(axis=0), cmap="gray")
    plt.axis("off")
    plt.title("Original image MIP")
    plt.subplot(1, 2, 2)
    plt.imshow(recon.max(axis=0), cmap="gray")
    plt.axis("off")
    plt.title("Deconvolved image MIP")
    plt.tight_layout()
    plt.show()
