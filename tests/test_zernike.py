"""Tests for the hand-rolled Zernike polynomials in pupil_psf.py."""
import numpy as np
import pytest

import pupil_psf as pp


# A fine polar grid on the unit disk, reused by several tests.
@pytest.fixture(scope="module")
def disk():
    N = 400
    f = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(f, f, indexing="ij")
    rho = np.hypot(X, Y)
    phi = np.arctan2(Y, X)
    mask = (rho <= 1.0).astype(float)
    dA = (f[1] - f[0]) ** 2
    area = mask.sum() * dA
    return rho, phi, mask, dA, area


@pytest.mark.parametrize("j, nm", [
    (1, (0, 0)),    # piston
    (2, (1, 1)),    # tip
    (3, (1, -1)),   # tilt
    (4, (2, 0)),    # defocus
    (5, (2, -2)),   # oblique astigmatism
    (6, (2, 2)),    # vertical astigmatism
    (7, (3, -1)),   # vertical coma
    (8, (3, 1)),    # horizontal coma
    (11, (4, 0)),   # primary spherical
    (22, (6, 0)),   # secondary spherical
])
def test_noll_mapping(j, nm):
    assert pp.noll_to_nm(j) == nm


@pytest.mark.parametrize("j", [5, 6, 7, 8, 9, 10, 11, 12, 13, 22])
def test_rms_normalized_to_one(j, disk):
    rho, phi, mask, dA, area = disk
    z = pp.zernike(j, rho, phi, mask)
    rms = np.sqrt((z ** 2 * mask).sum() * dA / area)
    assert abs(rms - 1.0) < 0.02


def test_spherical_closed_form(disk):
    rho, phi, mask, _, _ = disk
    z11 = pp.zernike(11, rho, phi, mask)
    expected = np.sqrt(5) * (6 * rho ** 4 - 6 * rho ** 2 + 1) * mask
    assert np.abs(z11 - expected).max() < 1e-9


def test_orthogonality(disk):
    rho, phi, mask, dA, area = disk
    z11 = pp.zernike(11, rho, phi, mask)
    z6 = pp.zernike(6, rho, phi, mask)
    inner = (z11 * z6 * mask).sum() * dA / area
    assert abs(inner) < 0.02
