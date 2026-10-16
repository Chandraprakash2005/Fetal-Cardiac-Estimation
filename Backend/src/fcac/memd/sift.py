

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline

# Prime bases for the Hammersley sequence's 2nd..nth dimensions.
_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47)


def _van_der_corput(n_points: int, base: int) -> np.ndarray:
    """Radical-inverse (van der Corput) sequence of length `n_points` in `base`."""
    seed = np.arange(1, n_points + 1, dtype=np.int64)
    seq = np.zeros(n_points, dtype=np.float64)
    denominator = 1.0 / base
    while np.any(seed > 0):
        seq += (seed % base) * denominator
        denominator /= base
        seed //= base
    return seq


def direction_vectors(n_directions: int, n_channels: int) -> np.ndarray:
    """`n_directions` quasi-uniform unit vectors on the (n_channels-1)-sphere.

    Uses a Hammersley sequence (first dimension = the regular grid (i-0.5)/N, remaining
    dimensions = van der Corput in successive primes) mapped through hyperspherical
    angles, as in the reference MEMD implementation. Low-discrepancy sampling matters
    here: with only ~64 directions, i.i.d. random vectors leave visible gaps on the
    sphere and the averaged envelope inherits that bias.

    Returns: (n_directions, n_channels), each row unit-norm.
    """
    if n_channels < 2:
        raise ValueError("MEMD needs at least 2 channels; use 1-D EMD otherwise")

    n_angles = n_channels - 1
    seq = np.empty((n_directions, n_angles), dtype=np.float64)
    seq[:, 0] = (np.arange(1, n_directions + 1) - 0.5) / n_directions
    for j in range(1, n_angles):
        seq[:, j] = _van_der_corput(n_directions, _PRIMES[j - 1])

    # Map the unit hypercube to hyperspherical angles: the first n_angles-1 angles span
    # [0, pi), the last spans [0, 2*pi).
    angles = np.empty_like(seq)
    angles[:, : n_angles - 1] = seq[:, : n_angles - 1] * np.pi
    angles[:, n_angles - 1] = seq[:, n_angles - 1] * 2.0 * np.pi

    vectors = np.ones((n_directions, n_channels), dtype=np.float64)
    sin_cumprod = np.ones(n_directions, dtype=np.float64)
    for k in range(n_angles):
        vectors[:, k] = sin_cumprod * np.cos(angles[:, k])
        sin_cumprod = sin_cumprod * np.sin(angles[:, k])
    vectors[:, n_channels - 1] = sin_cumprod

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def local_maxima(p: np.ndarray) -> np.ndarray:
    """Indices of strict interior local maxima of a 1-D projection.

    Plateaus are represented by their midpoint so that a flat-topped peak contributes a
    single extremum rather than none.
    """
    n = p.shape[0]
    if n < 3:
        return np.empty(0, dtype=np.int64)

    d = np.diff(p)
    nonzero = np.flatnonzero(d)
    if nonzero.size == 0:  # constant signal
        return np.empty(0, dtype=np.int64)

    # Collapse plateaus: carry the last non-zero slope forward across zero-diff runs.
    sign = np.zeros(d.size, dtype=np.int8)
    sign[nonzero] = np.sign(d[nonzero]).astype(np.int8)
    filled = np.maximum.accumulate(np.where(sign != 0, np.arange(sign.size), -1))
    rising = np.where(filled >= 0, sign[np.maximum(filled, 0)], 0)

    # A maximum is a rise followed by a fall; on a plateau spanning [i, j], `rising`
    # holds the pre-plateau slope, so the transition lands at the plateau's end.
    peak_ends = np.flatnonzero((rising[:-1] > 0) & (sign[1:] < 0)) + 1
    if peak_ends.size == 0:
        return np.empty(0, dtype=np.int64)

    # Walk each peak back over its plateau to recover the midpoint.
    peaks = []
    for end in peak_ends:
        start = int(end)
        while start > 0 and d[start - 1] == 0:
            start -= 1
        peaks.append((start + int(end)) // 2)
    return np.asarray(peaks, dtype=np.int64)


def _mirror_extend(idx: np.ndarray, n: int, n_mirror: int = 2) -> np.ndarray:
    """Reflect the outermost extrema about the signal's endpoints.

    Cubic splines are wildly unconstrained outside the first/last knot, so without
    boundary anchors the envelope diverges at both ends and the sifting residue picks up
    a spurious low-frequency ramp. Mirroring a couple of extrema is the standard EMD
    boundary treatment.
    """
    if idx.size < 2:
        return idx
    left = -idx[1 : 1 + n_mirror][::-1]
    right = 2 * (n - 1) - idx[max(0, idx.size - 1 - n_mirror) : idx.size - 1][::-1]
    return np.concatenate([left, idx, right])


def _direction_envelope(
    x: np.ndarray, projection: np.ndarray, t: np.ndarray
) -> Optional[np.ndarray]:
    """Envelope of the multivariate signal at one projection's maxima.

    x          : (C, N) signal
    projection : (N,) signal projected onto one direction
    t          : (N,) sample-index grid, precomputed by the caller
    Returns    : (C, N) envelope, or None if this direction has too few extrema to
                 interpolate (fewer than 3) -- the direction abstains from the mean
                 rather than contributing a degenerate spline.
    """
    idx = local_maxima(projection)
    if idx.size < 3:
        return None

    knots = _mirror_extend(idx, x.shape[1])
    # Mirrored knots index outside [0, N): clip when *sampling* x, but keep the mirrored
    # abscissae so the spline's boundary slope stays physical.
    values = x[:, np.clip(knots, 0, x.shape[1] - 1)]  # (C, n_knots)

    unique_knots, unique_pos = np.unique(knots, return_index=True)
    if unique_knots.size < 3:
        return None

    spline = CubicSpline(unique_knots, values[:, unique_pos], axis=1, extrapolate=True)
    return spline(t)


def _mean_envelope(
    x: np.ndarray, directions: np.ndarray, t: np.ndarray
) -> Tuple[Optional[np.ndarray], int]:
    """Multivariate local mean: envelopes averaged over all directions.

    Returns (mean (C, N) or None, number of directions that abstained).
    """
    total = np.zeros_like(x)
    n_used = 0
    n_abstained = 0
    for direction in directions:
        envelope = _direction_envelope(x, direction @ x, t)
        if envelope is None:
            n_abstained += 1
            continue
        total += envelope
        n_used += 1
    if n_used == 0:
        return None, n_abstained
    return total / n_used, n_abstained


def _sift_stop(previous: np.ndarray, current: np.ndarray, eps: float = 1e-12) -> float:
    """Huang's normalised standard-deviation criterion, summed over channels."""
    numerator = np.sum((previous - current) ** 2)
    denominator = np.sum(previous**2)
    return float(numerator / (denominator + eps))


def memd(
    x: np.ndarray,
    n_directions: int = 64,
    stopping_tolerance: float = 0.075,
    max_sift_iterations: int = 300,
    max_imfs: Optional[int] = None,
    abstain_fraction: float = 0.5,
) -> np.ndarray:
    """Decompose a multivariate signal into multivariate IMFs.

    x                   : (C, N) float array -- one preprocessed window, C=4 channels.
    n_directions        : projection directions on the (C-1)-sphere (cfg.memd.directions).
    stopping_tolerance  : Huang SD threshold ending the sift for one IMF.
    max_sift_iterations : hard cap on sifting per IMF.
    max_imfs            : optional cap; None (the default) lets K be data-determined,
                          which CLAUDE.md section 2 requires.
    abstain_fraction    : stop extracting IMFs once this fraction of directions finds
                          fewer than 3 extrema -- i.e. the residue is a trend, not an
                          oscillation.

    Returns: (K, C, N) float32. The last row is the residue, so `imfs.sum(axis=0)`
             reconstructs `x` to within float error.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected (C, N) signal, got shape {x.shape}")

    n_channels, n_samples = x.shape
    directions = direction_vectors(n_directions, n_channels)
    t = np.arange(n_samples, dtype=np.float64)

    imfs: List[np.ndarray] = []
    residue = x.copy()

    while max_imfs is None or len(imfs) < max_imfs:
        h = residue.copy()
        extracted = None

        for _ in range(max_sift_iterations):
            mean, n_abstained = _mean_envelope(h, directions, t)
            if mean is None or n_abstained > abstain_fraction * n_directions:
                # Not enough oscillation left anywhere to define an envelope.
                break
            h_next = h - mean
            sd = _sift_stop(h, h_next)
            h = h_next
            extracted = h
            if sd < stopping_tolerance:
                break

        if extracted is None:
            break

        imfs.append(extracted)
        residue = residue - extracted

        # Would the *residue* still support a sift? If not, we are done.
        _, n_abstained = _mean_envelope(residue, directions, t)
        if n_abstained > abstain_fraction * n_directions:
            break

    imfs.append(residue)  # final trend/residue row
    return np.stack(imfs, axis=0).astype(np.float32)
