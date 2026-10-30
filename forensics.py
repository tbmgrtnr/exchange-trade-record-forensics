#!/usr/bin/env python3
"""Shared library for the exchange-data forensics toolkit.

Provides streaming access to exchange trade archives and the statistical
building blocks used by the command-line tools in this repository:

* ``TradeReader``       -- chunked, memory-bounded reading of (timestamp,
                           quantity) pairs from a CSV/ZIP trade archive
* ``WindowStats``       -- one-pass accumulator of all digit and trade-size
                           statistics for a time window
* ``MinuteGrid``        -- per-minute trade counts and first-digit counts
                           (input to the rolling-window placebo analysis)
* Benford test functions (chi-square, Kolmogorov-Smirnov, Euclidean
  distance, mantissa arc test, MAD, distortion factor, second-order test,
  summation test)

The default column layout is that of Binance's official futures trade
archives (https://data.binance.vision):

    trade_id, price, quantity, quote_quantity, timestamp_ms, is_buyer_maker

Only the quantity and timestamp columns are used. Both column positions and
the timestamp unit are parameters of ``TradeReader``, so records of other
exchanges can be analyzed by pointing the tools at the right columns.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

BENFORD = np.log10(1.0 + 1.0 / np.arange(1, 10))  # P(first digit = 1..9)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def utc_ms(spec: str) -> int:
    """Parse ``YYYY-MM-DD[THH:MM[:SS]]`` as UTC and return Unix milliseconds."""
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(spec, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue
    raise ValueError(f"unrecognized UTC time specification: {spec!r}")


def fmt_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M")


# ---------------------------------------------------------------------------
# Digit helpers
# ---------------------------------------------------------------------------

def first_digits(q: np.ndarray) -> np.ndarray:
    """First significant digit (1..9) of each positive value.

    The significand is rounded to nine decimal places before truncation so
    that binary floating-point noise (e.g. 0.3 / 0.1 = 2.9999...) cannot
    shift a value across a digit boundary. Trade sizes are decimal numbers
    with far fewer than nine significant digits, so the rounding is exact
    for real-world exchange data.
    """
    q = np.asarray(q, dtype=np.float64)
    expo = np.floor(np.log10(q))
    signif = np.round(q * np.power(10.0, -expo), 9)
    return np.floor(signif).astype(np.int64).clip(1, 9)


def digit_counts(q: np.ndarray) -> np.ndarray:
    """Counts of first significant digits 1..9 (length-9 vector)."""
    return np.bincount(first_digits(q), minlength=10)[1:10]


def mantissas(q: np.ndarray) -> np.ndarray:
    """Base-10 mantissa (fractional part of log10) of each positive value."""
    return np.mod(np.log10(np.asarray(q, dtype=np.float64)), 1.0)


# ---------------------------------------------------------------------------
# Streaming reader
# ---------------------------------------------------------------------------

class TradeReader:
    """Chunked reader yielding (timestamp_ms, quantity) numpy arrays.

    Reads plain CSV or single-file ZIP archives. A header row, if present,
    is detected and skipped automatically. Rows that do not parse are
    dropped silently (some archives contain occasional stray header lines).
    """

    def __init__(self, path: str, qty_col: int = 2, time_col: int = 4,
                 time_unit: str = "ms", chunksize: int = 5_000_000):
        self.path = path
        self.qty_col = qty_col
        self.time_col = time_col
        self.time_factor = {"ms": 1.0, "s": 1000.0, "us": 0.001}[time_unit]
        self.chunksize = chunksize

    def _has_header(self) -> bool:
        probe = pd.read_csv(self.path, nrows=1, header=None, dtype=str)
        try:
            float(probe.iloc[0, self.qty_col])
            return False
        except (ValueError, TypeError, IndexError):
            return True

    def stream(self, progress: bool = False):
        header = 0 if self._has_header() else None
        cols = sorted({self.qty_col, self.time_col})
        pos = {c: i for i, c in enumerate(cols)}
        rows_done = 0
        for chunk in pd.read_csv(self.path, header=header, usecols=cols,
                                 chunksize=self.chunksize, na_filter=False,
                                 dtype=str, engine="c",
                                 on_bad_lines="skip"):
            q = pd.to_numeric(chunk.iloc[:, pos[self.qty_col]],
                              errors="coerce").to_numpy(dtype=np.float64)
            t = pd.to_numeric(chunk.iloc[:, pos[self.time_col]],
                              errors="coerce").to_numpy(dtype=np.float64)
            ok = np.isfinite(q) & np.isfinite(t) & (q > 0)
            q, t = q[ok], t[ok]
            t = (t * self.time_factor).astype(np.int64)
            rows_done += len(chunk)
            if progress:
                print(f"   ... {rows_done // 1_000_000} M rows",
                      file=sys.stderr)
            yield t, q


# ---------------------------------------------------------------------------
# One-pass accumulators
# ---------------------------------------------------------------------------

@dataclass
class WindowStats:
    """Accumulates digit and trade-size statistics for one time window."""

    label: str
    start_ms: int
    end_ms: int
    large_size: float = 1.0        # threshold for the "large trade" share
    keep_values: bool = False      # keep all quantities (small windows only)
    sample_every: int = 0          # keep every k-th quantity (0 = none)

    n: int = 0
    qty_sum: float = 0.0
    fd: np.ndarray = field(default_factory=lambda: np.zeros(9, np.int64))
    digit_qty_sum: np.ndarray = field(
        default_factory=lambda: np.zeros(9, np.float64))
    mantissa_sums: np.ndarray = field(
        default_factory=lambda: np.zeros(4, np.float64))  # m, m^2, m^3, m^4
    arc_cos_sum: float = 0.0
    arc_sin_sum: float = 0.0
    df_sum: float = 0.0            # sum of collapsed values >= 10 (see below)
    df_n: int = 0
    min_size: float = np.inf
    min_count: int = 0
    large_count: int = 0
    _values: list = field(default_factory=list)
    _samples: list = field(default_factory=list)

    def update(self, t: np.ndarray, q: np.ndarray) -> None:
        mask = (t >= self.start_ms) & (t < self.end_ms)
        if not mask.any():
            return
        qs = q[mask]
        n_before = self.n
        self.n += len(qs)
        self.qty_sum += float(qs.sum())

        d = first_digits(qs)
        self.fd += np.bincount(d, minlength=10)[1:10]
        self.digit_qty_sum += np.bincount(d, weights=qs, minlength=10)[1:10]

        m = mantissas(qs)
        self.mantissa_sums += np.array(
            [m.sum(), (m ** 2).sum(), (m ** 3).sum(), (m ** 4).sum()])
        two_pi_m = 2.0 * np.pi * m
        self.arc_cos_sum += float(np.cos(two_pi_m).sum())
        self.arc_sin_sum += float(np.sin(two_pi_m).sum())
        big = m[qs >= 10.0]
        self.df_sum += float(np.power(10.0, 1.0 + big).sum())
        self.df_n += len(big)

        chunk_min = float(qs.min())
        if chunk_min < self.min_size:
            self.min_size = chunk_min
            self.min_count = int((qs == chunk_min).sum())
        elif chunk_min == self.min_size:
            self.min_count += int((qs == self.min_size).sum())
        self.large_count += int((qs >= self.large_size).sum())

        if self.keep_values:
            self._values.append(qs.copy())
        if self.sample_every:
            idx = np.arange(len(qs)) + n_before
            self._samples.append(qs[idx % self.sample_every == 0].copy())

    # -- derived quantities -------------------------------------------------

    @property
    def values(self) -> np.ndarray:
        return (np.concatenate(self._values) if self._values
                else np.empty(0))

    @property
    def samples(self) -> np.ndarray:
        return (np.concatenate(self._samples) if self._samples
                else np.empty(0))

    @property
    def rel_freq(self) -> np.ndarray:
        return self.fd / self.fd.sum()

    @property
    def mad(self) -> float:
        """Mean absolute deviation of first-digit frequencies from Benford."""
        return float(np.mean(np.abs(self.rel_freq - BENFORD)))

    @property
    def mean_size(self) -> float:
        return self.qty_sum / self.n

    @property
    def share_min(self) -> float:
        return self.min_count / self.n

    @property
    def share_large(self) -> float:
        return self.large_count / self.n

    def mantissa_moments(self) -> dict:
        """Mean, variance, excess kurtosis and skewness of the mantissas.

        Under Benford's Law the mantissa is uniform on [0, 1): mean 0.5,
        variance 1/12, skewness 0, excess kurtosis -1.2.
        """
        n = self.n
        m1, m2, m3, m4 = self.mantissa_sums / n
        mean = m1
        var = m2 - mean ** 2
        c3 = m3 - 3 * mean * m2 + 2 * mean ** 3
        c4 = m4 - 4 * mean * m3 + 6 * mean ** 2 * m2 - 3 * mean ** 4
        return {"mean": mean, "var": var,
                "skewness": c3 / var ** 1.5,
                "ex_kurtosis": c4 / var ** 2 - 3.0}

    def distortion_factor(self) -> float:
        """Distortion factor following Drake and Nigrini (2000), computed
        exactly as by the R package ``benford.analysis`` (the
        implementation behind the paper's Table 3): observations of at
        least 10 are collapsed into [10, 100) and their mean is compared
        with the expectation under Benford's Law, 90 / ln(10) ~= 39.09
        (per the formulation, values below 10 do not enter this
        statistic). Negative values indicate an excess of low
        significands, i.e. understated numbers. The package's exact
        constants are retained for comparability with the published
        value."""
        if self.df_n == 0:
            return float("nan")
        am = self.df_sum / self.df_n
        return 100.0 * (am - 39.0865) / 39.0685

    def arc_test(self) -> tuple:
        """Mantissa arc test: squared distance of the mantissa's center of
        mass on the unit circle from the origin, with asymptotic p-value
        exp(-n * L2). Under Benford's Law the center of mass is (0, 0)."""
        xbar = self.arc_cos_sum / self.n
        ybar = self.arc_sin_sum / self.n
        l2 = xbar ** 2 + ybar ** 2
        return l2, float(np.exp(-l2 * self.n))


class MinuteGrid:
    """Per-minute trade counts and first-digit counts over a fixed span.

    This is the only information the rolling-window placebo analysis needs:
    with per-minute first-digit counts, the Benford deviation of every
    rolling window is an exact aggregate, not an approximation.
    """

    def __init__(self, start_ms: int, n_minutes: int):
        self.start_ms = start_ms
        self.n_minutes = n_minutes
        self.n = np.zeros(n_minutes, np.int64)
        self.fd = np.zeros((n_minutes, 9), np.int64)

    def update(self, t: np.ndarray, q: np.ndarray) -> None:
        idx = (t - self.start_ms) // 60_000
        mask = (idx >= 0) & (idx < self.n_minutes)
        if not mask.any():
            return
        idx = idx[mask].astype(np.int64)
        d = first_digits(q[mask])
        np.add.at(self.n, idx, 1)
        np.add.at(self.fd, (idx, d - 1), 1)

    def window(self, i0: int, i1: int) -> tuple:
        """Total trade count and first-digit counts for minutes [i0, i1)."""
        return int(self.n[i0:i1].sum()), self.fd[i0:i1].sum(axis=0)


def scan_file(path: str, windows: list, grids: list = (),
              qty_col: int = 2, time_col: int = 4, time_unit: str = "ms",
              progress: bool = True, stop_ms: int = None) -> None:
    """Single pass over a trade archive, feeding all accumulators.

    ``stop_ms`` stops reading once a chunk lies entirely past that time
    (archives are approximately time-ordered), saving time when only an
    early part of a large file is needed.
    """
    reader = TradeReader(path, qty_col=qty_col, time_col=time_col,
                         time_unit=time_unit)
    for t, q in reader.stream(progress=progress):
        for w in windows:
            w.update(t, q)
        for g in grids:
            g.update(t, q)
        if stop_ms is not None and len(t) and t.min() > stop_ms:
            break


# ---------------------------------------------------------------------------
# Goodness-of-fit tests on first-digit counts
# ---------------------------------------------------------------------------

def chi2_test(fd: np.ndarray) -> tuple:
    """Pearson chi-square statistic and asymptotic p-value (8 df)."""
    from scipy import stats
    n = fd.sum()
    expected = n * BENFORD
    stat = float(((fd - expected) ** 2 / expected).sum())
    return stat, float(stats.chi2.sf(stat, 8))


def ks_test(fd: np.ndarray, n_sim: int = 10_000, seed: int = 42) -> tuple:
    """Kolmogorov-Smirnov statistic sqrt(n) * sup |ECDF - Benford CDF| on
    the discrete first-digit distribution, with a Monte-Carlo p-value
    (n_sim multinomial samples drawn under the Benford null)."""
    n = int(fd.sum())
    stat = _ks_stat(fd, n)
    sims = np.random.default_rng(seed).multinomial(n, BENFORD, size=n_sim)
    null = np.array([_ks_stat(s, n) for s in sims])
    return stat, float((null >= stat).mean())


def _ks_stat(fd: np.ndarray, n: int) -> float:
    diff = np.cumsum(fd / n) - np.cumsum(BENFORD)
    return float(np.sqrt(n) * np.max(np.abs(diff)))


def euclid_test(fd: np.ndarray, n_sim: int = 10_000, seed: int = 42) -> tuple:
    """Euclidean distance statistic sqrt(n) * ||observed - Benford||_2 on
    first-digit frequencies, with a Monte-Carlo p-value."""
    n = int(fd.sum())
    stat = _euclid_stat(fd, n)
    sims = np.random.default_rng(seed).multinomial(n, BENFORD, size=n_sim)
    null = np.array([_euclid_stat(s, n) for s in sims])
    return stat, float((null >= stat).mean())


def _euclid_stat(fd: np.ndarray, n: int) -> float:
    return float(np.sqrt(n) * np.sqrt(((fd / n - BENFORD) ** 2).sum()))


def per_digit_z(fd: np.ndarray) -> tuple:
    """Per-digit two-sided z-test of each observed frequency against its
    Benford expectation. Returns (z statistics, p-values), length 9."""
    from scipy import stats
    n = fd.sum()
    obs = fd / n
    se = np.sqrt(BENFORD * (1 - BENFORD) / n)
    z = (obs - BENFORD) / se
    return z, 2 * stats.norm.sf(np.abs(z))


# ---------------------------------------------------------------------------
# Second-order and summation tests (need the window's raw values)
# ---------------------------------------------------------------------------

def second_order_counts(values: np.ndarray) -> np.ndarray:
    """First-digit counts of the positive differences between consecutive
    ordered observations (Nigrini's second-order Benford test). Differences
    are rounded to 12 decimals to remove binary floating-point noise before
    the digit is read; zero differences carry no digit and are dropped."""
    diffs = np.diff(np.sort(np.asarray(values, dtype=np.float64)))
    diffs = np.round(diffs, 12)
    diffs = diffs[diffs > 0]
    return digit_counts(diffs)


def summation_frequencies(digit_qty_sum: np.ndarray) -> np.ndarray:
    """Share of the total summed quantity contributed by each first digit.
    Benford's summation theorem predicts equal shares (1/9) for authentic
    multi-magnitude data."""
    return digit_qty_sum / digit_qty_sum.sum()


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def print_digit_table(fd: np.ndarray, title: str) -> None:
    obs = fd / fd.sum()
    z, p = per_digit_z(fd)
    print(f"\n{title}")
    print(f"{'Digit':<22}" + "".join(f"{d:>8}" for d in range(1, 10)))
    print(f"{'Relative frequency':<22}" +
          "".join(f"{100 * o:>7.1f}%" for o in obs))
    print(f"{'Benford expectation':<22}" +
          "".join(f"{100 * b:>7.1f}%" for b in BENFORD))
    print(f"{'Difference (pp)':<22}" +
          "".join(f"{100 * (o - b):>+7.1f} " for o, b in zip(obs, BENFORD)))
    print(f"{'Significance':<22}" +
          "".join(f"{stars(pv):>8}" for pv in p))
