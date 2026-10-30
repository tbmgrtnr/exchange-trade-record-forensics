#!/usr/bin/env python3
"""Placebo analysis: the suspect window vs. all authentic windows.

Locates the suspect window's Benford deviation within the distribution of
all authentic windows of the same length: For every rolling window of
``--window-minutes`` (advanced in one-minute steps) across the authentic
segments, it computes the mean absolute deviation (MAD) of the first-digit
distribution from Benford's Law, and compares the resulting distribution
with the suspect window's MAD.

Reported:

* per segment: number of windows, median/95th/99th percentile/maximum MAD,
  and how many windows reach the suspect window's MAD
* overall: the same across all authentic windows pooled, plus the count of
  non-overlapping (independent) windows, which yields a conservative
  placebo p-value
* optionally an N-matched variant (``--n-match``): each window is extended
  until it contains at least as many trades as the suspect window, which
  removes the small-sample noise advantage of thinner windows

Example (Binance case study):

    python placebo_windows.py \
        --suspect data/BTCUSDT-trades-2021-05.zip \
            2021-05-19T13:16 2021-05-19T13:56 \
        --segment "reference May 9-18" data/BTCUSDT-trades-2021-05.zip \
            2021-05-09 2021-05-19 \
        --segment "May 19 pre-window" data/BTCUSDT-trades-2021-05.zip \
            2021-05-19 2021-05-19T13:16 \
        --segment "May 19 post-window" data/BTCUSDT-trades-2021-05.zip \
            2021-05-19T13:56 2021-05-20 \
        --segment "May 20" data/BTCUSDT-trades-2021-05.zip \
            2021-05-20 2021-05-21 \
        --segment "April 18 crash" data/BTCUSDT-trades-2021-04-18.zip \
            2021-04-18 2021-04-19 \
        --n-match "reference May 9-18" --figdir figures
"""

import argparse
from collections import defaultdict

import numpy as np

from forensics import (BENFORD, MinuteGrid, WindowStats, fmt_ms, scan_file,
                       utc_ms)


def mad_of(fd: np.ndarray) -> float:
    return float(np.mean(np.abs(fd / fd.sum() - BENFORD)))


def rolling_mads(grid: MinuteGrid, wlen: int, min_trades: int) -> np.ndarray:
    """MAD of every rolling ``wlen``-minute window with enough trades."""
    cfd = np.vstack([np.zeros((1, 9), np.int64), np.cumsum(grid.fd, 0)])
    cn = np.concatenate([[0], np.cumsum(grid.n)])
    out = []
    for i in range(grid.n_minutes - wlen + 1):
        fd = cfd[i + wlen] - cfd[i]
        if cn[i + wlen] - cn[i] >= min_trades:
            out.append(mad_of(fd))
    return np.array(out)


def nonoverlapping_mads(grid: MinuteGrid, wlen: int,
                        min_trades: int) -> np.ndarray:
    out = []
    for i in range(0, grid.n_minutes - wlen + 1, wlen):
        n, fd = grid.window(i, i + wlen)
        if n >= min_trades:
            out.append(mad_of(fd))
    return np.array(out)


def n_matched_mads(grid: MinuteGrid, target_n: int) -> np.ndarray:
    """From each start minute, extend the window until it holds at least
    ``target_n`` trades, then compute its MAD."""
    cfd = np.vstack([np.zeros((1, 9), np.int64), np.cumsum(grid.fd, 0)])
    cn = np.concatenate([[0], np.cumsum(grid.n)])
    out = []
    for i in range(grid.n_minutes):
        j = int(np.searchsorted(cn, cn[i] + target_n))
        if j > grid.n_minutes:
            break
        out.append(mad_of(cfd[j] - cfd[i]))
    return np.array(out)


def save_figure(all_mads: np.ndarray, suspect_mad: float,
                figdir: str) -> None:
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(figdir, exist_ok=True)
    grey, red = (0.62, 0.65, 0.69), "#E31A1C"
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(all_mads, bins=60, color=grey, edgecolor="white")
    ax.axvline(suspect_mad, color=red, linestyle="--", linewidth=1.5)
    ax.annotate("suspect window", xy=(suspect_mad, 1),
                xycoords=("data", "axes fraction"), xytext=(-6, -12),
                textcoords="offset points", ha="right", color=red)
    ax.set_xlabel("First-digit MAD of rolling window")
    ax.set_ylabel("Number of windows")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "placebo_distribution.pdf"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suspect", nargs=3, required=True,
                    metavar=("FILE", "START", "END"))
    ap.add_argument("--segment", nargs=4, action="append", required=True,
                    metavar=("LABEL", "FILE", "START", "END"),
                    help="authentic segment (repeatable); segments must not "
                         "include the suspect window")
    ap.add_argument("--window-minutes", type=int, default=None,
                    help="rolling window length (default: the suspect "
                         "window's length)")
    ap.add_argument("--min-trades", type=int, default=1000,
                    help="minimum trades for a window to count (default "
                         "1000)")
    ap.add_argument("--n-match", default=None, metavar="LABEL",
                    help="also run the N-matched variant on this segment")
    ap.add_argument("--qty-col", type=int, default=2)
    ap.add_argument("--time-col", type=int, default=4)
    ap.add_argument("--time-unit", choices=["ms", "s", "us"], default="ms")
    ap.add_argument("--figdir", default=None)
    args = ap.parse_args()

    sus_file, sus_start, sus_end = args.suspect
    s0, s1 = utc_ms(sus_start), utc_ms(sus_end)
    wlen = args.window_minutes or max(1, round((s1 - s0) / 60_000))

    suspect = WindowStats("suspect", s0, s1)
    grids, labels = [], []
    by_file = defaultdict(lambda: ([], []))
    by_file[sus_file][0].append(suspect)
    for label, path, start, end in args.segment:
        g0, g1 = utc_ms(start), utc_ms(end)
        if max(g0, s0) < min(g1, s1):
            raise SystemExit(f"segment {label!r} overlaps the suspect "
                             "window; authentic segments must exclude it")
        grid = MinuteGrid(g0, round((g1 - g0) / 60_000))
        grids.append(grid)
        labels.append(label)
        by_file[path][1].append(grid)

    for path, (windows, file_grids) in by_file.items():
        print(f">> scanning {path}")
        scan_file(path, windows, grids=file_grids, qty_col=args.qty_col,
                  time_col=args.time_col, time_unit=args.time_unit)

    if suspect.n == 0:
        raise SystemExit("no trades found in the suspect window")
    sus_mad = suspect.mad
    print(f"\n===== Placebo analysis: rolling {wlen}-minute windows =====")
    print(f"Suspect window {fmt_ms(s0)} to {fmt_ms(s1)}: "
          f"n={suspect.n:,}  MAD={sus_mad:.4f}")

    hdr = (f"{'Segment':<28}{'windows':>9}{'median':>9}{'p95':>9}"
           f"{'p99':>9}{'max':>9}{'>= suspect':>11}")
    print("\n" + hdr)
    print("-" * len(hdr))
    pooled, pooled_nov = [], []
    for grid, label in zip(grids, labels):
        mads = rolling_mads(grid, wlen, args.min_trades)
        if not len(mads):
            print(f"{label:<28}{'0':>9}")
            continue
        pooled.append(mads)
        pooled_nov.append(nonoverlapping_mads(grid, wlen, args.min_trades))
        print(f"{label:<28}{len(mads):>9,}{np.median(mads):>9.4f}"
              f"{np.quantile(mads, .95):>9.4f}{np.quantile(mads, .99):>9.4f}"
              f"{mads.max():>9.4f}{(mads >= sus_mad).sum():>11}")

    allm = np.concatenate(pooled)
    nov = np.concatenate(pooled_nov)
    n_ge = int((nov >= sus_mad).sum())
    print(f"\nAll authentic windows: {len(allm):,}   max={allm.max():.4f}   "
          f"windows >= suspect MAD: {(allm >= sus_mad).sum()}")
    print(f"Non-overlapping (independent) windows: {len(nov)}   "
          f"max={nov.max():.4f}   >= suspect MAD: {n_ge}")
    print(f"Implied placebo p-value: "
          f"{'< ' if n_ge == 0 else ''}{max(n_ge, 1) / len(nov):.4f}")

    if args.n_match:
        i = labels.index(args.n_match)
        nm = n_matched_mads(grids[i], suspect.n)
        print(f"\nN-matched windows (>= {suspect.n:,} trades, segment "
              f"{args.n_match!r}): {len(nm):,}   median={np.median(nm):.4f}"
              f"   max={nm.max():.4f}   >= suspect MAD: "
              f"{(nm >= sus_mad).sum()}")

    if args.figdir:
        save_figure(allm, sus_mad, args.figdir)


if __name__ == "__main__":
    main()
