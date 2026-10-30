#!/usr/bin/env python3
"""Benchmark comparison: suspect window vs. an authentic reference period.

Compares the first-digit distribution of a suspect window against a
reference period from the same trade record and reports:

* both windows' digit distributions and MADs
* the difference-in-differences per digit: (suspect deviation from
  Benford) minus (reference deviation from Benford)
* a paired bootstrap of the difference-in-differences with confidence
  intervals (seeded, so runs are reproducible)

Optionally the reference can be restricted to the suspect window's
time of day on each reference day (``--time-match``), which rules out
time-of-day effects, or to a single day (``--day``).

Example (Binance case study: back-filled window vs. May 9-18):

    python benchmark_comparison.py data/BTCUSDT-trades-2021-05.zip \
        --suspect-start 2021-05-19T13:16 --suspect-end 2021-05-19T13:57 \
        --ref-start 2021-05-09 --ref-end 2021-05-19 --figdir figures
"""

import argparse

import numpy as np

from forensics import (BENFORD, WindowStats, fmt_ms, print_digit_table,
                       scan_file, utc_ms)

DAY_MS = 86_400_000


def bootstrap_dd(fd_suspect: np.ndarray, fd_ref: np.ndarray,
                 sample_n: int = 10_000, n_boot: int = 10_000,
                 seed: int = 42) -> np.ndarray:
    """Paired bootstrap of the per-digit difference-in-differences.

    Draws one sample of ``sample_n`` first digits from each empirical digit
    distribution, then resamples both jointly ``n_boot`` times and returns
    the (n_boot, 9) array of per-digit frequency differences.
    """
    rng = np.random.default_rng(seed)
    digits = np.arange(1, 10)
    sus = rng.choice(digits, size=sample_n, p=fd_suspect / fd_suspect.sum())
    ref = rng.choice(digits, size=sample_n, p=fd_ref / fd_ref.sum())
    out = np.empty((n_boot, 9))
    for b in range(n_boot):
        idx = rng.integers(0, sample_n, sample_n)
        f_sus = np.bincount(sus[idx], minlength=10)[1:10] / sample_n
        f_ref = np.bincount(ref[idx], minlength=10)[1:10] / sample_n
        out[b] = f_sus - f_ref
    return out


def report(suspect: WindowStats, ref_fd: np.ndarray, ref_label: str,
           figdir: str = None, seed: int = 42) -> None:
    print(f"\n===== Benchmark comparison =====")
    print(f"Suspect window (UTC): {fmt_ms(suspect.start_ms)} to "
          f"{fmt_ms(suspect.end_ms)}   n={suspect.n:,}")
    print(f"Reference:            {ref_label}   n={ref_fd.sum():,}")

    print_digit_table(suspect.fd, "--- Suspect window ---")
    print_digit_table(ref_fd, f"--- Reference ({ref_label}) ---")

    obs_s = suspect.rel_freq
    obs_r = ref_fd / ref_fd.sum()
    mad_s = float(np.mean(np.abs(obs_s - BENFORD)))
    mad_r = float(np.mean(np.abs(obs_r - BENFORD)))
    print(f"\nMAD suspect window:   {mad_s:.4f}")
    print(f"MAD reference:        {mad_r:.4f}")

    dd = (obs_s - BENFORD) - (obs_r - BENFORD)
    boot = bootstrap_dd(suspect.fd, ref_fd, seed=seed)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    print("\n--- Difference-in-differences per digit "
          "(bootstrap 95% CI, seeded) ---")
    print(f"{'Digit':>6}{'DD (pp)':>10}{'CI low':>10}{'CI high':>10}"
          f"{'excl. 0':>9}")
    for i in range(9):
        print(f"{i + 1:>6}{100 * dd[i]:>+10.2f}{100 * lo[i]:>+10.2f}"
              f"{100 * hi[i]:>+10.2f}{'yes' if lo[i] * hi[i] > 0 else 'no':>9}")

    if figdir:
        save_figures(dd, boot, figdir)


def save_figures(dd: np.ndarray, boot: np.ndarray, figdir: str) -> None:
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(figdir, exist_ok=True)
    grey, dark = (0.62, 0.65, 0.69), "#2C3E50"
    digits = np.arange(1, 10)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(0, color=dark, linewidth=1)
    ax.bar(digits, 100 * dd, color=grey, edgecolor=dark)
    ax.set_xticks(digits)
    ax.set_xlabel("Digit")
    ax.set_ylabel("Difference (pp)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "dd.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(0, color=dark, linewidth=1)
    ax.boxplot([100 * boot[:, i] for i in range(9)],
               patch_artist=True,
               boxprops=dict(facecolor=grey, color=dark),
               medianprops=dict(color=dark), flierprops=dict(markersize=2))
    ax.set_xlabel("Digit")
    ax.set_ylabel("Bootstrapped difference (pp)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "dd_bootstrap.pdf"))
    plt.close(fig)


def time_matched_windows(ref_start: int, ref_end: int,
                         sus_start: int, sus_end: int) -> list:
    """One window per reference day covering the suspect window's clock
    time. The suspect window must lie within a single UTC day."""
    day_anchor = (sus_start // DAY_MS) * DAY_MS
    offs_start, offs_end = sus_start - day_anchor, sus_end - day_anchor
    first_day = (ref_start // DAY_MS) * DAY_MS
    spans = []
    day = first_day
    while day < ref_end:
        s, e = day + offs_start, day + offs_end
        if s >= ref_start and e <= ref_end:
            spans.append((s, e))
        day += DAY_MS
    return spans


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trades", help="trade archive (CSV or ZIP)")
    ap.add_argument("--suspect-start", required=True)
    ap.add_argument("--suspect-end", required=True)
    ap.add_argument("--ref-start", required=True)
    ap.add_argument("--ref-end", required=True)
    ap.add_argument("--time-match", action="store_true",
                    help="restrict the reference to the suspect window's "
                         "time of day on each reference day")
    ap.add_argument("--day", default=None,
                    help="restrict the reference to this single UTC day "
                         "(YYYY-MM-DD); combines with --time-match")
    ap.add_argument("--qty-col", type=int, default=2)
    ap.add_argument("--time-col", type=int, default=4)
    ap.add_argument("--time-unit", choices=["ms", "s", "us"], default="ms")
    ap.add_argument("--seed", type=int, default=42,
                    help="bootstrap random seed (default 42)")
    ap.add_argument("--figdir", default=None)
    args = ap.parse_args()

    sus_start, sus_end = utc_ms(args.suspect_start), utc_ms(args.suspect_end)
    ref_start, ref_end = utc_ms(args.ref_start), utc_ms(args.ref_end)
    if args.day:
        ref_start = utc_ms(args.day)
        ref_end = ref_start + DAY_MS
    if max(ref_start, sus_start) < min(ref_end, sus_end):
        raise SystemExit("the reference period overlaps the suspect window; "
                         "choose a reference that excludes it")

    suspect = WindowStats("suspect window", sus_start, sus_end)
    if args.time_match:
        spans = time_matched_windows(ref_start, ref_end, sus_start, sus_end)
        refs = [WindowStats(f"ref {fmt_ms(s)}", s, e) for s, e in spans]
        ref_label = (f"{fmt_ms(ref_start)} to {fmt_ms(ref_end)}, "
                     f"time-matched ({len(spans)} days)")
    else:
        refs = [WindowStats("reference", ref_start, ref_end)]
        ref_label = f"{fmt_ms(ref_start)} to {fmt_ms(ref_end)}"

    scan_file(args.trades, [suspect] + refs, qty_col=args.qty_col,
              time_col=args.time_col, time_unit=args.time_unit)
    if suspect.n == 0:
        raise SystemExit("no trades found in the suspect window")

    ref_fd = np.sum([r.fd for r in refs], axis=0)
    if ref_fd.sum() == 0:
        raise SystemExit("no trades found in the reference period")
    report(suspect, ref_fd, ref_label, figdir=args.figdir, seed=args.seed)


if __name__ == "__main__":
    main()
