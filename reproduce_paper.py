#!/usr/bin/env python3
"""Verify the published test statistics of the paper's forensic sequence.

Runs the complete test battery on Binance's official public trade record
and checks each computed value against the corresponding number reported
in the paper (Tables 3 and 4, the placebo analysis, and the in-text MAD
anchors). Exits non-zero if any value does not reproduce.

Data (download with ``python download_data.py``):

* ``data/BTCUSDT-trades-2021-05.zip``    -- May 2021 futures trades
* ``data/BTCUSDT-trades-2021-04-18.zip`` -- April 18, 2021 futures trades

Window conventions as in the paper: The back-filled interval is
13:16-13:56 UTC on May 19, 2021, with minute 13:56 included in the Benford
analysis of Table 3 (942,924 trades) and the window taken as the plain
40-minute interval [13:16, 13:56) in the trade-size cross-check of Table 4
and the placebo analysis (936,525 trades). The reference period is
May 9-18, the control crash April 18.

Usage:  python reproduce_paper.py [--may PATH] [--april PATH]
                                  [--figdir figures]
"""

import argparse
import sys

import numpy as np

from benchmark_comparison import report as benchmark_report
from digit_tests import run_digit_tests
from forensics import (MinuteGrid, WindowStats, chi2_test, euclid_test,
                       ks_test, scan_file, utc_ms)
from placebo_windows import (n_matched_mads, nonoverlapping_mads,
                             rolling_mads, save_figure as placebo_figure)

checks = []


# Tolerance policy: each published value is matched up to one unit of its
# last reported decimal place (counts must match exactly), so a check can
# only pass if the computed value rounds to the published one up to the
# rounding convention of the table.
def check(label: str, published, computed, atol) -> None:
    ok = abs(published - computed) <= atol
    checks.append(ok)
    mark = "ok " if ok else "DIFF"
    print(f"  [{mark}] {label:<46} paper: {published:>12,.4f}   "
          f"computed: {computed:>12,.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--may", default="data/BTCUSDT-trades-2021-05.zip")
    ap.add_argument("--april", default="data/BTCUSDT-trades-2021-04-18.zip")
    ap.add_argument("--figdir", default="figures")
    args = ap.parse_args()

    # ---- pass 1: May archive (stops after May 20, the last day needed) ----
    w41 = WindowStats("back-filled window (Table 3)",
                      utc_ms("2021-05-19T13:16"), utc_ms("2021-05-19T13:57"),
                      keep_values=True)
    w40 = WindowStats("back-filled window (Table 4)",
                      utc_ms("2021-05-19T13:16"), utc_ms("2021-05-19T13:56"),
                      keep_values=True)
    ref = WindowStats("reference May 9-18",
                      utc_ms("2021-05-09"), utc_ms("2021-05-19"),
                      sample_every=20)
    grid_ref = MinuteGrid(utc_ms("2021-05-09"), 10 * 1440)
    grid_pre = MinuteGrid(utc_ms("2021-05-19"), 13 * 60 + 16)
    grid_post = MinuteGrid(utc_ms("2021-05-19T13:56"), 10 * 60 + 4)
    grid_may20 = MinuteGrid(utc_ms("2021-05-20"), 1440)

    print(f">> pass 1/2: streaming {args.may} (about 50 M rows)")
    scan_file(args.may, [w41, w40, ref],
              grids=[grid_ref, grid_pre, grid_post, grid_may20],
              stop_ms=utc_ms("2021-05-21"))

    # ---- pass 2: April 18 control day ----
    apr = WindowStats("April 18 control crash",
                      utc_ms("2021-04-18"), utc_ms("2021-04-19"),
                      keep_values=True)
    grid_apr = MinuteGrid(utc_ms("2021-04-18"), 1440)
    print(f">> pass 2/2: streaming {args.april}")
    scan_file(args.april, [apr], grids=[grid_apr])

    # ---- full reports (also writes the figures) ----
    run_digit_tests(w41, figdir=args.figdir)
    benchmark_report(w41, ref.fd, "May 9-18", figdir=args.figdir)

    # ---- checks vs. the published numbers ----
    print("\n===== Comparison against the published values =====")

    print("\nTable 3, Panel A (relative frequencies, %):")
    pub_freq = [53.5, 16.9, 11.6, 4.2, 5.5, 1.0, 3.3, 3.3, 0.5]
    obs = 100 * w41.rel_freq
    for d in range(9):
        check(f"digit {d + 1} frequency", pub_freq[d], obs[d], 0.1)

    print("\nTable 3, Panel B (mantissa descriptives, MAD, distortion):")
    mm = w41.mantissa_moments()
    check("observations", 942_924, w41.n, 0)
    check("mantissa mean", 0.250, mm["mean"], 0.001)
    check("mantissa variance", 0.088, mm["var"], 0.001)
    check("mantissa ex. kurtosis", -0.609, mm["ex_kurtosis"], 0.001)
    check("mantissa skewness", 0.817, mm["skewness"], 0.001)
    check("mean absolute deviation (MAD)", 0.052, w41.mad, 0.001)
    check("distortion factor", -54.294, w41.distortion_factor(), 0.001)

    print("\nTable 3, Panel C (goodness-of-fit statistics):")
    chi2, _ = chi2_test(w41.fd)
    ks, _ = ks_test(w41.fd)
    eu, _ = euclid_test(w41.fd)
    arc, _ = w41.arc_test()
    check("Pearson chi-square", 304_183, chi2, 1.0)
    check("Kolmogorov-Smirnov", 227.38, ks, 0.01)
    check("Euclidean distance", 246.28, eu, 0.01)
    check("mantissa arc test", 0.13, arc, 0.01)

    print("\nIn-text MAD anchors (one consistent public source):")
    check("back-filled window MAD", 0.052, w40.mad, 0.001)
    check("reference period MAD", 0.026, ref.mad, 0.001)
    check("April 18 MAD", 0.027, apr.mad, 0.001)

    print("\nTable 4 (trade-size cross-check):")
    from scipy import stats as sps
    check("n back-filled", 936_525, w40.n, 0)
    check("n reference", 49_955_413, ref.n, 0)
    check("n April 18", 6_123_918, apr.n, 0)
    check("mean size back-filled", 0.027, w40.mean_size, 0.001)
    check("mean size reference", 0.096, ref.mean_size, 0.001)
    check("mean size April 18", 0.085, apr.mean_size, 0.001)
    check("share at minimum back-filled (%)", 47.3,
          100 * w40.share_min, 0.1)
    check("share at minimum reference (%)", 19.3,
          100 * ref.share_min, 0.1)
    check("share at minimum April 18 (%)", 19.7,
          100 * apr.share_min, 0.1)
    check("share >= 1 BTC back-filled (%)", 0.46,
          100 * w40.share_large, 0.01)
    check("share >= 1 BTC reference (%)", 1.25,
          100 * ref.share_large, 0.01)
    check("share >= 1 BTC April 18 (%)", 1.32,
          100 * apr.share_large, 0.01)
    check("99th percentile back-filled", 0.50,
          float(np.quantile(w40.values, .99)), 0.01)
    check("99th percentile reference", 1.12,
          float(np.quantile(ref.samples, .99)), 0.01)
    check("99th percentile April 18", 1.19,
          float(np.quantile(apr.values, .99)), 0.01)
    d_ref, _ = sps.ks_2samp(np.log(w40.values), np.log(ref.samples))
    d_apr, _ = sps.ks_2samp(np.log(w40.values), np.log(apr.values))
    check("KS D vs. reference", 0.47, d_ref, 0.01)
    check("KS D vs. April 18", 0.46, d_apr, 0.01)

    print("\nPlacebo analysis (rolling 40-minute windows):")
    segments = [("reference May 9-18", grid_ref),
                ("May 20", grid_may20),
                ("April 18", grid_apr),
                ("May 19 pre-window", grid_pre),
                ("May 19 post-window", grid_post)]
    pooled = [rolling_mads(g, 40, 1000) for _, g in segments]
    allm = np.concatenate([m for m in pooled if len(m)])
    nov = np.concatenate(
        [nonoverlapping_mads(g, 40, 1000) for _, g in segments])
    nm = n_matched_mads(grid_ref, w40.n)
    check("number of authentic windows", 18_485, len(allm), 0)
    check("maximum authentic MAD", 0.042, allm.max(), 0.001)
    check("non-overlapping windows", 466, len(nov), 0)
    check("windows >= back-filled MAD", 0, int((allm >= w40.mad).sum()), 0)
    check("N-matched maximum MAD", 0.037, nm.max(), 0.001)
    if args.figdir:
        placebo_figure(allm, w40.mad, args.figdir)

    # ---- summary ----
    n_fail = checks.count(False)
    print(f"\n===== {len(checks)} checks, {n_fail} failed =====")
    if args.figdir:
        print(f"Figures written to {args.figdir}/")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
