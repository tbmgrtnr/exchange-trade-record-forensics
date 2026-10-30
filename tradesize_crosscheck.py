#!/usr/bin/env python3
"""Trade-size cross-check: a non-digit test of a suspect trade window.

Complements the Benford digit tests with a method that uses no digit
information: it compares the shape of the trade-size distribution in the
suspect window against one or more authentic benchmarks (a reference
period and, ideally, a control event of comparable market stress).

For each sample it reports the number of trades, the mean trade size,
the share of trades at the minimum size, the share of large trades, and
the 99th-percentile size, and it tests each benchmark's log trade-size
distribution against the suspect window with a two-sample
Kolmogorov-Smirnov test.

If a benchmark is very large, its quantiles and KS test are computed on
an every-k-th-trade systematic sample (``--sample-every``, default 20)
to bound memory; counts, means and shares are always exact.

Example (Binance case study):

    python tradesize_crosscheck.py \
        --suspect data/BTCUSDT-trades-2021-05.zip \
            2021-05-19T13:16 2021-05-19T13:56 \
        --benchmark reference data/BTCUSDT-trades-2021-05.zip \
            2021-05-09 2021-05-19 \
        --benchmark "april 18 crash" data/BTCUSDT-trades-2021-04-18.zip \
            2021-04-18 2021-04-19
"""

import argparse
from collections import defaultdict

import numpy as np

from forensics import WindowStats, fmt_ms, scan_file, stars, utc_ms


def quantile(stats: WindowStats, q: float) -> float:
    vals = stats.values if stats.keep_values else stats.samples
    return float(np.quantile(vals, q))


def report(suspect: WindowStats, benchmarks: list) -> None:
    from scipy import stats as sps

    cols = [suspect] + benchmarks
    print("\n===== Trade-size cross-check =====")
    for w in cols:
        print(f"{w.label:<24} {fmt_ms(w.start_ms)} to {fmt_ms(w.end_ms)}")

    hdr = f"{'Measure':<34}" + "".join(f"{w.label[:16]:>18}" for w in cols)
    print("\n" + hdr)
    print("-" * len(hdr))
    rows = [
        ("Number of trades", lambda w: f"{w.n:,}"),
        ("Mean trade size", lambda w: f"{w.mean_size:.3f}"),
        ("Share at minimum size (%)", lambda w: f"{100 * w.share_min:.1f}"),
        ("Minimum size", lambda w: f"{w.min_size:g}"),
        ("Share >= large size (%)", lambda w: f"{100 * w.share_large:.2f}"),
        ("99th percentile", lambda w: f"{quantile(w, 0.99):.2f}"),
    ]
    for label, f in rows:
        print(f"{label:<34}" + "".join(f"{f(w):>18}" for w in cols))

    print("\n--- Two-sample Kolmogorov-Smirnov on log(size) vs. suspect ---")
    log_sus = np.log(suspect.values)
    for b in benchmarks:
        vals = b.values if b.keep_values else b.samples
        d, p = sps.ks_2samp(log_sus, np.log(vals))
        note = "" if b.keep_values else f" (systematic sample, n={len(vals):,})"
        print(f"  vs. {b.label:<22} D={d:.2f} {stars(p)}{note}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suspect", nargs=3, required=True,
                    metavar=("FILE", "START", "END"),
                    help="suspect window: archive path and UTC start/end")
    ap.add_argument("--benchmark", nargs=4, action="append", required=True,
                    metavar=("LABEL", "FILE", "START", "END"),
                    help="authentic benchmark window (repeatable)")
    ap.add_argument("--large-size", type=float, default=1.0,
                    help="threshold for the large-trade share (default 1.0)")
    ap.add_argument("--sample-every", type=int, default=20,
                    help="systematic sampling interval for benchmark "
                         "quantiles/KS (default 20)")
    ap.add_argument("--qty-col", type=int, default=2)
    ap.add_argument("--time-col", type=int, default=4)
    ap.add_argument("--time-unit", choices=["ms", "s", "us"], default="ms")
    args = ap.parse_args()

    sus_file, sus_start, sus_end = args.suspect
    suspect = WindowStats("suspect", utc_ms(sus_start), utc_ms(sus_end),
                          large_size=args.large_size, keep_values=True)

    by_file = defaultdict(list)
    by_file[sus_file].append(suspect)
    benchmarks = []
    for label, path, start, end in args.benchmark:
        w = WindowStats(label, utc_ms(start), utc_ms(end),
                        large_size=args.large_size,
                        sample_every=args.sample_every)
        benchmarks.append(w)
        by_file[path].append(w)

    for path, windows in by_file.items():
        print(f">> scanning {path}")
        scan_file(path, windows, qty_col=args.qty_col,
                  time_col=args.time_col, time_unit=args.time_unit)

    for w in [suspect] + benchmarks:
        if w.n == 0:
            raise SystemExit(f"no trades found for {w.label!r}")
    report(suspect, benchmarks)


if __name__ == "__main__":
    main()
