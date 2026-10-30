# Exchange Trade-Record Forensics

This repository provides the code implementing the tests of the forensic
sequence developed in:

> Baumgartner, T., & Guettler, A. (2026). Trust in a trustless network: The
> case of the Bitcoin flash crash on May 19, 2021. *International Review of
> Financial Analysis*, 105481. https://doi.org/10.1016/j.irfa.2026.105481

The sequence audits a suspect part of an exchange's self-reported trade
record. This is in the application of the paper: the 40-minute window of Bitcoin futures trades 
that Binance withheld during the flash crash of May 19, 2021 and back-filled
about a week later by testing it against the digit frequencies and
trade-size distributions that authentic data exhibit. Each step is
independent of Binance-specific features, so the tools can be applied
whenever an exchange's self-reported record is in doubt.

## The tests

| Tool | What it does | Paper output |
|---|---|---|
| `digit_tests.py` | Benford first-digit battery: digit distribution, mantissa descriptives, MAD, distortion factor, chi-square / Kolmogorov-Smirnov / Euclidean distance / mantissa arc tests, second-order test, summation test | Table 3, Figure 3 |
| `benchmark_comparison.py` | Suspect window vs. an authentic reference period: difference-in-differences per digit with a seeded paired bootstrap; optional time-of-day matching | Figure 4 |
| `tradesize_crosscheck.py` | Non-digit cross-check: shape of the trade-size distribution (mean, share at minimum size, large-trade share, upper percentiles) vs. authentic benchmarks, with two-sample Kolmogorov-Smirnov tests | Table 4 |
| `placebo_windows.py` | Placebo distribution: the Benford deviation of every rolling authentic window of the same length, vs. the suspect window | Figure 5 |
| `reproduce_paper.py` | Runs all of the above on Binance's official public record and checks every computed value against the numbers reported in the paper | Tables 3-4, Figures 3-5 |

All tools stream the trade archives chunk-wise, so the 9 GB May 2021
record is processed on an ordinary laptop (under 2 GB of RAM; the full
reproduction takes a few minutes).

## Data

The analysis runs on the exchange's own
official public trade record, which is exactly what makes it
independently verifiable:

```
python download_data.py
```

downloads the two archives from https://data.binance.vision into `data/`
and verifies Binance's published SHA-256 checksums:

* `BTCUSDT-trades-2021-05.zip` (~1.6 GB) — USDT-margined BTCUSDT futures
  trades, May 2021 (suspect window, reference period, placebo segments)
* `BTCUSDT-trades-2021-04-18.zip` (~60 MB) — April 18, 2021, the
  control crash

## Verifying the published values

```
pip install -r requirements.txt
python download_data.py
python reproduce_paper.py --figdir figures
```

`reproduce_paper.py` recomputes the test statistics of Table 3
(digit frequencies, mantissa descriptives, MAD 0.052, distortion factor
-54.294, chi-square 304,183, Kolmogorov-Smirnov 227.38, Euclidean
distance 246.28, mantissa arc 0.13), the in-text MAD anchors
(0.052 / 0.026 / 0.027), Table 4 (all rows, KS D = 0.47 / 0.46), and the
placebo analysis (18,485 authentic windows, maximum MAD 0.042, 466
independent windows, N-matched maximum 0.037) — and prints a line-by-line
comparison against the published values. It exits non-zero if any value
does not reproduce.

Window conventions from the paper: The back-filled interval is
13:16-13:56 UTC on May 19, 2021. Minute 13:56 is included in the Benford
analysis of Table 3 (942,924 trades). The trade-size cross-check of
Table 4 and the placebo analysis use the plain 40-minute interval
[13:16, 13:56) (936,525 trades). The reference period is May 9-18, the
control crash April 18, 2021.

Randomized components (the bootstrap and the Monte-Carlo p-values) use
fixed seeds, such that repeated runs give identical output.

## Applying the sequence to another exchange

1. **Isolate the suspect data**: the window whose authenticity is in
   doubt (an outage, a back-fill, a disputed feed).
2. **Digit tests** (`digit_tests.py`): test the suspect window's trade
   sizes against Benford's Law.
3. **Benchmark and cross-validate** (`benchmark_comparison.py`,
   `tradesize_crosscheck.py`): compare against an authentic reference
   period from the same market and, if available, a control event of
   comparable stress but with an undisputed record.
4. **Placebo** (`placebo_windows.py`): locate the suspect window in the
   full distribution of authentic same-length windows.

The tools read any CSV trade record; set `--qty-col`, `--time-col` and
`--time-unit` to match the exchange's format.

**Applicability caveats.** Benford-based tests presuppose that the
analyzed quantity spans several orders of magnitude and arises from a
multiplicative mixture. Trade sizes generally qualify, prices generally
do not (a price pinned near one level has a degenerate first digit).
Authentic data can deviate mildly from Benford's Law, i.e., no single
statistic should be read as proof. The sequence's conclusion rests on
the joint evidence of the digit tests, the non-digit trade-size
cross-check, the control event, and the placebo distribution. See the
paper for the underlying conditions and their discussion.

## Citation

If you use this code, please cite the article:

Baumgartner, T., & Guettler, A. (2026). Trust in a trustless network: The
case of the Bitcoin flash crash on May 19, 2021. *International Review of
Financial Analysis*, 105481. https://doi.org/10.1016/j.irfa.2026.105481

```bibtex
@article{BaumgartnerGuettler2026,
  author  = {Baumgartner, Tim and Guettler, Andre},
  title   = {Trust in a Trustless Network: The Case of the {B}itcoin Flash Crash on {M}ay 19, 2021},
  journal = {International Review of Financial Analysis},
  year    = {2026},
  doi     = {10.1016/j.irfa.2026.105481}
}
```

## License

MIT (see `LICENSE`).