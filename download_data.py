#!/usr/bin/env python3
"""Download the two Binance trade archives used by the paper's analysis.

Fetches Binance's official public trade record (data.binance.vision) for
the May 2021 futures month and the April 18, 2021 control day into
``data/``, and verifies each file against Binance's published SHA-256
checksum. The archives are not redistributed with this repository; they
come directly from the exchange, which is exactly what makes the analysis
independently verifiable.

Sizes: about 1.6 GB (May, ca. 9 GB unzipped)
and about 60 MB (April 18). the tools read the ZIP directly
"""

import hashlib
import os
import sys
import urllib.request

BASE = "https://data.binance.vision/data/futures/um"
FILES = [
    f"{BASE}/monthly/trades/BTCUSDT/BTCUSDT-trades-2021-05.zip",
    f"{BASE}/daily/trades/BTCUSDT/BTCUSDT-trades-2021-04-18.zip",
]


def fetch(url: str, dest: str) -> None:
    print(f">> {url}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    os.makedirs("data", exist_ok=True)
    ok = True
    for url in FILES:
        dest = os.path.join("data", os.path.basename(url))
        if os.path.exists(dest):
            print(f">> {dest} already exists, skipping download")
        else:
            fetch(url, dest)
        checksum_file = dest + ".CHECKSUM"
        fetch(url + ".CHECKSUM", checksum_file)
        expected = open(checksum_file).read().split()[0]
        actual = sha256(dest)
        if actual == expected:
            print(f"   checksum OK ({actual[:16]}...)")
        else:
            print(f"   CHECKSUM MISMATCH for {dest}!\n"
                  f"   expected {expected}\n   got      {actual}")
            ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
