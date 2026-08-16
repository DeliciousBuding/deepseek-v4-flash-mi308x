#!/usr/bin/env python3
"""Validate the checked-in MI308X AITER tuning tables without importing AITER."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = {
    ROOT / "tuning/dsv4-mi308x-80cu-a8w8-blockscale-bpreshuffle.csv": 13,
    ROOT / "tuning/dsv4-mi308x-80cu-a8w8-blockscale.csv": 24,
}
REQUIRED = {
    "gfx", "cu_num", "M", "N", "K", "libtype", "kernelId", "splitK",
    "us", "kernelName", "tflops", "bw", "errRatio",
}


def validate(path: Path, expected_rows: int) -> None:
    if not path.is_file():
        raise SystemExit(f"missing tuning table: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or set(reader.fieldnames) != REQUIRED:
            raise SystemExit(
                f"unexpected columns in {path.name}: {reader.fieldnames!r}"
            )
        rows = list(reader)

    if len(rows) != expected_rows:
        raise SystemExit(
            f"unexpected row count in {path.name}: {len(rows)} != {expected_rows}"
        )

    seen: set[tuple[str, ...]] = set()
    covered_m: set[int] = set()
    for lineno, row in enumerate(rows, 2):
        if row["gfx"] != "gfx942" or row["cu_num"] != "80":
            raise SystemExit(
                f"{path.name}:{lineno}: expected gfx942/80, got "
                f"{row['gfx']}/{row['cu_num']}"
            )
        if row["libtype"] != "ck":
            raise SystemExit(f"{path.name}:{lineno}: unexpected libtype={row['libtype']}")
        try:
            m, n, k = int(row["M"]), int(row["N"]), int(row["K"])
            float(row["us"])
            err = float(row["errRatio"])
        except ValueError as exc:
            raise SystemExit(f"{path.name}:{lineno}: malformed numeric field: {exc}") from exc
        if err != 0.0:
            raise SystemExit(f"{path.name}:{lineno}: errRatio must be 0.0, got {err}")
        key = (row["gfx"], row["cu_num"], str(m), str(n), str(k))
        if key in seen:
            raise SystemExit(f"{path.name}:{lineno}: duplicate tuning key {key}")
        seen.add(key)
        covered_m.add(m)

    print(
        f"OK {path.name}: rows={len(rows)} gfx=gfx942 cu=80 "
        f"M={sorted(covered_m)}"
    )


def main() -> int:
    for path, expected_rows in TABLES.items():
        validate(path, expected_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
