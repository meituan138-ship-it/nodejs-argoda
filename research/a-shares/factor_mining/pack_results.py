"""Pack the mining results into a small zip to send back (no data, no parquet).

    python pack_results.py          # -> mining_results_<date>.zip (usually < 1 MB)

Contains factor code, the ledger, accepted formulas, NOTES.md and any final_check
output. Everything else (cache/, results/accepted/*.parquet) is rebuilt from the
code on the reviewer's side, so it never needs uploading.
"""
from __future__ import annotations

import datetime as dt
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIMIT_MB = 25


def main() -> None:
    files = sorted(HERE.glob("factors/*.py")) + sorted(HERE.glob("results/accepted/*.py.txt"))
    files += [p for p in (HERE / "NOTES.md", HERE / "results" / "ledger.csv", HERE / "results" / "holdout_openings.log")
              if p.exists()]
    files += sorted(HERE.glob("results/final_check_*.json")) + sorted(HERE.glob("*.log"))
    out = HERE / f"mining_results_{dt.datetime.now():%Y%m%d_%H%M}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in files:
            z.write(p, p.relative_to(HERE))
    mb = out.stat().st_size / 1e6
    print(f"{out.name}: {len(files)} files, {mb:.2f} MB")
    if mb > LIMIT_MB:
        print(f"larger than {LIMIT_MB} MB — probably big *.log files; delete or truncate them and run again")
    if not (HERE / "results" / "ledger.csv").exists():
        print("warning: results/ledger.csv not found — nothing was mined yet?")


if __name__ == "__main__":
    main()
