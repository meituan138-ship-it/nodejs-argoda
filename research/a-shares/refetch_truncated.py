"""Re-download stocks whose cached history looks truncated (length a multiple of
640 = one API page and starting after 2012), slowly to stay under the WAF limit."""
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from ash.data import FetchError, fetch_stock


def suspicious():
    out = []
    for f in glob.glob("data/daily/*.parquet"):
        d = pd.read_parquet(f)
        if len(d) and len(d) % 640 == 0 and d.index[0] > pd.Timestamp("2012-01-10"):
            out.append(f)
    return out


for rnd in range(3):
    sus = suspicious()
    print(f"round {rnd}: {len(sus)} suspicious", flush=True)
    if not sus:
        break
    for f in sus:
        os.remove(f)
    codes = [os.path.basename(f)[:8] for f in sus]

    def one(c):
        try:
            fetch_stock(c)
            time.sleep(0.3)
            return True
        except FetchError:
            return False

    with ThreadPoolExecutor(4) as ex:
        ok = sum(ex.map(one, codes))
    print(f"  refetched {ok}/{len(codes)}", flush=True)
print("DONE", len(suspicious()), "still suspicious")
