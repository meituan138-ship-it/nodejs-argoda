"""Parallel downloader: 4 processes × 12 threads (JSON parsing is CPU-bound)."""
import sys
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

from ash.data import code_universe, fetch_stock, load_index


def work(chunk):
    with ThreadPoolExecutor(12) as ex:
        return sum(1 for df in ex.map(fetch_stock, chunk) if df is not None)


if __name__ == "__main__":
    codes = code_universe()
    chunks = [codes[i::16] for i in range(16)]
    with Pool(4) as p:
        for i, n in enumerate(p.imap_unordered(work, chunks)):
            print(f"chunk {i + 1}/16 done ({n} with data)", flush=True)
    for c in ("sh000300", "sh000905", "sh000852", "sh000001"):
        x = load_index(c)
        print(c, None if x is None else (x.index[0], len(x)), flush=True)
    print("DONE", flush=True)
