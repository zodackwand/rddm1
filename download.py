import sys
import time
import zipfile

import requests
from tqdm import tqdm

from data import DATASETS, SOURCES, get_raw_dir

ATTEMPTS = 5


def download_once(url, dest):
    # Servers here ignore Range requests, so a broken download restarts from zero
    tmp = dest.with_name(dest.name + ".part")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0)) or None
            with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    bar.update(len(chunk))
        tmp.rename(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def download_file(url, dest):
    if dest.exists():
        print(f"skip {dest.name}")
        return

    for attempt in range(1, ATTEMPTS + 1):
        try:
            download_once(url, dest)
            return
        except Exception as e:
            print(f"attempt {attempt}/{ATTEMPTS} failed: {e}")
            if attempt < ATTEMPTS:
                time.sleep(10)
    raise RuntimeError(f"could not download {dest.name} after {ATTEMPTS} attempts")


def download_raw_dataset(name):
    out = get_raw_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    for filename, url in SOURCES[name]:
        path = out / filename
        download_file(url, path)
        marker = out / f".{filename}.extracted"
        if path.suffix == ".zip" and not marker.exists():
            print(f"extract {filename}")
            with zipfile.ZipFile(path) as z:
                z.extractall(out)
            marker.touch()


if __name__ == "__main__":
    # python download.py - downloads all datasets
    # python download.py BIDMC CAPNO - downloads specified
    failed = []
    for name in sys.argv[1:] or DATASETS:
        try:
            download_raw_dataset(name)
        except Exception as e:
            print(f"{name}: {e}")
            failed.append(name)

    if failed:
        print(f"\nfailed: {', '.join(failed)}")
    else:
        print("\nall datasets ready")
