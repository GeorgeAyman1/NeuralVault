"""
Run this on Colab after the vector .dat files are available in the working directory.

It builds IVF indexes for 1M, 10M, and 20M databases then zips each for Google Drive upload.

Usage (from /content/vec_db after the .dat files are present in /content/vec_db):
    python scripts/build_indexes_colab.py
"""
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.build_index import build_index

DIMENSION = 64
CONFIGS = [
    {
        "dat_file": "../OpenSubtitles_en_1M_emb_64.dat",
        "db_size": 1_000_000,
        "index_dir": "ivf_1m",
        "zip_name": "../saved_db_1m.zip",
        "n_clusters": 2048,
        "sample_size": 300_000,
    },
    {
        "dat_file": "../OpenSubtitles_en_10M_emb_64.dat",
        "db_size": 10_000_000,
        "index_dir": "ivf_10m",
        "zip_name": "../saved_db_10m.zip",
        "n_clusters": 4096,
        "sample_size": 500_000,
    },
    {
        "dat_file": "../OpenSubtitles_en_20M_emb_64.dat",
        "db_size": 20_000_000,
        "index_dir": "ivf_20m",
        "zip_name": "../saved_db_20m.zip",
        "n_clusters": 4096,
        "sample_size": 500_000,
    },
]


def zip_index_dir(index_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in index_dir.iterdir():
            zf.write(f, arcname=f"{index_dir.name}/{f.name}")
    size_mb = zip_path.stat().st_size / 1e6
    print(f"  Zipped to {zip_path}  ({size_mb:.1f} MB)")


def main():
    base = Path(__file__).parent.parent

    for cfg in CONFIGS:
        dat_path = (base / cfg["dat_file"]).resolve()
        index_dir = base / cfg["index_dir"]
        zip_path = (base / cfg["zip_name"]).resolve()

        if not dat_path.exists():
            print(f"[SKIP] {dat_path} not found")
            continue

        print(f"\n{'='*50}")
        print(f"Building index for {cfg['db_size']//1_000_000}M vectors")

        if not index_dir.exists():
            build_index(
                db_path=dat_path,
                index_dir=index_dir,
                n_clusters=cfg["n_clusters"],
                sample_size=cfg["sample_size"],
                block_size=200_000,
                seed=42,
                dim=DIMENSION,
                db_size=cfg["db_size"],
            )
        else:
            print(f"  Index already exists at {index_dir}, skipping build")

        print(f"  Zipping index directory ...")
        zip_index_dir(index_dir, zip_path)

    print("\nDone. Upload the .zip files to Google Drive and update the notebook GDRIVE IDs.")


if __name__ == "__main__":
    main()
