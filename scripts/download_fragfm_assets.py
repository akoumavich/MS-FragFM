"""Fetch FragFM's released NPGen checkpoints and processed data (Google Drive).

NPGen is the natural-product benchmark, the closest of FragFM's four datasets to
MassSpecGym chemistry, so it is the one worth profiling.
"""

import subprocess
import sys
import zipfile
from pathlib import Path

FRAGFM = Path(__file__).resolve().parents[2] / "FragFM"

ASSETS = {
    "19qJ1Dds1AkZHbLL6bIY-LeN0BsJCYQb9": ("ae_model.zip", FRAGFM / "save"),
    "17IzYwvqpcfQCY_1TJWG8yJzScUj51xBR": ("flow_model.zip", FRAGFM / "save"),
    "1ldyqntL2uWUswZoJjpWi8mq3L2-jdr68": ("npgen_processed.zip", FRAGFM / "data" / "processed"),
}


def main():
    for file_id, (name, dest) in ASSETS.items():
        dest.mkdir(parents=True, exist_ok=True)
        archive = dest / name
        if not archive.exists():
            subprocess.run(
                [sys.executable, "-m", "gdown", "--id", file_id, "-O", str(archive)],
                check=True,
            )
        print(f"extracting {archive} -> {dest}")
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)

    print("\ncontents:")
    for p in sorted((FRAGFM / "save").rglob("*"))[:40]:
        print("  ", p.relative_to(FRAGFM))
    for p in sorted((FRAGFM / "data" / "processed").glob("*")):
        print("  ", p.relative_to(FRAGFM))


if __name__ == "__main__":
    main()
