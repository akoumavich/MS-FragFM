"""Fetch FragFM's released NPGen checkpoints and processed data (Google Drive).

NPGen is the natural-product benchmark, the closest of FragFM's four datasets to
MassSpecGym chemistry, so it is the one worth profiling.
"""

import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

FRAGFM = Path(__file__).resolve().parents[2] / "FragFM"

ASSETS = {
    "19qJ1Dds1AkZHbLL6bIY-LeN0BsJCYQb9": ("ae_model.bin", FRAGFM / "save"),
    "17IzYwvqpcfQCY_1TJWG8yJzScUj51xBR": ("flow_model.bin", FRAGFM / "save"),
    "1ldyqntL2uWUswZoJjpWi8mq3L2-jdr68": ("npgen_processed.bin", FRAGFM / "data" / "processed"),
}


def extract(archive, dest):
    """The README calls these zips; Drive serves at least one that is not.
    Sniff the container instead of trusting the extension."""
    head = archive.open("rb").read(512)
    if head[:4] == bytes.fromhex("504b0304"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
        return "zip"
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            t.extractall(dest)
        return "tar"
    if head[:2] == bytes.fromhex("1f8b"):
        with tarfile.open(archive, "r:gz") as t:
            t.extractall(dest)
        return "tar.gz"
    if head[:2] == b"PK":
        raise SystemExit(f"{archive}: truncated zip -- delete it and retry")
    if head.lstrip()[:15] == b"<!DOCTYPE html>" or b"<html" in head[:200].lower():
        raise SystemExit(
            f"{archive}: Google Drive returned an HTML page, not the file "
            f"(quota or confirmation interstitial). Delete it and retry, or "
            f"download by hand."
        )
    raise SystemExit(f"{archive}: unrecognised container, first bytes {head[:16]!r}")


def main():
    for file_id, (name, dest) in ASSETS.items():
        dest.mkdir(parents=True, exist_ok=True)
        archive = dest / name
        if not archive.exists():
            subprocess.run(
                # gdown 5.x dropped --id; the bare id is still accepted.
                [sys.executable, "-m", "gdown", file_id, "-O", str(archive)],
                check=True,
            )
        kind = extract(archive, dest)
        print(f"extracted {archive.name} ({kind}) -> {dest}")

    print("\ncontents:")
    for p in sorted((FRAGFM / "save").rglob("*"))[:40]:
        print("  ", p.relative_to(FRAGFM))
    for p in sorted((FRAGFM / "data" / "processed").glob("*")):
        print("  ", p.relative_to(FRAGFM))


if __name__ == "__main__":
    main()
