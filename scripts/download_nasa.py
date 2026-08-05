"""Download the official NASA battery archive and record provenance."""

import argparse
import json
from pathlib import Path

from tristatelite.data.download import download_file

DEFAULT_URL = "https://data.nasa.gov/docs/legacy/battery_alt_dataset.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=Path("data/raw/battery_alt_dataset.zip"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/nasa_download.json"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provenance = download_file(args.url, args.output, force=args.force)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Archive: {args.output}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
