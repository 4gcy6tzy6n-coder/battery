"""Generate safe NASA archive inventory and unconfirmed schema reports."""

import argparse
import json
from pathlib import Path

from tristatelite.data.archive_inventory import inventory_zip
from tristatelite.data.schema_report import (
    build_schema_report,
    render_schema_markdown,
    validate_report_consistency,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/raw/battery_alt_dataset.zip"))
    parser.add_argument(
        "--download-manifest", type=Path, default=Path("data/manifests/nasa_download.json")
    )
    parser.add_argument(
        "--inventory-json",
        type=Path,
        default=Path("data/manifests/nasa_archive_inventory.json"),
    )
    parser.add_argument(
        "--schema-json", type=Path, default=Path("data/manifests/nasa_schema_report.json")
    )
    parser.add_argument(
        "--schema-markdown", type=Path, default=Path("data/manifests/nasa_schema_report.md")
    )
    return parser.parse_args()


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    inventory = inventory_zip(args.archive)
    schema = build_schema_report(args.archive)
    download_manifest = None
    if args.download_manifest.exists():
        download_manifest = json.loads(args.download_manifest.read_text(encoding="utf-8"))
    validate_report_consistency(inventory, schema, download_manifest)

    _write_text(args.inventory_json, json.dumps(inventory, indent=2, ensure_ascii=False) + "\n")
    _write_text(args.schema_json, json.dumps(schema, indent=2, ensure_ascii=False) + "\n")
    _write_text(args.schema_markdown, render_schema_markdown(inventory, schema))
    print(f"Inventory JSON: {args.inventory_json}")
    print(f"Schema JSON: {args.schema_json}")
    print(f"Schema Markdown: {args.schema_markdown}")


if __name__ == "__main__":
    main()
