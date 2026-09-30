"""Command-line interface for the public SIESTA reproduction utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .metrics import compare_records
from .provenance import write_receipt
from .records import load_records, read_jsonl, sha256_file, validate_records, write_jsonl
from .sleap_export import export_sleap
from .split import group_disjoint_split


def _write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate(args: argparse.Namespace) -> None:
    records = load_records(args.records, images_root=args.images_root, verify_images=True)
    print(json.dumps(validate_records(records), indent=2, sort_keys=True))


def _split(args: argparse.Namespace) -> None:
    records = load_records(args.records)
    train, validation, receipt = group_disjoint_split(records, validation_fraction=args.validation_fraction, seed=args.seed)
    output = Path(args.output_dir)
    write_jsonl(output / "train.jsonl", train)
    write_jsonl(output / "validation.jsonl", validation)
    _write_json(output / "split.json", receipt)
    write_receipt(output / "run.json", operation="split", inputs=[args.records], parameters={"validation_fraction": args.validation_fraction, "seed": args.seed}, result=receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def _export(args: argparse.Namespace) -> None:
    records = load_records(args.records)
    result = export_sleap(records, args.output, manifest_path=args.records, images_root=args.images_root, coordinate_scale=args.coordinate_scale)
    write_receipt(str(args.output) + ".run.json", operation="export-sleap", inputs=[args.records], parameters={"coordinate_scale": args.coordinate_scale}, result=result)
    print(json.dumps(result, indent=2, sort_keys=True))


def _evaluate(args: argparse.Namespace) -> None:
    result = compare_records(read_jsonl(args.predictions), read_jsonl(args.reference), thresholds=args.thresholds)
    _write_json(args.output, result)
    write_receipt(str(args.output) + ".run.json", operation="evaluate", inputs=[args.predictions, args.reference], parameters={"thresholds": args.thresholds}, result={"metrics_sha256": sha256_file(args.output)})
    print(json.dumps(result, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="siesta-repro", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate VLM records, source images and provenance")
    validate.add_argument("records", type=Path)
    validate.add_argument("--images-root", type=Path)
    validate.set_defaults(func=_validate)
    split = commands.add_parser("split", help="make a deterministic source-group-disjoint split")
    split.add_argument("records", type=Path)
    split.add_argument("--output-dir", type=Path, required=True)
    split.add_argument("--validation-fraction", type=float, default=1 / 6)
    split.add_argument("--seed", type=int, default=20260910)
    split.set_defaults(func=_split)
    export = commands.add_parser("export-sleap", help="export VLM coordinates as SLEAP user-label packages")
    export.add_argument("records", type=Path)
    export.add_argument("output", type=Path)
    export.add_argument("--images-root", type=Path)
    export.add_argument("--coordinate-scale", type=float, default=1.0)
    export.set_defaults(func=_export)
    evaluate = commands.add_parser("evaluate", help="compare already-associated crop-level records")
    evaluate.add_argument("predictions", type=Path)
    evaluate.add_argument("reference", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--thresholds", type=float, nargs="+", default=[2.0, 4.0, 8.0, 16.0])
    evaluate.set_defaults(func=_evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
