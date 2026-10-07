from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import sys
import zipfile

from .container import Reader
from .converter import convert, prepare


def main() -> int:
    parser = argparse.ArgumentParser(description="Experimental portable Llama/Qwen3.5 FP16 RKLLM converter")
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare", help="Extract a weight-independent NPU plan from an official FP16 model")
    preparation.add_argument("--reference", required=True, type=Path)
    preparation.add_argument("--model", required=True, type=Path)
    preparation.add_argument("--output", required=True, type=Path)
    conversion = commands.add_parser("convert", help="Convert matching HF weights using NumPy and a cached NPU plan")
    conversion.add_argument("--plan", required=True, type=Path)
    conversion.add_argument("--model", required=True, type=Path)
    conversion.add_argument("--output", required=True, type=Path)
    inspection = commands.add_parser("inspect", help="Inspect the recovered outer directory")
    inspection.add_argument("model", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.reference, args.model, args.output)
        elif args.command == "convert":
            result = convert(args.plan, args.model, args.output)
        else:
            with Reader(args.model) as reader:
                result = {"version": reader.version, "data_offset": reader.data_offset,
                          "architecture": reader.metadata.get("general.architecture", b"").decode(),
                          "toolkit_version": reader.metadata.get("rkllm.version", b"").decode(),
                          "metadata_entries": len(reader.entries), "tensors": len(reader.tensors)}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, struct.error, zipfile.BadZipFile) as exc:
        print(f"Conversion failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
