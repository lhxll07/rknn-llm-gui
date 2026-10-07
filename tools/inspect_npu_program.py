"""Inspect recovered RKLLM program tables, tasks and register commands."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from native_converter.program import inspect_program, read_program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="RKLLM file, rkplan, or raw program.bin")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--commands", action="store_true", help="Include every register command")
    args = parser.parse_args()
    result = inspect_program(read_program(args.input), args.commands)
    if args.output:
        with args.output.open("x") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
            file.write("\n")
        print(json.dumps({"output": str(args.output), "program_bytes": result["program_bytes"],
                          "operators": len(result["operators"]), "tensors": len(result["tensors"]),
                          "tasks": len(result["tasks"]), "regcmd_words": result["regcmd_words"]}, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
