"""Create a fresh Qwen3.5 language reference using the exact repository SDK."""

import argparse
from pathlib import Path
import sys
import tempfile
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-context", type=int, default=128)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    wheel = root / "rkllm-toolkit/packages/rkllm_toolkit-1.3.1-cp310-cp310-linux_x86_64.whl"
    if args.output.exists():
        raise FileExistsError(args.output)
    work = root / "gui/runs/native-converter-research"
    work.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qwen-reference-sdk-", dir=work) as sdk:
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(sdk)
        sys.path.insert(0, sdk)
        from rkllm.api import RKLLM
        import rkllm
        print(f"Reference SDK: {rkllm.__file__}", flush=True)
        model = RKLLM()
        for label, operation in [
            ("load", lambda: model.load_huggingface(str(args.model.resolve()), device="cpu", dtype="float32")),
            ("build", lambda: model.build(do_quantization=False, optimization_level=0,
                                         quantized_dtype="w8a8", quantized_algorithm="normal",
                                         target_platform="rk3588", num_npu_core=1,
                                         max_context=args.max_context)),
            ("export", lambda: model.export_rkllm(str(args.output.resolve()))),
        ]:
            result = operation()
            print(f"{label}: {result}", flush=True)
            if result != 0:
                raise RuntimeError(f"Official {label} failed: {result}")


if __name__ == "__main__":
    main()
