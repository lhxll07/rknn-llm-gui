"""Package a matching local model, plan and reference for RK3588 validation."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parent.parent
RESEARCH = ROOT / "gui/runs/native-converter-research"
BPE_FIXTURE = RESEARCH / "bpe-board-fixtures/h128-f256-l1-seed20261007"


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        digest = hashlib.sha256()
        while block := file.read(1024 * 1024):
            digest.update(block)
        return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=BPE_FIXTURE / "model")
    parser.add_argument("--plan", type=Path, default=BPE_FIXTURE / "model.rkplan")
    parser.add_argument("--reference", type=Path, default=BPE_FIXTURE / "official.rkllm")
    parser.add_argument("--smoke-binary", type=Path, help="Optional prebuilt AArch64 verification program")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = [
        (ROOT / "LICENSE", "LICENSE"),
        (ROOT / "tools/run_native_board_check.sh", "run-board-check.sh"),
        (ROOT / "tools/rkllm_native_smoke.cpp", "src/rkllm_native_smoke.cpp"),
        (ROOT / "rkllm-runtime/Linux/librkllm_api/include/rkllm.h", "include/rkllm.h"),
        (ROOT / "rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so", "lib/librkllmrt.so"),
        (args.plan, "fixtures/model.rkplan"),
        (args.reference, "fixtures/reference.rkllm"),
    ]
    files.extend((path, "native_converter/" + path.name)
                 for path in sorted((ROOT / "native_converter").iterdir())
                 if path.suffix == ".py" or path.name in {"README.md", "requirements.txt"})
    names = {"config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "generation_config.json", "model.safetensors.index.json"}
    index = args.model / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text())["weight_map"].values())
    else:
        shards = {"model.safetensors"}
    for name in shards:
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Shard names must be relative paths inside the model directory")
    for name in sorted(names | shards):
        path = args.model / name
        if name in shards or name in {"config.json", "tokenizer.json"} or path.is_file():
            files.append((path, "fixtures/model/" + name))
    if args.smoke_binary:
        with args.smoke_binary.open("rb") as file:
            header = file.read(20)
        if (header[:4] != b"\x7fELF" or header[4:6] != b"\x02\x01"
                or int.from_bytes(header[18:20], "little") != 183):
            raise ValueError("The prebuilt verification program must be a little-endian AArch64 ELF")
        files.append((args.smoke_binary, "rkllm_native_smoke"))
    for path, _ in files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing validation material: {path}; supply --model, --plan and --reference")
    with zipfile.ZipFile(args.plan) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    bundle = {"format": "rkllm-native-board-check-v1", "max_context": manifest["max_context"],
              "vocab_size": manifest["profile"]["vocab_size"],
              "reference_sha256": sha256(args.reference),
              "files_sha256": {name: sha256(path) for path, name in files},
              "numpy_conversion_only": True,
              "arm64_conversion_verified": False, "runtime_verified": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with args.output.open("xb") as output:
            created = True
            with tarfile.open(fileobj=output, mode="w:gz", dereference=True) as archive:
                prefix = "rkllm-native-board-check/"
                for path, name in files:
                    archive.add(path, arcname=prefix + name, recursive=False)
                data = (json.dumps(bundle, indent=2) + "\n").encode()
                info = tarfile.TarInfo(prefix + "bundle.json")
                info.size = len(data)
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(data))
    except Exception:
        if created:
            args.output.unlink(missing_ok=True)
        raise
    print(json.dumps({"bundle": str(args.output), "bytes": args.output.stat().st_size,
                      "runtime_verified": False}, indent=2))


if __name__ == "__main__":
    main()
