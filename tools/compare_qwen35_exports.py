"""Compare Qwen exports, allowing a Qwen pre-tokenizer fix and 1-ULP SSM exp."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from native_converter.container import Reader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    with Reader(args.reference) as reference, Reader(args.native) as native:
        metadata_changes = {}
        if reference.metadata.keys() != native.metadata.keys() or reference.tensors != native.tensors:
            raise ValueError("Metadata keys or tensor directories differ")
        for key, value in reference.metadata.items():
            other = native.metadata[key]
            if value != other:
                if key != "tokenizer.ggml.pre" or value != b"default" or other != b"qwen2":
                    raise ValueError(f"Unexpected metadata difference: {key}")
                metadata_changes[key] = {"reference": value.decode(), "native": other.decode()}
        payload_size = len(reference.buffer) - reference.data_offset
        if payload_size != len(native.buffer) - native.data_offset:
            raise ValueError("Export payload sizes differ")
        differences = []
        for start in range(0, payload_size, 8 * 1024 * 1024):
            end = start + 8 * 1024 * 1024
            x = reference.buffer[reference.data_offset + start:reference.data_offset + end]
            y = native.buffer[native.data_offset + start:native.data_offset + end]
            if x != y:
                indices = np.flatnonzero(np.frombuffer(x, "u1") != np.frombuffer(y, "u1"))
                differences.extend((indices + start).tolist())
        allowed = set()
        tensors = []
        for tensor in reference.tensors:
            if not tensor.name.endswith("ssm_a"):
                continue
            offset = reference.data_offset + tensor.offset
            allowed.update(range(tensor.offset, tensor.offset + 4 * tensor.shape[0]))
            x = np.ndarray(tensor.shape, "<f4", buffer=reference.buffer, offset=offset)
            y = np.ndarray(tensor.shape, "<f4", buffer=native.buffer, offset=native.data_offset + tensor.offset)
            indices = np.flatnonzero(x.view("u4") != y.view("u4"))
            if indices.size:
                ulp = np.abs(x.view("i4").astype("i8") - y.view("i4").astype("i8"))[indices]
                if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(ulp > 1):
                    raise ValueError(f"Unexpected exponent differences: {tensor.name}")
                tensors.append({"name": tensor.name, "indices": indices.tolist(),
                                "reference": x[indices].tolist(), "native": y[indices].tolist(),
                                "ulp": ulp.tolist()})
        if not set(differences) <= allowed:
            raise ValueError("Unexpected difference outside SSM exponent values")
        report = {"model": "Qwen/Qwen3.5-0.8B", "source": "ModelScope",
                  "toolkit_reference": "1.3.1", "target": "rk3588", "num_npu_core": 1,
                  "precision": "FP16", "max_context": reference.metadata["rkllm.max_context"],
                  "bytes": len(native.buffer), "official_byte_identical": not differences and not metadata_changes,
                  "different_payload_bytes": len(differences), "metadata_changes": metadata_changes,
                  "different_fp32_values": sum(len(t["indices"]) for t in tensors),
                  "differences": tensors, "native_sha256": hashlib.sha256(native.buffer).hexdigest(),
                  "reference_sha256": hashlib.sha256(reference.buffer).hexdigest(),
                  "program_reused": True, "vision_exported": False}
    with args.report.open("x") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "differences"}, indent=2))


if __name__ == "__main__":
    main()
