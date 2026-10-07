"""Safetensors reading and recovered single-core RK3588 FP16 weight layout."""

from __future__ import annotations

import json
import math
from pathlib import Path
import struct

import numpy as np


class Weights:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.files = {}
        self.tensors = {}
        try:
            self._load()
        except Exception:
            self.close()
            raise

    def _load(self):
        index = self.directory / "model.safetensors.index.json"
        weight_map = None
        if index.is_file():
            weight_map = json.loads(index.read_text())["weight_map"]
            names = sorted(set(weight_map.values()))
        else:
            names = ["model.safetensors"]
        for name in names:
            path = (self.directory / name).resolve()
            if not path.is_relative_to(self.directory.resolve()):
                raise ValueError("Safetensors shard must be inside the model directory")
            size = path.stat().st_size
            with path.open("rb") as file:
                prefix = file.read(8)
                if len(prefix) != 8:
                    raise ValueError("Truncated safetensors header")
                header_size = struct.unpack("<Q", prefix)[0]
                if header_size > 64 * 1024 * 1024 or header_size > size - 8:
                    raise ValueError("Invalid safetensors header size")
                header = json.loads(file.read(header_size))
            self.files[name] = np.memmap(path, mode="r", dtype=np.uint8)
            for key, value in header.items():
                if key == "__metadata__":
                    continue
                if key in self.tensors:
                    raise ValueError(f"Duplicate tensor: {key}")
                dtype = {"F32": "<f4", "F16": "<f2", "BF16": "<u2"}.get(value["dtype"])
                if dtype is None:
                    raise ValueError(f"Unsupported source dtype for {key}: {value['dtype']}")
                shape = tuple(value["shape"])
                start, end = value["data_offsets"]
                if not shape or any(not isinstance(n, int) or n <= 0 for n in shape):
                    raise ValueError(f"Invalid shape for {key}")
                if (start < 0 or end < start or end + header_size + 8 > size
                        or end - start != math.prod(shape) * np.dtype(dtype).itemsize):
                    raise ValueError(f"Invalid data range for {key}")
                self.tensors[key] = (name, value["dtype"], shape, start + header_size + 8)
        if weight_map is not None:
            if set(weight_map) != set(self.tensors):
                raise ValueError("Safetensors index does not match shard contents")
            for key, name in weight_map.items():
                if self.tensors[key][0] != name:
                    raise ValueError(f"Incorrect shard mapping for {key}")

    def get(self, name: str) -> np.ndarray:
        if name not in self.tensors:
            raise ValueError(f"Missing weight: {name}")
        shard, kind, shape, offset = self.tensors[name]
        dtype = {"F32": "<f4", "F16": "<f2", "BF16": "<u2"}[kind]
        result = np.ndarray(shape, dtype=dtype, buffer=self.files[shard], offset=offset)
        if kind == "BF16":
            result = (result.astype("<u4") << 16).view("<f4")
        return result

    def close(self):
        for file in self.files.values():
            file._mmap.close()
        self.files.clear()


def source_name(name: str, profile: dict | None = None) -> str:
    if profile and profile.get("model_type") == "qwen3_5":
        from .qwen35 import source_name as qwen_source_name
        return qwen_source_name(name)
    direct = {"token_embd.weight": "model.embed_tokens.weight",
              "output_norm.weight": "model.norm.weight", "output.weight": "lm_head.weight"}
    if name in direct:
        return direct[name]
    parts = name.split(".")
    if len(parts) != 4 or parts[0] != "blk" or not parts[1].isdigit() or parts[3] != "weight":
        raise ValueError(f"Unsupported Llama tensor: {name}")
    mapping = {
        "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
        "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
        "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj",
        "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm",
    }
    if parts[2] not in mapping:
        raise ValueError(f"Unsupported Llama tensor: {name}")
    return f"model.layers.{parts[1]}.{mapping[parts[2]]}.weight"


def is_matrix(name: str, profile: dict | None = None) -> bool:
    if profile and profile.get("model_type") == "qwen3_5":
        from .qwen35 import is_matrix as qwen_is_matrix
        return qwen_is_matrix(name)
    return name == "output.weight" or (name.startswith("blk.") and "norm" not in name)


def pack_matrix(matrix: np.ndarray, name: str, profile: dict) -> bytes:
    """FP16 B layout: N/16, K/32, 16, 32; Q/K also need RoPE permutation."""
    if matrix.ndim != 2:
        raise ValueError("Matrix must have rank two")
    n, k = matrix.shape
    if n % 16 or k % 32:
        raise ValueError("This prototype requires N divisible by 16 and K divisible by 32")
    data = matrix.astype("<f2")
    if not np.isfinite(data).all():
        raise ValueError(f"Weight {name} cannot be represented as finite FP16")
    if profile.get("model_type") != "qwen3_5" and (".attn_q." in name or ".attn_k." in name):
        heads = profile["num_attention_heads"] if ".attn_q." in name else profile["num_key_value_heads"]
        if n % (2 * heads):
            raise ValueError("Unsupported RoPE head dimensions")
        data = data.reshape(heads, 2, n // heads // 2, k).transpose(0, 2, 1, 3).reshape(n, k)
    return data.reshape(n // 16, 16, k // 32, 32).transpose(0, 2, 1, 3).tobytes()
