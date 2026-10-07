"""A bounded Llama/Qwen3.5 FP16 converter with a cached NPU program.

Preparing a plan extracts a weight-independent program from a matching
official model. Conversion reads HF weights and writes all model weights
itself, with no Toolkit/native library, subprocess, or network dependency.
Generating NPU programs for new configurations is not yet implemented.
"""

from __future__ import annotations

import base64
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import struct
import zipfile

import numpy as np

from .container import Entry, Reader, TensorInfo, align, write_directory
from .weights import Weights, is_matrix, pack_matrix, source_name
from . import qwen35


FORMAT = "rkllm-native-plan-v1"
MAX_PLAN_BYTES = 256 * 1024 * 1024
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "generation_config.json")


def model_profile(directory: Path) -> dict:
    config = json.loads((directory / "config.json").read_text())
    if config.get("model_type") == "qwen3_5":
        return qwen35.profile(directory)
    if (config.get("model_type") != "llama"
            or config.get("architectures") != ["LlamaForCausalLM"]
            or config.get("hidden_act", "silu") != "silu"
            or config.get("attention_bias", False) or config.get("mlp_bias", False)
            or config.get("tie_word_embeddings", False)
            or config.get("pretraining_tp", 1) != 1
            or config.get("rope_scaling") or config.get("sliding_window")):
        raise ValueError("Only ordinary LlamaForCausalLM with SiLU, separate lm_head and no biases/scaling is supported")
    keys = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
            "vocab_size", "max_position_embeddings")
    profile = {key: config[key] for key in keys}
    if any(type(value) is not int or value <= 0 for value in profile.values()):
        raise ValueError("Model dimensions must be positive integers")
    profile["num_key_value_heads"] = config.get("num_key_value_heads", profile["num_attention_heads"])
    profile["head_dim"] = config.get("head_dim", profile["hidden_size"] // profile["num_attention_heads"])
    if (type(profile["num_key_value_heads"]) is not int or profile["num_key_value_heads"] <= 0
            or type(profile["head_dim"]) is not int or profile["head_dim"] <= 0
            or profile["hidden_size"] != profile["head_dim"] * profile["num_attention_heads"]
            or profile["num_attention_heads"] % profile["num_key_value_heads"]):
        raise ValueError("Unsupported attention head configuration")
    rope = config.get("rope_parameters") or {}
    if rope.get("rope_type", "default") != "default" or set(rope) - {"rope_type", "rope_theta"}:
        raise ValueError("Only default RoPE is supported")
    profile["rope_theta"] = float(rope.get("rope_theta", config.get("rope_theta", 10000.0)))
    profile["rms_norm_eps"] = float(config.get("rms_norm_eps", 1e-6))
    if not all(math.isfinite(profile[key]) and profile[key] > 0 for key in ("rope_theta", "rms_norm_eps")):
        raise ValueError("Invalid RoPE or normalization parameters")
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        profile[key] = config.get(key)
    return profile


def tokenizer_hashes(directory: Path, profile: dict | None = None) -> dict:
    if not (directory / "tokenizer.json").is_file():
        raise ValueError("A tokenizer.json is required by this prototype")
    result = {}
    files = TOKENIZER_FILES + (("chat_template.jinja", "merges.txt", "vocab.json")
                               if profile and profile.get("model_type") == "qwen3_5" else ())
    for name in files:
        path = directory / name
        if path.is_file():
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def expected_tensors(profile: dict) -> dict:
    if profile.get("model_type") == "qwen3_5":
        return qwen35.expected_tensors(profile)
    h, f, v = (profile[name] for name in ("hidden_size", "intermediate_size", "vocab_size"))
    kv = profile["num_key_value_heads"] * profile["head_dim"]
    result = {"token_embd.weight": (v, h)}
    for layer in range(profile["num_hidden_layers"]):
        prefix = f"blk.{layer}."
        shapes = {"attn_q": (h, h), "attn_k": (kv, h), "attn_v": (kv, h),
                  "attn_output": (h, h), "ffn_gate": (f, h), "ffn_up": (f, h),
                  "ffn_down": (h, f), "attn_norm": (h,), "ffn_norm": (h,)}
        result.update({prefix + name + ".weight": shape for name, shape in shapes.items()})
    result["output_norm.weight"] = (h,)
    result["output.weight"] = (v, h)
    return result


def _json_entry(entry: Entry) -> dict:
    data = asdict(entry)
    if entry.kind == 8:
        data["value"] = base64.b64encode(entry.value).decode()
    elif entry.kind == 9 and entry.element_kind == 8:
        data["value"] = [base64.b64encode(value).decode() for value in entry.value]
    return data


def _entry_from_json(data: dict) -> Entry:
    entry = Entry(**data)
    if entry.kind == 8:
        entry.value = base64.b64decode(entry.value, validate=True)
    elif entry.kind == 9 and entry.element_kind == 8:
        entry.value = [base64.b64decode(value, validate=True) for value in entry.value]
    return entry


def _validate_tokenizer_metadata(metadata: dict):
    if (metadata.get("tokenizer.ggml.model") == b"gpt2"
            and not isinstance(metadata.get("tokenizer.ggml.merges"), list)):
        raise ValueError("GPT-2 tokenizer metadata is missing BPE merges, a known Runtime loading failure. Use a complete BPE reference")


def _validate_reference(reader: Reader, profile: dict, directory: Path):
    metadata = reader.metadata
    _validate_tokenizer_metadata(metadata)
    required = {"general.file_type": 1,
                "rkllm.platform": 0, "rkllm.core_num": 1, "rkllm.version": b"1.3.1",
                "rkllm.has_embedding": True, "rkllm.has_tokenizer": True,
                "rkllm.lora_quant": False}
    for key, value in required.items():
        if metadata.get(key) != value:
            raise ValueError(f"Reference is not a supported RK3588 single-core FP16 1.3.1 model: {key}")
    if profile.get("model_type") == "qwen3_5":
        qwen35.validate_reference(metadata, profile)
        fields = {}
    else:
        if metadata.get("general.architecture") != b"llama":
            raise ValueError("Reference architecture does not match Llama")
        fields = {"llama.block_count": "num_hidden_layers", "llama.vocab_size": "vocab_size",
              "llama.context_length": "max_position_embeddings", "llama.embedding_length": "hidden_size",
              "llama.feed_forward_length": "intermediate_size", "llama.attention.head_count": "num_attention_heads",
              "llama.attention.head_count_kv": "num_key_value_heads", "llama.attention.key_length": "head_dim",
              "llama.attention.value_length": "head_dim", "llama.rope.dimension_count": "head_dim",
              "llama.rope.freq_base": "rope_theta", "llama.attention.layer_norm_rms_epsilon": "rms_norm_eps"}
    for key, config_key in fields.items():
        value = profile[config_key]
        if isinstance(value, float):
            value = float(np.float32(value))
        if metadata.get(key) != value:
            raise ValueError(f"Reference configuration does not match source config: {key}")
    if reader.alignment != 32:
        raise ValueError("Only the observed 32-byte alignment is supported")
    context = metadata.get("rkllm.max_context")
    if type(context) is not int or not 32 <= context <= 16384 or context % 32:
        raise ValueError("Unsupported NPU maximum context")
    expected = expected_tensors(profile)
    if {tensor.name for tensor in reader.tensors} != set(expected):
        raise ValueError("Reference contains unsupported or missing tensors")
    for tensor in reader.tensors:
        dtype = 1 if is_matrix(tensor.name, profile) or tensor.name == "token_embd.weight" else 0
        if tensor.shape != expected[tensor.name] or tensor.kind != dtype:
            raise ValueError(f"Unsupported reference tensor shape or type: {tensor.name}")
    tokenizer = json.loads((directory / "tokenizer.json").read_text())
    model = tokenizer.get("model", {})
    if model.get("type") not in {"BPE", "WordLevel"} or not isinstance(model.get("vocab"), dict):
        raise ValueError("Plan preparation currently supports BPE or WordLevel tokenizer.json")
    vocab = dict(model["vocab"])
    for token in tokenizer.get("added_tokens", []):
        vocab[token["content"]] = token["id"]
    if profile.get("model_type") == "qwen3_5":
        tokenizer_config = json.loads((directory / "tokenizer_config.json").read_text())
        for token_id, token in tokenizer_config.get("added_tokens_decoder", {}).items():
            if token["content"] in vocab and vocab[token["content"]] != int(token_id):
                raise ValueError("Conflicting tokenizer_config token ID")
            vocab[token["content"]] = int(token_id)
    tokens = [None] * profile["vocab_size"]
    for token, token_id in vocab.items():
        if type(token_id) is not int or not 0 <= token_id < len(tokens) or tokens[token_id] is not None:
            raise ValueError("Tokenizer IDs must be unique and inside vocab_size")
        tokens[token_id] = token.encode("utf-8")
    if profile.get("model_type") == "qwen3_5":
        # Qwen pads its embedding vocabulary beyond tokenizer vocabulary.
        # Tokenizer_config also supplies extra audio special tokens.
        for i, value in enumerate(tokens):
            if value is None:
                tokens[i] = f"[PAD{i}]".encode()
    if None in tokens or tokens != metadata.get("tokenizer.ggml.tokens"):
        raise ValueError("Reference token list does not match tokenizer.json")


def _program_size(header: bytes) -> int:
    if len(header) < 256:
        raise ValueError("Truncated NPU program header")
    magic, version, commands, metadata, memory = struct.unpack_from("<5Q", header)
    if magic != 0x4D4C4C52 or version != 1 or commands == 0 or metadata == 0:
        raise ValueError("Unsupported NPU program envelope")
    size = 256 + commands + metadata + memory
    if size > MAX_PLAN_BYTES:
        raise ValueError("NPU program exceeds the prototype's plan size limit")
    return size


def prepare(reference: Path, model: Path, output: Path) -> dict:
    profile = model_profile(model)
    hashes = tokenizer_hashes(model, profile)
    with Reader(reference) as reader:
        _validate_reference(reader, profile, model)
        offset = 0
        matrix_bytes = 0
        for tensor in reader.tensors:
            if tensor.offset != offset:
                raise ValueError(f"Unrecognized ordinary tensor placement: {tensor.name}")
            if is_matrix(tensor.name, profile):
                n, k = tensor.shape
                if n % 16 or k % 32:
                    raise ValueError("Padded or differently tiled matrices are not supported yet")
                matrix_bytes += n * k * 2
            else:
                offset = align(offset + math.prod(tensor.shape) * (2 if tensor.kind == 1 else 4))
        program_start = reader.data_offset + offset
        header = reader.buffer[program_start:program_start + 256]
        program_size = _program_size(header)
        program_end = program_start + program_size
        footer = b"rkllm-toolkit version: 1.3.1"
        expected_size = program_end + matrix_bytes + len(footer)
        if len(reader.buffer) != expected_size or reader.buffer[-len(footer):] != footer:
            raise ValueError("Reference tail layout differs from the recovered FP16 layout")
        program = reader.buffer[program_start:program_end]
        manifest = {
            "format": FORMAT, "profile": profile, "tokenizer_sha256": hashes,
            "entries": [_json_entry(entry) for entry in reader.entries],
            "tensors": [asdict(tensor) for tensor in reader.tensors],
            "program_sha256": hashlib.sha256(program).hexdigest(),
            "compiler_origin": "Program extracted from an official 1.3.1 model; compiler not reimplemented",
            "max_context": reader.metadata.get("rkllm.max_context"),
            "reference_size": len(reader.buffer),
        }
        if profile.get("model_type") == "qwen3_5":
            corrected_pre = qwen35.corrected_tokenizer_pre(model, reader.metadata.get("tokenizer.ggml.pre"))
            for entry in manifest["entries"]:
                if entry["key"] == "tokenizer.ggml.pre":
                    entry["value"] = base64.b64encode(corrected_pre).decode()
            manifest["tokenizer_correction"] = {
                "key": "tokenizer.ggml.pre", "reference": reader.metadata["tokenizer.ggml.pre"].decode(),
                "native": corrected_pre.decode(), "reason": "Qwen source regex splits individual digits; Runtime default differs"}
            weights = Weights(model)
            try:
                required = {source_name(name, profile) for name in expected_tensors(profile)}
                auxiliary = set(weights.tensors) - required
                if any(not name.startswith(("model.visual.", "mtp.")) for name in auxiliary):
                    raise ValueError("Unrecognized auxiliary Qwen tensors")
                manifest["excluded_language_auxiliary_shapes"] = {
                    name: weights.tensors[name][2] for name in sorted(auxiliary)}
                manifest["weight_rounding"] = "SSM -exp(A_log) uses float64 exp rounded to FP32; may differ from Torch/MKL by 1 ULP"
            finally:
                weights.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with output.open("xb") as file:
            created = True
            with zipfile.ZipFile(file, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=True))
                archive.writestr("program.bin", program)
    except Exception:
        if created:
            output.unlink(missing_ok=True)
        raise
    return {"plan": str(output), "program_bytes": len(program), "profile": profile}


def load_plan(path: Path) -> tuple[dict, bytes]:
    with zipfile.ZipFile(path) as archive:
        if sorted(archive.namelist()) != ["manifest.json", "program.bin"]:
            raise ValueError("Unrecognized plan archive")
        if any(info.file_size > MAX_PLAN_BYTES for info in archive.infolist()):
            raise ValueError("Plan archive is too large")
        manifest = json.loads(archive.read("manifest.json"))
        program = archive.read("program.bin")
    if (manifest.get("format") != FORMAT
            or hashlib.sha256(program).hexdigest() != manifest.get("program_sha256")
            or _program_size(program) != len(program)):
        raise ValueError("Invalid or corrupted NPU plan")
    return manifest, program


def convert(plan: Path, model: Path, output: Path) -> dict:
    manifest, program = load_plan(plan)
    profile = model_profile(model)
    if profile != manifest["profile"]:
        raise ValueError("Model configuration differs from the cached NPU program; a matching plan is required")
    if tokenizer_hashes(model, profile) != manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer or generation configuration differs from the plan")
    entries = [_entry_from_json(entry) for entry in manifest["entries"]]
    metadata = {entry.key: entry.value for entry in entries}
    _validate_tokenizer_metadata(metadata)
    if profile.get("model_type") == "qwen3_5":
        if metadata.get("tokenizer.ggml.pre") != qwen35.corrected_tokenizer_pre(model, b"qwen2"):
            raise ValueError("Qwen plan has the known incorrect default pre-tokenizer; prepare a new corrected plan")
    tensors = [TensorInfo(item["name"], tuple(item["shape"]), item["kind"], item["offset"])
               for item in manifest["tensors"]]
    expected = expected_tensors(profile)
    if {tensor.name: tensor.shape for tensor in tensors} != expected:
        raise ValueError("Plan tensor directory does not match its model profile")
    if len(tensors) != len(expected):
        raise ValueError("Plan has duplicate tensor names")
    for tensor in tensors:
        kind = 1 if is_matrix(tensor.name, profile) or tensor.name == "token_embd.weight" else 0
        if tensor.kind != kind:
            raise ValueError(f"Unsupported plan tensor type: {tensor.name}")
    weights = Weights(model)
    created = False
    try:
        required = {source_name(name, profile) for name in expected}
        auxiliary = manifest.get("excluded_language_auxiliary_shapes", {})
        if profile.get("model_type") != "qwen3_5" and auxiliary:
            raise ValueError("Auxiliary tensors are only allowed in the Qwen language profile")
        if any(not name.startswith(("model.visual.", "mtp.")) for name in auxiliary):
            raise ValueError("Unrecognized auxiliary tensors in plan")
        if set(weights.tensors) != required | set(auxiliary):
            raise ValueError("Source contains missing or additional tensors; adapters and other architectures are unsupported")
        for name, shape in auxiliary.items():
            if weights.tensors[name][2] != tuple(shape):
                raise ValueError(f"Auxiliary source shape differs from plan: {name}")
        for tensor in tensors:
            shape = qwen35.source_shape(tensor.name, tensor.shape) if profile.get("model_type") == "qwen3_5" else tensor.shape
            if weights.tensors[source_name(tensor.name, profile)][2] != shape:
                raise ValueError(f"Source shape does not match plan: {tensor.name}")
        def transformed(name):
            data = weights.get(source_name(name, profile))
            return qwen35.transform(data, name) if profile.get("model_type") == "qwen3_5" else data
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as file:
            created = True
            data_start = write_directory(file, entries, tensors)
            for tensor in tensors:
                if tensor.offset != file.tell() - data_start:
                    raise ValueError("Plan ordinary data offsets are inconsistent")
                if is_matrix(tensor.name, profile):
                    continue
                data = transformed(tensor.name).astype("<f2" if tensor.kind == 1 else "<f4")
                if not np.isfinite(data).all():
                    raise ValueError(f"Non-finite weight: {tensor.name}")
                file.write(data.tobytes())
                file.write(b"\0" * (align(file.tell() - data_start) - (file.tell() - data_start)))
                del data
            file.write(program)
            for tensor in tensors:
                if is_matrix(tensor.name, profile):
                    file.write(pack_matrix(transformed(tensor.name), tensor.name, profile))
            file.write(b"rkllm-toolkit version: 1.3.1")
    except Exception:
        if created:
            output.unlink(missing_ok=True)
        raise
    finally:
        weights.close()
    return {"output": str(output), "size": output.stat().st_size, "quantization": "FP16",
            "target": "RK3588", "cores": 1, "compiler": "cached NPU program",
            "runtime_verified": False, "language_only": profile.get("model_type") == "qwen3_5",
            "tokenizer_correction": manifest.get("tokenizer_correction"),
            "weight_rounding": manifest.get("weight_rounding", "reference matching FP16")}
