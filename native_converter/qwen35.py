"""Observed Qwen3.5-0.8B language weight transformations (FP16, one core).

Vision and MTP are excluded from language conversion. NPU code still comes
from a configuration-matching official reference, as with the Llama path.
"""

from pathlib import Path
import json

import numpy as np


def profile(directory: Path) -> dict:
    config = json.loads((directory / "config.json").read_text())
    text = config.get("text_config", {})
    dimensions = {"hidden_size": 1024, "intermediate_size": 3584,
                  "num_hidden_layers": 24, "num_attention_heads": 8,
                  "num_key_value_heads": 2, "head_dim": 256,
                  "vocab_size": 248320, "linear_conv_kernel_dim": 4,
                  "linear_key_head_dim": 128, "linear_value_head_dim": 128,
                  "linear_num_key_heads": 16, "linear_num_value_heads": 16,
                  "full_attention_interval": 4}
    required = {**dimensions, "hidden_act": "silu", "attention_bias": False,
                "attn_output_gate": True, "tie_word_embeddings": True,
                "layer_types": ["linear_attention"] * 3 + ["full_attention"]}
    required["layer_types"] *= 6
    if (config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]
            or config.get("model_type") != "qwen3_5"
            or not config.get("tie_word_embeddings")
            or any(text.get(k) != v for k, v in required.items())
            or text.get("mlp_only_layers", [])
            or text.get("rope_parameters") != {
                "mrope_interleaved": True, "mrope_section": [11, 11, 10],
                "rope_type": "default", "rope_theta": 10000000,
                "partial_rotary_factor": 0.25}):
        raise ValueError("Only the observed Qwen3.5-0.8B hybrid language configuration is supported")
    # Keep every configuration field, including vision fields used by the
    # reference metadata, to prevent accidental reuse on a different graph.
    return {**dimensions, "model_type": "qwen3_5", "source_config": config,
            "max_position_embeddings": text["max_position_embeddings"],
            "rms_norm_eps": text["rms_norm_eps"], "rope_theta": 10000000.0}


def expected_tensors(p: dict) -> dict:
    h, f, v = p["hidden_size"], p["intermediate_size"], p["vocab_size"]
    result = {"token_embd.weight": (v, h)}
    for layer in range(24):
        common = {"ffn_gate.weight": (f, h), "ffn_up.weight": (f, h),
                  "ffn_down.weight": (h, f), "attn_norm.weight": (h,),
                  "ffn_norm.weight": (h,)}
        if layer % 4 == 3:
            attention = {"attn_q.weight": (2048, h), "attn_gate.weight": (2048, h),
                         "attn_k.weight": (512, h), "attn_v.weight": (512, h),
                         "attn_output.weight": (h, 2048),
                         "attn_q_norm.weight": (256,), "attn_k_norm.weight": (256,)}
        else:
            attention = {"ssm_dt.bias": (16,), "ssm_a": (16,),
                         "ssm_conv1d.weight": (6144, 4), "ssm_norm.weight": (128,),
                         "ssm_out.weight": (h, 2048), "attn_qkv.weight": (6144, h),
                         "attn_gate.weight": (2048, h), "ssm_beta.weight": (16, h),
                         "ssm_alpha.weight": (16, h)}
        result.update({f"blk.{layer}.{k}": shape for k, shape in {**attention, **common}.items()})
    return {**result, "output_norm.weight": (h,), "output.weight": (v, h)}


def is_matrix(name: str) -> bool:
    suffix = name.split(".", 2)[-1] if name.startswith("blk.") else name
    return suffix in {"output.weight", "ssm_out.weight", "attn_qkv.weight",
                      "attn_gate.weight", "attn_q.weight", "attn_k.weight",
                      "attn_v.weight", "attn_output.weight", "ffn_gate.weight",
                      "ffn_up.weight", "ffn_down.weight"}


def source_name(name: str) -> str:
    direct = {"token_embd.weight": "embed_tokens.weight", "output.weight": "embed_tokens.weight",
              "output_norm.weight": "norm.weight"}
    if name in direct:
        return "model.language_model." + direct[name]
    _, layer, suffix = name.split(".", 2)
    mapping = {"ssm_dt.bias": "linear_attn.dt_bias", "ssm_a": "linear_attn.A_log",
               "ssm_conv1d.weight": "linear_attn.conv1d.weight", "ssm_norm.weight": "linear_attn.norm.weight",
               "ssm_out.weight": "linear_attn.out_proj.weight", "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
               "ssm_beta.weight": "linear_attn.in_proj_b.weight", "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
               "ffn_gate.weight": "mlp.gate_proj.weight", "ffn_up.weight": "mlp.up_proj.weight",
               "ffn_down.weight": "mlp.down_proj.weight", "attn_norm.weight": "input_layernorm.weight",
               "ffn_norm.weight": "post_attention_layernorm.weight", "attn_q.weight": "self_attn.q_proj.weight",
               "attn_k.weight": "self_attn.k_proj.weight", "attn_v.weight": "self_attn.v_proj.weight",
               "attn_output.weight": "self_attn.o_proj.weight", "attn_q_norm.weight": "self_attn.q_norm.weight",
               "attn_k_norm.weight": "self_attn.k_norm.weight"}
    mapping["attn_gate.weight"] = "self_attn.q_proj.weight" if int(layer) % 4 == 3 else "linear_attn.in_proj_z.weight"
    return f"model.language_model.layers.{layer}.{mapping[suffix]}"


def source_shape(name: str, shape: tuple) -> tuple:
    if name.endswith("ssm_conv1d.weight"):
        return (shape[0], 1, shape[1])
    if name.startswith("blk.") and int(name.split(".")[1]) % 4 == 3 and name.endswith(("attn_q.weight", "attn_gate.weight")):
        return (shape[0] * 2, shape[1])
    return shape


def transform(data: np.ndarray, name: str) -> np.ndarray:
    if name.endswith("ssm_a"):
        # Float64 exp rounded to FP32 avoids NumPy's lower-accuracy FP32 SIMD
        # exp. Torch/MKL reference differs by at most 1 ULP for 12/288 values
        # in this checkpoint; do not hide that difference in the plan.
        return (-np.exp(data.astype(np.float64))).astype("<f4")
    if name.endswith("ssm_conv1d.weight"):
        return data[:, 0, :]
    if (name == "output_norm.weight" or name.endswith(("attn_norm.weight", "ffn_norm.weight",
                                                       "attn_q_norm.weight", "attn_k_norm.weight"))):
        return data.astype("<f4") + np.float32(1)
    if name.startswith("blk.") and int(name.split(".")[1]) % 4 == 3:
        if name.endswith(("attn_q.weight", "attn_gate.weight")):
            start = 0 if name.endswith("attn_q.weight") else 256
            return data.reshape(8, 512, 1024)[:, start:start + 256, :].reshape(2048, 1024)
    # Qwen's Q/K rows retain their source order; Llama's permutation does
    # not apply to this graph with partial, interleaved mRoPE.
    return data


def validate_reference(metadata: dict, p: dict):
    required = {"general.architecture": b"custom", "rkllm.model_name": b"Qwen3_5ForConditionalGeneration",
                "custom.block_count": 24, "custom.n_deepstack_layers": 0,
                "custom.rope.scaling.type": b"mrope", "custom.rope.dimension_sections": [11, 11, 10, 0],
                "custom.rope.dimension_count": 64, "custom.ssm.conv_kernel": 4,
                "custom.ssm.state_size": 128, "custom.ssm.group_count": 16,
                "custom.ssm.time_step_rank": 16, "custom.ssm.inner_size": 2048,
                "custom.full_attention_interval": 4}
    fields = {"custom.vocab_size": "vocab_size", "custom.context_length": "max_position_embeddings",
              "custom.embedding_length": "hidden_size", "custom.feed_forward_length": "intermediate_size",
              "custom.attention.head_count": "num_attention_heads", "custom.attention.head_count_kv": "num_key_value_heads",
              "custom.attention.key_length": "head_dim", "custom.attention.value_length": "head_dim",
              "custom.rope.freq_base": "rope_theta", "custom.attention.layer_norm_rms_epsilon": "rms_norm_eps"}
    for key, field in fields.items():
        value = p[field]
        required[key] = float(np.float32(value)) if isinstance(value, float) else value
    vision = p["source_config"]["vision_config"]
    required.update({"rkllm.vision.spatial_merge_size": vision["spatial_merge_size"],
                     "rkllm.vision.spatial_patch_size": vision["patch_size"]})
    for key, value in required.items():
        if metadata.get(key) != value:
            raise ValueError(f"Qwen reference configuration mismatch: {key}")


def corrected_tokenizer_pre(directory: Path, reference_pre: bytes) -> bytes:
    """Use Runtime's single-digit Qwen pre-tokenizer for this source regex.

    Toolkit 1.3.1 emits 'default', which groups numbers differently: the
    source model says 42 for 17+25, while Runtime text input says 32. Supplying
    the source token IDs restores 42. The qwen2 metadata fix is validated
    separately with ordinary text on RK3588.
    """
    tokenizer = json.loads((directory / "tokenizer.json").read_text())
    pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    expected = {"type": "Sequence", "pretokenizers": [
        {"type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated", "invert": False},
        {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}]}
    if tokenizer.get("pre_tokenizer") != expected or reference_pre not in (b"default", b"qwen2"):
        raise ValueError("Unrecognized Qwen pre-tokenizer; cannot apply the observed Runtime correction")
    return b"qwen2"
