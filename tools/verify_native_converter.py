"""Differential verification against the repository's exact Toolkit wheel.

Run with an existing Python 3.10 environment containing the official model
dependencies. The isolated wheel is only used as the reference oracle;
conversion runs in a separate NumPy-only Python process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portable-python", default="python3")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "gui/runs/native-converter-research")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--tokenizer", choices=("bpe",), default="bpe")
    parser.add_argument("--keep-fixtures", type=Path, help="Keep verified model/plan/reference material in a new directory")
    args = parser.parse_args()
    wheel = ROOT / "rkllm-toolkit/packages/rkllm_toolkit-1.3.1-cp310-cp310-linux_x86_64.whl"
    args.work_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(prefix="differential-", dir=args.work_dir) as temp:
        work = Path(temp)
        sdk = work / "sdk"
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(sdk)
        sys.path.insert(0, str(sdk))
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
        from rkllm.api import RKLLM

        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT)

        def portable(*arguments, succeeds=True):
            process = subprocess.run([args.portable_python, "-m", "native_converter", *map(str, arguments)],
                                     cwd=ROOT, env=env, capture_output=True, text=True)
            if (process.returncode == 0) != succeeds:
                raise AssertionError(f"Unexpected CLI result: {arguments}\n{process.stdout}\n{process.stderr}")
            return process

        for hidden, intermediate, layers in [(128, 256, 1), (256, 768, 2)]:
            label = f"h{hidden}-f{intermediate}-l{layers}"
            plan = work / (label + ".rkplan")
            official_files = []
            native_files = []
            model_dirs = []
            for index, seed in enumerate([3407, 20261007]):
                torch.manual_seed(seed)
                vocab_size = 288
                config = LlamaConfig(vocab_size=vocab_size, hidden_size=hidden, intermediate_size=intermediate,
                                     num_hidden_layers=layers, num_attention_heads=hidden // 64,
                                     num_key_value_heads=hidden // 128, max_position_embeddings=128,
                                     tie_word_embeddings=False, bos_token_id=1, eos_token_id=2, pad_token_id=0)
                model = LlamaForCausalLM(config)
                # Vary normalization weights too: an all-ones-only fixture could
                # conceal a converter that fails to read these source tensors.
                with torch.no_grad():
                    for name, value in model.named_parameters():
                        if "norm" in name:
                            value.copy_(1 + torch.randn_like(value) * 0.03)
                directory = work / f"{label}-seed{index}"
                if index == 1:
                    model = model.to(torch.bfloat16 if hidden == 128 else torch.float16)
                source_dtype = str(next(model.parameters()).dtype)
                model.save_pretrained(directory, max_shard_size="300KB")
                backend = Tokenizer(models.BPE(unk_token="[UNK]"))
                backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
                backend.decoder = decoders.ByteLevel()
                trainer = trainers.BpeTrainer(vocab_size=vocab_size,
                                              initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                              special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"])
                corpus = ["hello world. Rockchip native converter model testing. "
                          "The quick brown fox jumps over the lazy dog. "
                          "A model computes attention and feed forward matrix products. "
                          "Testing byte level tokenization for RK3588." for _ in range(20)]
                backend.train_from_iterator(corpus, trainer=trainer)
                assert backend.get_vocab_size() == vocab_size
                tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="[BOS]",
                                                    eos_token="[EOS]", unk_token="[UNK]", pad_token="[PAD]")
                tokenizer.save_pretrained(directory)
                official = work / f"{label}-seed{index}-official.rkllm"
                native = work / f"{label}-seed{index}-native.rkllm"
                sdk_model = RKLLM()
                assert sdk_model.load_huggingface(str(directory), device="cpu", dtype="float32") == 0
                assert sdk_model.build(do_quantization=False, optimization_level=0,
                                       quantized_dtype="w8a8", quantized_algorithm="normal",
                                       target_platform="rk3588", num_npu_core=1, max_context=128) == 0
                assert sdk_model.export_rkllm(str(official)) == 0
                if index == 0:
                    portable("prepare", "--reference", official, "--model", directory, "--output", plan)
                portable("convert", "--plan", plan, "--model", directory, "--output", native)
                official_hash = hashlib.sha256(official.read_bytes()).hexdigest()
                native_hash = hashlib.sha256(native.read_bytes()).hexdigest()
                assert official_hash == native_hash, f"Binary mismatch for {label} seed {seed}"
                results.append({"configuration": label, "seed": seed, "bytes": native.stat().st_size,
                                "source_dtype": source_dtype,
                                "source_shards": len(list(directory.glob("*.safetensors"))),
                                "sha256": native_hash, "official_byte_identical": True})
                official_files.append(official)
                native_files.append(native)
                model_dirs.append(directory)
                if args.keep_fixtures:
                    destination = args.keep_fixtures / f"{label}-seed{seed}"
                    destination.mkdir(parents=True, exist_ok=False)
                    shutil.copytree(directory, destination / "model")
                    for source, name in [(plan, "model.rkplan"), (official, "official.rkllm"),
                                         (native, "native.rkllm")]:
                        shutil.copyfile(source, destination / name)
                print(f"PASS: {label}, seed={seed}, {native.stat().st_size} bytes", flush=True)
                del model, sdk_model
            assert native_files[0].read_bytes() != native_files[1].read_bytes(), "Different weights must produce different models"
            # Existing output must survive; rejected configurations must not leave output files.
            before = native_files[1].read_bytes()
            portable("convert", "--plan", plan, "--model", model_dirs[1], "--output", native_files[1], succeeds=False)
            assert native_files[1].read_bytes() == before
            config_path = model_dirs[1] / "config.json"
            original = config_path.read_text()
            changed = json.loads(original)
            changed["rms_norm_eps"] = 1e-5
            config_path.write_text(json.dumps(changed))
            rejected = work / (label + "-rejected.rkllm")
            portable("convert", "--plan", plan, "--model", model_dirs[1], "--output", rejected, succeeds=False)
            assert not rejected.exists()
            config_path.write_text(original)
        # Malformed containers must fail cleanly, rather than access arbitrary offsets.
        truncated = work / "truncated.rkllm"
        truncated.write_bytes(b"\xda\xee\xb3\x36")
        portable("inspect", truncated, succeeds=False)
        # Reproduce the missing metadata observed in the original WordLevel
        # export: it was byte-identical but could not load in Runtime 1.3.1.
        with zipfile.ZipFile(plan) as archive:
            incomplete_manifest = json.loads(archive.read("manifest.json"))
            program = archive.read("program.bin")
        incomplete_manifest["entries"] = [entry for entry in incomplete_manifest["entries"]
                                          if entry["key"] != "tokenizer.ggml.merges"]
        incomplete_plan = work / "missing-merges.rkplan"
        with zipfile.ZipFile(incomplete_plan, "x") as archive:
            archive.writestr("manifest.json", json.dumps(incomplete_manifest))
            archive.writestr("program.bin", program)
        invalid_output = work / "missing-merges.rkllm"
        portable("convert", "--plan", incomplete_plan, "--model", model_dirs[-1],
                 "--output", invalid_output, succeeds=False)
        assert not invalid_output.exists()
        report = {"toolkit": "repository 1.3.1 wheel", "portable_python": args.portable_python,
                  "tokenizer": args.tokenizer,
                  "hardware_validation_in_this_run": False,
                  "results": results, "checks": ["all four outputs byte-identical to official",
                  "different source weights produce different outputs", "existing output preserved",
                  "changed normalization configuration rejected", "truncated container rejected",
                  "missing BPE merges rejected"],
                  "limitations": ["NPU programs reused from configuration-matching reference",
                                  "this differential run does not exercise ARM64 or Runtime; see separate board report"]}
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
