from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

try:
    from .runtime import llm_python, vision_python
except ImportError:
    from runtime import llm_python, vision_python


ROOT_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = Path(__file__).resolve().parent / "runs"
VISION_DEMO_DIR = ROOT_DIR / "examples/multimodal_model_demo"
VISION_EXPORT_SCRIPT = VISION_DEMO_DIR / "export/export_vision.py"
VISION_RKNN_SCRIPT = VISION_DEMO_DIR / "export/export_vision_rknn.py"
VISION_CALIBRATION_SCRIPT = VISION_DEMO_DIR / "data/make_input_embeds_for_quantize.py"
SUPPORTED_VISION_MODELS = {
    "qwen2_5-vl-3b": "qwen2.5vl",
    "qwen3-vl": "qwen3vl",
    "qwen3.5": "qwen3.5",
}


def _required_outputs(config: dict) -> list[Path]:
    paths = [Path(config["output_path"])]
    if config.get("conversion_mode") == "vision":
        paths.extend(
            [Path(config["vision_onnx_path"]), Path(config["vision_rknn_path"])]
        )
    return paths


def _verify_outputs(config: dict) -> None:
    missing = [str(path) for path in _required_outputs(config) if not path.is_file()]
    if missing:
        raise FileNotFoundError("以下输出文件没有生成：" + "、".join(missing))


def _run_command(command: list[str], cwd: Path) -> None:
    print("$ " + shlex.join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in iter(process.stdout.readline, ""):
        if line:
            print(line.rstrip(), flush=True)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"外部工具执行失败，退出码：{return_code}。")


def _copy_artifact(source: Path, target: Path, label: str) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"{label}没有生成：{source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    print(f"已保存{label}：{target}", flush=True)


def _build_rkllm(config: dict, dataset: Path) -> None:
    model_path = Path(config["model_path"])
    output_path = Path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("正在加载 RKLLM 工具包……", flush=True)
    from rkllm.api import RKLLM

    llm = RKLLM()
    print(f"正在加载模型：{model_path}", flush=True)
    ret = llm.load_huggingface(
        model=str(model_path),
        model_lora=None,
        device=config["device"],
        dtype=config["dtype"],
        custom_config=None,
        load_weight=True,
    )
    if ret != 0:
        raise RuntimeError(f"模型加载失败，返回码：{ret}。")

    print("正在构建 RKLLM 模型……", flush=True)
    ret = llm.build(
        do_quantization=True,
        optimization_level=config["optimization_level"],
        quantized_dtype=config["quantized_dtype"],
        quantized_algorithm=config["quantized_algorithm"],
        target_platform=config["target_platform"],
        num_npu_core=config["num_npu_core"],
        extra_qparams=None,
        dataset=str(dataset),
        hybrid_rate=config["hybrid_rate"],
        max_context=config["max_context"],
    )
    if ret != 0:
        raise RuntimeError(f"模型构建失败，返回码：{ret}。")

    print(f"正在导出：{output_path}", flush=True)
    ret = llm.export_rkllm(str(output_path))
    if ret != 0:
        raise RuntimeError(f"模型导出失败，返回码：{ret}。")
    print("转换完成。", flush=True)


def _run_text(config: dict) -> None:
    model_path = Path(config["model_path"])
    dataset = Path(config["calibration_path"])
    if not model_path.is_dir():
        raise FileNotFoundError(f"模型目录不存在：{model_path}")
    if not dataset.is_file():
        raise FileNotFoundError(f"校准数据 JSON 不存在：{dataset}")
    _build_rkllm(config, dataset)


def _prepare_visual_workspace() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    work_dir = RUNS_DIR / "visual" / stamp
    data_dir = work_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(VISION_DEMO_DIR / "data/datasets.json", data_dir / "datasets.json")
    dataset_source = VISION_DEMO_DIR / "data/datasets"
    dataset_target = data_dir / "datasets"
    try:
        dataset_target.symlink_to(dataset_source, target_is_directory=True)
    except OSError:
        shutil.copytree(dataset_source, dataset_target)
    return work_dir


def _check_visual_dependencies(python: Path) -> None:
    if not python.is_file():
        raise RuntimeError(f"未找到视觉转换环境：{python}。请重新运行安装器。")
    probe = subprocess.run(
        [
            str(python),
            "-c",
            "import importlib\n"
            "for name in ('rknn.api', 'onnx', 'numpy'):\n"
            " try:\n"
            "  importlib.import_module(name)\n"
            " except ModuleNotFoundError:\n"
            "  print('缺失：' + name)\n"
            "  raise SystemExit(1)\n"
            " except Exception as exc:\n"
            "  print('加载失败：' + name + '：' + str(exc))\n"
            "  raise SystemExit(1)\n",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if probe.returncode != 0:
        missing_names = probe.stdout.strip() or "未知依赖"
        raise RuntimeError(
            f"视觉转换环境不可用：{missing_names}。请重新运行安装器。"
        )


def _run_visual(config: dict) -> None:
    llm_runtime = llm_python()
    vision_runtime = vision_python()
    if not llm_runtime.is_file():
        raise RuntimeError(f"未找到 LLM 转换环境：{llm_runtime}。请重新运行安装器。")
    _check_visual_dependencies(vision_runtime)
    model_path = Path(config["model_path"])
    if not model_path.is_dir():
        raise FileNotFoundError(f"模型目录不存在：{model_path}")

    model_name = config["vision_model_name"]
    if model_name not in SUPPORTED_VISION_MODELS:
        raise ValueError(f"不支持的视觉模型类型：{model_name}")
    if config["vision_model_type"] != SUPPORTED_VISION_MODELS[model_name]:
        raise ValueError("视觉模型类型与校准模型类型不匹配。")

    work_dir = _prepare_visual_workspace()
    try:
        onnx_name = f"{model_name}_vision.onnx"
        generated_onnx = work_dir / "onnx" / onnx_name
        generated_rknn = work_dir / "rknn" / f"{Path(onnx_name).stem}_{config['target_platform']}.rknn"

        print("开始导出视觉 ONNX 模型……", flush=True)
        _run_command(
            [
                str(llm_runtime),
                str(VISION_EXPORT_SCRIPT),
                "--path",
                str(model_path),
                "--model_name",
                model_name,
                "--batch_size",
                str(config["vision_batch_size"]),
                "--height",
                str(config["vision_height"]),
                "--width",
                str(config["vision_width"]),
                "--device",
                config["device"],
            ],
            work_dir,
        )
        _copy_artifact(generated_onnx, Path(config["vision_onnx_path"]), "视觉 ONNX 文件")

        print("开始转换视觉 RKNN 模型……", flush=True)
        _run_command(
            [
                str(vision_runtime),
                str(VISION_RKNN_SCRIPT),
                "--path",
                str(generated_onnx),
                "--model_name",
                model_name,
                "--target-platform",
                config["target_platform"],
                "--batch_size",
                str(config["vision_batch_size"]),
                "--height",
                str(config["vision_height"]),
                "--width",
                str(config["vision_width"]),
            ],
            work_dir,
        )
        _copy_artifact(generated_rknn, Path(config["vision_rknn_path"]), "视觉 RKNN 文件")

        print("正在生成多模态校准数据……", flush=True)
        _run_command(
            [
                str(llm_runtime),
                str(VISION_CALIBRATION_SCRIPT),
                "--path",
                str(model_path),
                "--model_type",
                config["vision_model_type"],
            ],
            work_dir,
        )
        dataset = work_dir / "data/llm_inputs.json"
        if not dataset.is_file():
            raise FileNotFoundError(f"多模态校准数据没有生成：{dataset}")

        print("开始导出视觉模型的 RKLLM 部分……", flush=True)
        _build_rkllm(config, dataset)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        print("已清理视觉转换临时目录。", flush=True)


def run(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("配置文件必须是 JSON 对象。")
    mode = config.get("conversion_mode", "text")
    if mode == "vision":
        _run_visual(config)
    elif mode == "text":
        _run_text(config)
    else:
        raise ValueError(f"不支持的转换模式：{mode}")
    _verify_outputs(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="RKLLM 模型转换工作进程")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.config)
    except Exception as exc:
        print(f"错误：{exc}", flush=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
