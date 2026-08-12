from __future__ import annotations

import importlib.util
import json
import math
import os
import signal
import shlex
import subprocess
import sys
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path

import gradio as gr

warnings.filterwarnings(
    "ignore",
    message="The 'head' parameter in the Blocks constructor will be removed in Gradio 6.0.*",
    category=DeprecationWarning,
)

try:
    from .runtime import llm_python, platform_label, vision_python
except ImportError:
    from runtime import llm_python, platform_label, vision_python


GUI_DIR = Path(__file__).resolve().parent
ROOT_DIR = GUI_DIR.parent
RUNS_DIR = GUI_DIR / "runs"
WORKER_PATH = GUI_DIR / "worker.py"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
STYLE_TAG = f"<style>{(GUI_DIR / 'style.css').read_text(encoding='utf-8')}</style>"

_process: subprocess.Popen[str] | None = None
_process_lock = threading.Lock()

DEFAULT_PRESET = "RK3588"
DEFAULT_QUANTIZATION = "W8A8"
PRESETS = {
    "RK3588": {
        "target_platform": "rk3588",
        "num_npu_core": 3,
        "optimization_level": 1,
        "max_context": 4096,
        "hybrid_rate": 0.0,
        "default_quantization": "W8A8",
    },
    "RK3576": {
        "target_platform": "rk3576",
        "num_npu_core": 2,
        "optimization_level": 1,
        "max_context": 4096,
        "hybrid_rate": 0.0,
        "default_quantization": "W4A16",
    },
}

QUANTIZATIONS = {
    "W8A8": {"quantized_dtype": "w8a8", "quantized_algorithm": "normal"},
    "W8A8 / G128": {"quantized_dtype": "w8a8_g128", "quantized_algorithm": "normal"},
    "W4A16": {"quantized_dtype": "w4a16", "quantized_algorithm": "grq"},
    "W4A16 / G128": {"quantized_dtype": "w4a16_g128", "quantized_algorithm": "grq"},
}

VISION_MODELS = {
    "Qwen2.5-VL": {
        "script_name": "qwen2_5-vl-3b",
        "model_type": "qwen2.5vl",
        "height": 448,
        "width": 448,
        "size_multiple": 28,
    },
    "Qwen3-VL": {
        "script_name": "qwen3-vl",
        "model_type": "qwen3vl",
        "height": 448,
        "width": 448,
        "size_multiple": 32,
    },
    "Qwen3.5": {
        "script_name": "qwen3.5",
        "model_type": "qwen3.5",
        "height": 448,
        "width": 448,
        "size_multiple": 32,
    },
}


def _combined_preset(platform_name: str, quantization_name: str) -> dict:
    preset = dict(PRESETS[platform_name])
    preset.pop("default_quantization", None)
    preset.update(QUANTIZATIONS[quantization_name])
    # optimization_level=1 (质量优先) silently falls back to W8A8 for W4A16
    # quantization, so force 0 to make the selected quantization take effect.
    if preset["quantized_dtype"].startswith("w4a16"):
        preset["optimization_level"] = 0
    return preset


def _cuda_available() -> bool:
    python = llm_python()
    if not python.is_file():
        return False
    try:
        probe = subprocess.run(
            [str(python), "-c", "import torch; print(int(torch.cuda.is_available()))"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0 and probe.stdout.strip() == "1"


def _preset_summary(config: dict) -> str:
    return (
        f"平台：{config['target_platform'].upper()}  |  "
        f"量化：{config['quantized_dtype'].upper()}  |  "
        f"NPU 核心：{config['num_npu_core']}  |  "
        f"优化等级：{config['optimization_level']}"
    )


def apply_preset(name: str) -> tuple[dict, str, int, str]:
    quantization = PRESETS[name]["default_quantization"]
    preset = _combined_preset(name, quantization)
    return preset, _preset_summary(preset), preset["max_context"], quantization


def apply_quantization(name: str, platform_name: str) -> tuple[dict, str]:
    preset = _combined_preset(platform_name, name)
    return preset, _preset_summary(preset)


def apply_vision_model(name: str) -> tuple[int, int]:
    model = VISION_MODELS[name]
    return model["height"], model["width"]


def toggle_conversion_mode(mode: str):
    is_vision = mode == "视觉模型"
    return gr.update(visible=is_vision), gr.update(visible=not is_vision)


def _select_model_file(selected: str | list[str] | None) -> str:
    """根据选中的模型文件自动填入其所在目录。"""
    if not selected:
        return ""
    path = Path(selected[0] if isinstance(selected, list) else selected)
    return str(path.parent)


def _text(value: str | None, label: str) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"请填写{label}。")
    return str(value).strip()


def _integer(value: float | None, label: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须是整数。") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{label}必须是整数。")
    return int(number)


def _validate_inputs(
    model_path: str,
    calibration_path: str,
    output_path: str,
    preset_config: dict,
    max_context: float,
    conversion_mode: str,
    vision_model: str,
    vision_onnx_path: str,
    vision_rknn_path: str,
    vision_batch_size: float,
    vision_height: float,
    vision_width: float,
) -> tuple[Path, Path, Path, dict, int]:
    model = Path(_text(model_path, "模型目录")).expanduser()
    calibration_text = str(calibration_path or "").strip()
    calibration = Path(calibration_text).expanduser() if calibration_text else None
    output = Path(_text(output_path, "输出 RKLLM 文件路径")).expanduser()
    preset = dict(
        preset_config
        or _combined_preset(DEFAULT_PRESET, DEFAULT_QUANTIZATION)
    )

    if not model.is_dir():
        raise ValueError("模型目录不存在。")
    if conversion_mode == "文本模型" and (calibration is None or not calibration.is_file()):
        raise ValueError("校准数据 JSON 不存在。")
    if not output.name:
        raise ValueError("请填写输出文件路径。")

    if conversion_mode not in {"文本模型", "视觉模型"}:
        raise ValueError("不支持的转换类型。")
    if conversion_mode == "视觉模型":
        if vision_model not in VISION_MODELS:
            raise ValueError("请选择视觉模型类型。")
        _text(vision_onnx_path, "视觉 ONNX 输出路径")
        _text(vision_rknn_path, "视觉 RKNN 输出路径")
        batch_size = _integer(vision_batch_size, "图像批次")
        height = _integer(vision_height, "图像高度")
        width = _integer(vision_width, "图像宽度")
        if batch_size < 1 or height < 1 or width < 1:
            raise ValueError("图像批次、高度和宽度必须大于 0。")
        size_multiple = VISION_MODELS[vision_model]["size_multiple"]
        if height % size_multiple or width % size_multiple:
            raise ValueError(
                f"{vision_model} 的图像高度和宽度必须是 {size_multiple} 的倍数。"
            )

    context = _integer(max_context, "最大上下文长度")
    if context < 32 or context > 16384 or context % 32:
        raise ValueError("最大上下文长度必须在 32 到 16384 之间，并且是 32 的倍数。")

    return model, calibration, output, preset, context


def _make_config(
    model_path: str,
    calibration_path: str,
    output_path: str,
    preset_config: dict,
    device: str,
    dtype: str,
    max_context: float,
    conversion_mode: str,
    vision_model: str,
    vision_onnx_path: str,
    vision_rknn_path: str,
    vision_batch_size: float,
    vision_height: float,
    vision_width: float,
) -> tuple[dict, Path]:
    model, calibration, output, preset, context = _validate_inputs(
        model_path,
        calibration_path,
        output_path,
        preset_config,
        max_context,
        conversion_mode,
        vision_model,
        vision_onnx_path,
        vision_rknn_path,
        vision_batch_size,
        vision_height,
        vision_width,
    )
    if device not in {"cpu", "cuda"}:
        raise ValueError("运行设备必须是 cpu 或 cuda。")
    if device == "cuda" and not CUDA_AVAILABLE:
        raise ValueError("当前环境未检测到可用的 CUDA，请选择 CPU。")
    if dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("不支持的权重类型。")

    output = output.resolve()
    config = {
        "model_path": str(model.resolve()),
        "calibration_path": str(calibration.resolve()) if calibration else "",
        "output_path": str(output),
        "device": device,
        "dtype": dtype,
        **preset,
        "max_context": context,
        "conversion_mode": "vision" if conversion_mode == "视觉模型" else "text",
    }
    if conversion_mode == "视觉模型":
        vision = VISION_MODELS[vision_model]
        vision_onnx = Path(_text(vision_onnx_path, "视觉 ONNX 输出路径")).expanduser().resolve()
        vision_rknn = Path(_text(vision_rknn_path, "视觉 RKNN 输出路径")).expanduser().resolve()
        if len({output, vision_onnx, vision_rknn}) != 3:
            raise ValueError("RKLLM、ONNX 和 RKNN 必须使用三个不同的输出路径。")
        config.update(
            {
                "vision_model_name": vision["script_name"],
                "vision_model_type": vision["model_type"],
                "vision_onnx_path": str(vision_onnx),
                "vision_rknn_path": str(vision_rknn),
                "vision_batch_size": _integer(vision_batch_size, "图像批次"),
                "vision_height": _integer(vision_height, "图像高度"),
                "vision_width": _integer(vision_width, "图像宽度"),
            }
        )
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    config_path = RUNS_DIR / f"{stamp}.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config, config_path


def _format_log(lines: list[str]) -> str:
    return "\n".join(lines)


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    """终止 worker 及其启动的全部转换子进程。"""
    process_group = process.pid
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        try:
            process.wait(timeout=1)
        except (subprocess.TimeoutExpired, ChildProcessError):
            pass
        return

    def force_kill() -> None:
        try:
            process.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process_group, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except (subprocess.TimeoutExpired, ChildProcessError):
                pass
        except ChildProcessError:
            pass

    threading.Thread(target=force_kill, daemon=True).start()


def start_conversion(
    model_path: str,
    calibration_path: str,
    output_path: str,
    preset_config: dict,
    device: str,
    dtype: str,
    max_context: float,
    conversion_mode: str,
    vision_model: str,
    vision_onnx_path: str,
    vision_rknn_path: str,
    vision_batch_size: float,
    vision_height: float,
    vision_width: float,
):
    global _process
    lines: list[str] = []
    process: subprocess.Popen[str] | None = None
    runtime_python = llm_python()
    if not runtime_python.is_file():
        yield "运行环境无效", f"未找到 Python 运行环境：{runtime_python}", ""
        return
    try:
        config, config_path = _make_config(
            model_path,
            calibration_path,
            output_path,
            preset_config,
            device,
            dtype,
            max_context,
            conversion_mode,
            vision_model,
            vision_onnx_path,
            vision_rknn_path,
            vision_batch_size,
            vision_height,
            vision_width,
        )
    except (KeyError, TypeError, ValueError) as exc:
        yield "输入无效", str(exc), ""
        return

    output_paths = [Path(config["output_path"])]
    if config["conversion_mode"] == "vision":
        output_paths.extend(
            [Path(config["vision_onnx_path"]), Path(config["vision_rknn_path"])]
        )
    existing_outputs = [str(path) for path in output_paths if path.exists()]
    if existing_outputs:
        lines.append("以下已有输出文件将被覆盖：")
        lines.extend(f"- {path}" for path in existing_outputs)

    try:
        task_running = False
        with _process_lock:
            if _process is not None and _process.poll() is None:
                task_running = True
            else:
                command = [str(runtime_python), str(WORKER_PATH), "--config", str(config_path)]
                lines.append("$ " + shlex.join(command))
                process = subprocess.Popen(
                    command,
                    cwd=ROOT_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
                _process = process

        if task_running:
            yield "任务进行中", "已有转换任务正在运行。", ""
            return

        yield "转换中", _format_log(lines), ""
        assert process is not None
        assert process.stdout is not None
        while True:
            line = process.stdout.readline()
            if line:
                lines.append(line.rstrip())
                yield "转换中", _format_log(lines), ""
                continue
            if process.poll() is not None:
                break
            time.sleep(0.2)

        return_code = process.wait()
        output = config["output_path"]
        status = "已完成" if return_code == 0 else f"失败（退出码 {return_code}）"
        if return_code == 0 and not Path(output).is_file():
            status = "失败"
            lines.append("转换程序已退出，但没有找到输出文件。")
        else:
            lines.append(f"配置文件：{config_path}")
        yield status, _format_log(lines), output if return_code == 0 and status == "已完成" else ""
    except (OSError, subprocess.SubprocessError) as exc:
        lines.append(f"启动或管理转换进程失败：{exc}")
        yield "失败", _format_log(lines), ""
    finally:
        if process is not None and process.poll() is None:
            _kill_process_group(process)
        with _process_lock:
            if _process is process:
                _process = None


def stop_conversion() -> tuple[str, str]:
    with _process_lock:
        process = _process
    if process is None or process.poll() is not None:
        return "空闲", "当前没有正在运行的转换任务。"
    _kill_process_group(process)
    return "正在停止", "已请求停止转换。"


def _probe_environment(python: Path, code: str) -> list[str]:
    if not python.is_file():
        return [f"环境缺失：{python}"]
    try:
        probe = subprocess.run(
            [str(python), "-c", code],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        return ["环境检查超时"]
    if probe.returncode == 0:
        return [line for line in probe.stdout.strip().splitlines() if line]
    details = probe.stderr.strip().splitlines()
    return ["环境加载失败", details[-1] if details else "没有更多错误信息"]


def check_environment() -> str:
    llm_runtime = llm_python()
    vision_runtime = vision_python()
    rows = [f"主机：{platform_label()}"]
    rows.append(f"界面 Python：{sys.version.split()[0]}")
    rows.append(f"Gradio：{'可用' if importlib.util.find_spec('gradio') else '缺失'}")
    rows.append(f"语言环境（LLM）：{llm_runtime}")
    llm_probe = (
        "import importlib.metadata as m; "
        "import importlib.util as u; "
        "import torch; "
        "version = lambda n: m.version(n) if u.find_spec(n.replace('-', '_')) else '缺失'; "
        "print('RKLLM 工具包：' + ('可用' if u.find_spec('rkllm') else '缺失')); "
        "print('Torch：' + torch.__version__); "
        "print('Torchvision：' + version('torchvision')); "
        "print('ONNX：' + version('onnx')); "
        "print('CUDA：' + ('可用' if torch.cuda.is_available() else '不可用'))"
    )
    rows.extend(f"  {line}" for line in _probe_environment(llm_runtime, llm_probe))
    rows.append(f"视觉环境（Vision）：{vision_runtime}")
    vision_probe = (
        "import importlib.metadata as m; "
        "import importlib.util as u; "
        "version = lambda n: m.version(n) if u.find_spec(n.replace('-', '_')) else '缺失'; "
        "print('RKNN 工具包：' + ('可用' if u.find_spec('rknn') else '缺失')); "
        "print('Torch：' + version('torch')); "
        "print('ONNX：' + version('onnx'))"
    )
    rows.extend(f"  {line}" for line in _probe_environment(vision_runtime, vision_probe))
    return "\n".join(rows)


CUDA_AVAILABLE = _cuda_available()
DEFAULT_DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"


with gr.Blocks(title="RKLLM 工作台", head=STYLE_TAG, fill_height=True) as demo:
    with gr.Column(elem_classes=["app-shell"]):
        gr.Markdown("# RKLLM 工作台", elem_classes=["app-header"])
        with gr.Row(elem_classes=["workspace"]):
            with gr.Column(scale=3, elem_classes=["panel"]):
                gr.Markdown("## 模型转换")
                conversion_mode = gr.Dropdown(
                    ["文本模型", "视觉模型"],
                    value="文本模型",
                    label="转换类型",
                )
                model_path = gr.Textbox(label="模型目录", placeholder="/模型目录路径")
                model_browser = gr.FileExplorer(
                    root_dir="/",
                    glob="**/*",
                    file_count="single",
                    label="从模型目录中选择文件",
                    height=180,
                )
                model_browser.change(_select_model_file, inputs=model_browser, outputs=model_path)
                preset = gr.Dropdown(list(PRESETS), value=DEFAULT_PRESET, label="目标平台预设")
                quantization = gr.Dropdown(
                    list(QUANTIZATIONS),
                    value=DEFAULT_QUANTIZATION,
                    label="RKLLM 量化方式",
                )
                preset_config = gr.State(_combined_preset(DEFAULT_PRESET, DEFAULT_QUANTIZATION))
                preset_summary = gr.Textbox(
                    label="预设参数",
                    value=_preset_summary(_combined_preset(DEFAULT_PRESET, DEFAULT_QUANTIZATION)),
                    interactive=False,
                )
                context = gr.Number(value=4096, precision=0, label="最大上下文长度")
                preset.change(
                    apply_preset,
                    inputs=preset,
                    outputs=[preset_config, preset_summary, context, quantization],
                )
                quantization.change(
                    apply_quantization,
                    inputs=[quantization, preset],
                    outputs=[preset_config, preset_summary],
                )
                calibration_path = gr.Textbox(
                    label="校准数据 JSON",
                    value="examples/rkllm_api_demo/export/data_quant.json",
                )
                with gr.Group(visible=False) as vision_options:
                    vision_model = gr.Dropdown(
                        list(VISION_MODELS),
                        value="Qwen2.5-VL",
                        label="视觉模型类型",
                    )
                    with gr.Row():
                        vision_batch_size = gr.Number(value=1, precision=0, label="图像批次")
                        vision_height = gr.Number(value=448, precision=0, label="图像高度")
                        vision_width = gr.Number(value=448, precision=0, label="图像宽度")
                    vision_onnx_path = gr.Textbox(
                        label="视觉 ONNX 输出",
                        value="gui/runs/vision/model_vision.onnx",
                    )
                    vision_rknn_path = gr.Textbox(
                        label="视觉 RKNN 输出",
                        value="gui/runs/vision/model_vision.rknn",
                    )
                conversion_mode.change(
                    toggle_conversion_mode,
                    inputs=conversion_mode,
                    outputs=[vision_options, calibration_path],
                )
                vision_model.change(
                    apply_vision_model,
                    inputs=vision_model,
                    outputs=[vision_height, vision_width],
                )
                output_path = gr.Textbox(label="输出 RKLLM 文件", value="gui/runs/model.rkllm")
                with gr.Accordion("高级设置", open=False):
                    with gr.Row():
                        device = gr.Dropdown(
                            ["cuda", "cpu"], value=DEFAULT_DEVICE, label="运行设备"
                        )
                        dtype = gr.Dropdown(
                            ["float32", "float16", "bfloat16"],
                            value="float32",
                            label="权重类型",
                        )
                with gr.Row():
                    check = gr.Button("检查环境")
                    start = gr.Button("开始转换", variant="primary")
                    stop = gr.Button("停止")
                environment = gr.Textbox(label="环境信息", lines=7, interactive=False)
            with gr.Column(scale=4, elem_classes=["panel", "result-panel"]):
                status = gr.Textbox(label="状态", value="空闲", interactive=False, elem_id="status")
                log = gr.Textbox(
                    label="任务日志",
                    lines=22,
                    max_lines=22,
                    autoscroll=False,
                    show_copy_button=True,
                    interactive=False,
                    elem_id="task-log",
                )
                result = gr.Textbox(label="输出文件", interactive=False)

    check.click(check_environment, outputs=environment)
    start.click(
        start_conversion,
        inputs=[
            model_path,
            calibration_path,
            output_path,
            preset_config,
            device,
            dtype,
            context,
            conversion_mode,
            vision_model,
            vision_onnx_path,
            vision_rknn_path,
            vision_batch_size,
            vision_height,
            vision_width,
        ],
        outputs=[status, log, result],
    )
    stop.click(stop_conversion, outputs=[status, log])


if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.queue(max_size=2).launch(server_name="127.0.0.1", server_port=port, share=False)
