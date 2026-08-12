from __future__ import annotations

import importlib.util
import json
import math
import mimetypes
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
import shutil
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

try:
    from .runtime import llm_python, platform_label, vision_python
except ImportError:
    from runtime import llm_python, platform_label, vision_python


GUI_DIR = Path(__file__).resolve().parent
ROOT_DIR = GUI_DIR.parent
RUNS_DIR = GUI_DIR / "runs"
WORKER_PATH = GUI_DIR / "worker.py"
INDEX_PATH = GUI_DIR / "index.html"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PRESET = "RK3588"
DEFAULT_QUANTIZATION = "W8A8"
PRESETS = {
    "RK3588": {"target_platform": "rk3588", "num_npu_core": 3, "optimization_level": 1, "max_context": 4096, "hybrid_rate": 0.0, "default_quantization": "W8A8"},
    "RK3576": {"target_platform": "rk3576", "num_npu_core": 2, "optimization_level": 1, "max_context": 4096, "hybrid_rate": 0.0, "default_quantization": "W4A16"},
}
QUANTIZATIONS = {
    "W8A8": {"quantized_dtype": "w8a8", "quantized_algorithm": "normal"},
    "W8A8 / G128": {"quantized_dtype": "w8a8_g128", "quantized_algorithm": "normal"},
    "W4A16": {"quantized_dtype": "w4a16", "quantized_algorithm": "grq"},
    "W4A16 / G128": {"quantized_dtype": "w4a16_g128", "quantized_algorithm": "grq"},
}
VISION_MODELS = {
    "Qwen2.5-VL": {"script_name": "qwen2_5-vl-3b", "model_type": "qwen2.5vl", "height": 448, "width": 448, "size_multiple": 28},
    "Qwen3-VL": {"script_name": "qwen3-vl", "model_type": "qwen3vl", "height": 448, "width": 448, "size_multiple": 32},
    "Qwen3.5": {"script_name": "qwen3.5", "model_type": "qwen3.5", "height": 448, "width": 448, "size_multiple": 32},
}

_job_lock = threading.Lock()
_job: dict | None = None


def combined_preset(platform_name: str, quantization_name: str) -> dict:
    if platform_name not in PRESETS or quantization_name not in QUANTIZATIONS:
        raise ValueError("平台或量化方式不支持。")
    preset = dict(PRESETS[platform_name])
    preset.pop("default_quantization", None)
    preset.update(QUANTIZATIONS[quantization_name])
    if preset["quantized_dtype"].startswith("w4a16"):
        preset["optimization_level"] = 0
    return preset


def preset_summary(config: dict) -> str:
    return f"{config['target_platform'].upper()}  ·  {config['quantized_dtype'].upper()}  ·  {config['num_npu_core']} 个 NPU 核心  ·  优化等级 {config['optimization_level']}"


def cuda_available() -> bool:
    python = llm_python()
    if not python.is_file():
        return False
    try:
        probe = subprocess.run([str(python), "-c", "import torch; print(int(torch.cuda.is_available()))"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0 and probe.stdout.strip() == "1"


CUDA_AVAILABLE = cuda_available()
DEFAULT_DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"


def text_value(value, label: str) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"请填写{label}。")
    return str(value).strip()


def integer_value(value, label: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须是整数。") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{label}必须是整数。")
    return int(number)


def make_config(data: dict) -> tuple[dict, Path]:
    mode = data.get("conversion_mode", "文本模型")
    model = Path(text_value(data.get("model_path"), "模型目录")).expanduser()
    output = Path(text_value(data.get("output_path"), "输出 RKLLM 文件路径")).expanduser().resolve()
    calibration_text = str(data.get("calibration_path") or "").strip()
    calibration = Path(calibration_text).expanduser() if calibration_text else None
    if mode not in {"文本模型", "视觉模型"}:
        raise ValueError("不支持的转换类型。")
    if not model.is_dir():
        raise ValueError("模型目录不存在。")
    if mode == "文本模型" and (calibration is None or not calibration.is_file()):
        raise ValueError("校准数据 JSON 不存在。")

    platform_name = data.get("platform", DEFAULT_PRESET)
    quantization = data.get("quantization", DEFAULT_QUANTIZATION)
    preset = combined_preset(platform_name, quantization)
    context = integer_value(data.get("max_context", 4096), "最大上下文长度")
    if context < 32 or context > 16384 or context % 32:
        raise ValueError("最大上下文长度必须在 32 到 16384 之间，并且是 32 的倍数。")
    device = data.get("device", DEFAULT_DEVICE)
    dtype = data.get("dtype", "float32")
    if device not in {"cpu", "cuda"}:
        raise ValueError("运行设备必须是 CPU 或 CUDA。")
    if device == "cuda" and not CUDA_AVAILABLE:
        raise ValueError("当前环境未检测到可用的 CUDA，请选择 CPU。")
    if dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("不支持的权重类型。")

    config = {"model_path": str(model.resolve()), "calibration_path": str(calibration.resolve()) if calibration else "", "output_path": str(output), "device": device, "dtype": dtype, **preset, "max_context": context, "conversion_mode": "vision" if mode == "视觉模型" else "text"}
    if mode == "视觉模型":
        model_name = data.get("vision_model")
        if model_name not in VISION_MODELS:
            raise ValueError("请选择视觉模型类型。")
        vision = VISION_MODELS[model_name]
        batch = integer_value(data.get("vision_batch_size"), "图像批次")
        height = integer_value(data.get("vision_height"), "图像高度")
        width = integer_value(data.get("vision_width"), "图像宽度")
        if min(batch, height, width) < 1:
            raise ValueError("图像批次、高度和宽度必须大于 0。")
        if height % vision["size_multiple"] or width % vision["size_multiple"]:
            raise ValueError(f"{model_name} 的图像高度和宽度必须是 {vision['size_multiple']} 的倍数。")
        onnx = Path(text_value(data.get("vision_onnx_path"), "视觉 ONNX 输出路径")).expanduser().resolve()
        rknn = Path(text_value(data.get("vision_rknn_path"), "视觉 RKNN 输出路径")).expanduser().resolve()
        if len({output, onnx, rknn}) != 3:
            raise ValueError("RKLLM、ONNX 和 RKNN 必须使用三个不同的输出路径。")
        config.update({"vision_model_name": vision["script_name"], "vision_model_type": vision["model_type"], "vision_onnx_path": str(onnx), "vision_rknn_path": str(rknn), "vision_batch_size": batch, "vision_height": height, "vision_width": width})

    config_path = RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S-%f}.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config, config_path


def kill_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    def force_kill() -> None:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except (subprocess.TimeoutExpired, ChildProcessError):
                pass

    threading.Thread(target=force_kill, daemon=True).start()


def start_job(data: dict) -> str:
    global _job
    config, config_path = make_config(data)
    outputs = [Path(config["output_path"])]
    if config["conversion_mode"] == "vision":
        outputs += [Path(config["vision_onnx_path"]), Path(config["vision_rknn_path"])]
    lines = []
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        lines.append("以下已有输出文件将被覆盖：")
        lines.extend(f"- {path}" for path in existing)
    with _job_lock:
        if _job and _job["process"].poll() is None:
            raise ValueError("已有转换任务正在运行。")
        command = [str(llm_python()), str(WORKER_PATH), "--config", str(config_path)]
        lines.append("$ " + shlex.join(command))
        process = subprocess.Popen(command, cwd=ROOT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        _job = {"id": uuid4().hex, "process": process, "lines": lines, "config": config, "config_path": str(config_path), "status": "转换中", "output": ""}
        threading.Thread(target=read_job_output, args=(process,), daemon=True).start()
        return _job["id"]


def read_job_output(process: subprocess.Popen[str]) -> None:
    global _job
    try:
        assert process.stdout is not None
        for line in iter(process.stdout.readline, ""):
            if line:
                with _job_lock:
                    if _job and _job["process"] is process:
                        _job["lines"].append(line.rstrip())
        code = process.wait()
        with _job_lock:
            if not _job or _job["process"] is not process:
                return
            config = _job["config"]
            output = config["output_path"]
            if code == 0 and Path(output).is_file():
                _job["status"] = "转换完成"
                _job["output"] = output
                _job["lines"].append(f"配置文件：{_job['config_path']}")
            else:
                _job["status"] = f"失败（退出码 {code}）"
                if code == 0:
                    _job["lines"].append("转换程序已退出，但没有找到输出文件。")
    except Exception as exc:
        with _job_lock:
            if _job and _job["process"] is process:
                _job["status"] = "转换失败"
                _job["lines"].append(f"任务异常：{exc}")


def job_snapshot() -> dict:
    with _job_lock:
        if not _job:
            return {"status": "空闲", "log": "", "output": ""}
        return {"status": _job["status"], "log": "\n".join(_job["lines"]), "output": _job["output"]}


def stop_job() -> dict:
    with _job_lock:
        job = _job
    if not job or job["process"].poll() is not None:
        return {"status": "空闲", "log": "当前没有正在运行的转换任务。", "output": ""}
    kill_process_group(job["process"])
    with _job_lock:
        if _job is job:
            _job["status"] = "正在停止"
            _job["lines"].append("正在停止转换及其子进程……")
    return job_snapshot()


def probe_environment(python: Path, code: str) -> list[str]:
    if not python.is_file():
        return [f"环境缺失：{python}"]
    try:
        result = subprocess.run([str(python), "-c", code], capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired:
        return ["环境检查超时"]
    if result.returncode == 0:
        return [line for line in result.stdout.strip().splitlines() if line]
    details = result.stderr.strip().splitlines()
    return ["环境加载失败", details[-1] if details else "没有更多错误信息"]


def environment_snapshot() -> str:
    llm = llm_python()
    vision = vision_python()
    rows = [f"主机：{platform_label()}", f"界面 Python：{sys.version.split()[0]}", f"Gradio：{'已移除，使用本地 HTML' if not importlib.util.find_spec('gradio') else '兼容环境可用'}", "", f"语言模型环境：{llm}"]
    llm_code = "import importlib.metadata as m; import importlib.util as u; import torch; v=lambda n:m.version(n) if u.find_spec(n.replace('-', '_')) else '缺失'; print('RKLLM：'+('可用' if u.find_spec('rkllm') else '缺失')); print('Torch：'+torch.__version__); print('Torchvision：'+v('torchvision')); print('ONNX：'+v('onnx')); print('CUDA：'+('可用' if torch.cuda.is_available() else '不可用'))"
    rows.extend(f"  {line}" for line in probe_environment(llm, llm_code))
    rows += ["", f"视觉转换环境：{vision}"]
    vision_code = "import importlib.metadata as m; import importlib.util as u; v=lambda n:m.version(n) if u.find_spec(n.replace('-', '_')) else '缺失'; print('RKNN：'+('可用' if u.find_spec('rknn') else '缺失')); print('Torch：'+v('torch')); print('ONNX：'+v('onnx'))"
    rows.extend(f"  {line}" for line in probe_environment(vision, vision_code))
    return "\n".join(rows)


def select_directory(initial_path: str = "") -> str:
    initial = Path(initial_path).expanduser() if initial_path else Path.home()
    if not initial.is_dir():
        initial = initial.parent if initial.parent.is_dir() else Path.home()

    if platform_label() == "WSL2" and shutil.which("powershell.exe"):
        script = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$dialog = New-Object System.Windows.Forms.FolderBrowserDialog; "
            "$dialog.Description = '选择目录'; "
            "if ($dialog.ShowDialog() -eq 'OK') { $dialog.SelectedPath }"
        )
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-STA", "-Command", script],
            capture_output=True,
            text=True,
            timeout=300,
        )
        windows_path = result.stdout.strip()
        if not windows_path:
            return ""
        converted = subprocess.run(
            ["wslpath", "-u", windows_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if converted.returncode != 0:
            raise ValueError("无法将 Windows 路径转换为 WSL 路径。")
        return converted.stdout.strip()

    if shutil.which("zenity"):
        result = subprocess.run(
            [
                "zenity",
                "--file-selection",
                "--directory",
                "--title=选择目录",
                f"--filename={initial.resolve()}/",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode in {0, 1}:
            return result.stdout.strip()
        raise ValueError(result.stderr.strip() or "目录选择器启动失败。")

    if shutil.which("kdialog"):
        result = subprocess.run(
            ["kdialog", "--getexistingdirectory", str(initial.resolve())],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode in {0, 1}:
            return result.stdout.strip()
        raise ValueError(result.stderr.strip() or "目录选择器启动失败。")

    raise ValueError("未找到目录选择器。请安装 Zenity 或 KDialog。")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        return

    def send_bytes(self, content: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, payload: dict, status: int = 200) -> None:
        self.send_bytes(json.dumps(payload, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        if path == "/":
            self.send_bytes(INDEX_PATH.read_bytes(), "text/html; charset=utf-8")
        elif path == "/app.js":
            self.send_bytes((GUI_DIR / "app.js").read_bytes(), "text/javascript; charset=utf-8")
        elif path == "/style.css":
            self.send_bytes((GUI_DIR / "style.css").read_bytes(), "text/css; charset=utf-8")
        elif path == "/api/config":
            self.send_json({"platform": platform_label(), "default_device": DEFAULT_DEVICE, "cuda": CUDA_AVAILABLE, "presets": PRESETS, "quantizations": QUANTIZATIONS, "vision_models": VISION_MODELS})
        elif path == "/api/job":
            self.send_json(job_snapshot())
        else:
            self.send_json({"error": "页面不存在"}, 404)

    def do_POST(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        try:
            if path == "/api/environment":
                self.send_json({"text": environment_snapshot()})
            elif path == "/api/start":
                self.send_json({"job_id": start_job(self.read_json())})
            elif path == "/api/stop":
                self.send_json(stop_job())
            elif path == "/api/select-directory":
                selected = select_directory(self.read_json().get("initial_path", ""))
                self.send_json({"path": selected})
            elif path == "/api/open-folder":
                target = Path(text_value(self.read_json().get("path"), "路径")).expanduser()
                target = target if target.is_dir() else target.parent
                if not target.is_dir():
                    raise ValueError("目录不存在。")
                if sys.platform == "linux":
                    subprocess.Popen(["xdg-open", str(target)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    subprocess.Popen(["explorer.exe", str(target)])
                self.send_json({"ok": True})
            else:
                self.send_json({"error": "接口不存在"}, 404)
        except (ValueError, json.JSONDecodeError, OSError, subprocess.SubprocessError) as exc:
            self.send_json({"error": str(exc)}, 400)


def main() -> None:
    port = int(os.environ.get("GRADIO_SERVER_PORT", os.environ.get("RKLLM_WORKBENCH_PORT", "7860")))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"RKLLM 工作台已启动：http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
