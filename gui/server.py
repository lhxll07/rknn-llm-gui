from __future__ import annotations

import json
import math
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
MAX_JSON_BYTES = 1024 * 1024
MAX_LOG_LINES = 10000
KEEP_CONFIGS = 10
KEEP_VISUAL_WORKSPACES = 3

DEFAULT_PRESET = "RK3588"
PRESETS = {
    "RK3588": {"target_platform": "rk3588", "num_npu_core": 3, "optimization_level": 1, "max_context": 4096, "hybrid_rate": 0.0, "default_quantization": "W8A8"},
    "RK3576": {"target_platform": "rk3576", "num_npu_core": 2, "optimization_level": 1, "max_context": 4096, "hybrid_rate": 0.0, "default_quantization": "W4A16"},
}
DEFAULT_QUANTIZATION = PRESETS[DEFAULT_PRESET]["default_quantization"]
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


def _newest(directory: Path, pattern: str = "*", directories_only: bool = False) -> list[Path]:
    children = directory.glob(pattern)
    if directories_only:
        children = (path for path in children if path.is_dir() and not path.is_symlink())
    return sorted(children, key=lambda path: path.stat().st_mtime, reverse=True)


def _cleanup_files(directory: Path, pattern: str, keep: int) -> None:
    files = [path for path in _newest(directory, pattern) if path.is_file()]
    for path in files[keep:]:
        path.unlink(missing_ok=True)


def _cleanup_directories(directory: Path, keep: int) -> None:
    for path in _newest(directory, directories_only=True)[keep:]:
        shutil.rmtree(path, ignore_errors=True)


def cleanup_runs() -> None:
    _cleanup_files(RUNS_DIR, "*.json", KEEP_CONFIGS)
    visual_dir = RUNS_DIR / "visual"
    if visual_dir.is_dir():
        _cleanup_directories(visual_dir, KEEP_VISUAL_WORKSPACES)


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
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"请填写{label}。")
    if "\x00" in value:
        raise ValueError(f"{label}包含无效字符。")
    return value.strip()


def integer_value(value, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label}必须是整数。")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label}必须是整数。") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{label}必须是整数。")
    return int(number)


def resolve_user_path(value: str, label: str, reject_symlink: bool = False) -> Path:
    try:
        path = Path(text_value(value, label)).expanduser()
    except RuntimeError as exc:
        raise ValueError(f"{label}包含无法解析的用户目录。") from exc
    if not path.is_absolute():
        path = ROOT_DIR / path
    if reject_symlink and path.is_symlink():
        raise ValueError(f"{label}必须是普通文件，不能是符号链接。")
    return path.resolve()


def output_paths(config: dict) -> list[Path]:
    paths = [Path(config["output_path"])]
    if config.get("conversion_mode") == "vision":
        paths.extend(
            [Path(config["vision_onnx_path"]), Path(config["vision_rknn_path"])]
        )
    return paths


def validate_output_paths(paths: list[Path]) -> None:
    invalid = [
        str(path)
        for path in paths
        if path.is_symlink() or (path.exists() and not path.is_file())
    ]
    if invalid:
        raise ValueError("输出路径必须是普通文件：" + "、".join(invalid))

    for index, path in enumerate(paths):
        for other in paths[index + 1 :]:
            try:
                same_file = path.exists() and other.exists() and os.path.samefile(path, other)
            except OSError:
                same_file = False
            if path == other or same_file:
                raise ValueError("RKLLM、ONNX 和 RKNN 必须使用三个不同的文件。")


def validate_output_sources(outputs: list[Path], sources: list[tuple[Path, str]]) -> None:
    for output in outputs:
        for source, label in sources:
            try:
                same_file = output.exists() and source.exists() and os.path.samefile(output, source)
            except OSError:
                same_file = False
            if output == source or same_file:
                raise ValueError(f"输出文件不能覆盖{label}。")


def file_signature(path: Path) -> tuple[int, int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def changed_output_paths(paths: list[Path], before: dict[str, tuple[int, int, int] | None]) -> list[Path]:
    return [path for path in paths if path.is_file() and file_signature(path) != before.get(str(path))]


def make_config(data: dict) -> tuple[dict, Path]:
    if not isinstance(data, dict):
        raise ValueError("请求数据必须是 JSON 对象。")
    mode = data.get("conversion_mode", "文本模型")
    if not isinstance(mode, str):
        raise ValueError("不支持的转换类型。")
    model = resolve_user_path(data.get("model_path"), "模型目录")
    output = resolve_user_path(data.get("output_path"), "输出 RKLLM 文件路径", reject_symlink=True)
    calibration_value = data.get("calibration_path")
    if calibration_value is not None and not isinstance(calibration_value, str):
        raise ValueError("校准数据 JSON 路径无效。")
    calibration_text = (calibration_value or "").strip()
    calibration = resolve_user_path(calibration_text, "校准数据 JSON") if calibration_text else None
    if mode not in {"文本模型", "视觉模型"}:
        raise ValueError("不支持的转换类型。")
    if not model.is_dir():
        raise ValueError("模型目录不存在。")
    if model == output or model in output.parents:
        raise ValueError("输出文件不能放在模型目录内，以免覆盖源模型。")
    if mode == "文本模型" and (calibration is None or not calibration.is_file()):
        raise ValueError("校准数据 JSON 不存在。")

    platform_name = data.get("platform") or DEFAULT_PRESET
    if not isinstance(platform_name, str) or platform_name not in PRESETS:
        raise ValueError("平台或量化方式不支持。")
    quantization = data.get("quantization") or PRESETS[platform_name]["default_quantization"]
    if not isinstance(quantization, str):
        raise ValueError("平台或量化方式不支持。")
    preset = combined_preset(platform_name, quantization)
    context = integer_value(data.get("max_context", 4096), "最大上下文长度")
    if context < 32 or context > 16384 or context % 32:
        raise ValueError("最大上下文长度必须在 32 到 16384 之间，并且是 32 的倍数。")
    device = data.get("device", DEFAULT_DEVICE)
    dtype = data.get("dtype", "float32")
    if not isinstance(device, str) or device not in {"cpu", "cuda"}:
        raise ValueError("运行设备必须是 CPU 或 CUDA。")
    if device == "cuda" and not CUDA_AVAILABLE:
        raise ValueError("当前环境未检测到可用的 CUDA，请选择 CPU。")
    if not isinstance(dtype, str) or dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("不支持的权重类型。")

    config = {"model_path": str(model.resolve()), "calibration_path": str(calibration.resolve()) if calibration else "", "output_path": str(output), "device": device, "dtype": dtype, **preset, "max_context": context, "conversion_mode": "vision" if mode == "视觉模型" else "text"}
    if mode == "视觉模型":
        model_name = data.get("vision_model")
        if not isinstance(model_name, str) or model_name not in VISION_MODELS:
            raise ValueError("请选择视觉模型类型。")
        vision = VISION_MODELS[model_name]
        batch = integer_value(data.get("vision_batch_size"), "图像批次")
        height = integer_value(data.get("vision_height"), "图像高度")
        width = integer_value(data.get("vision_width"), "图像宽度")
        if min(batch, height, width) < 1:
            raise ValueError("图像批次、高度和宽度必须大于 0。")
        if height % vision["size_multiple"] or width % vision["size_multiple"]:
            raise ValueError(f"{model_name} 的图像高度和宽度必须是 {vision['size_multiple']} 的倍数。")
        onnx = resolve_user_path(data.get("vision_onnx_path"), "视觉 ONNX 输出路径", reject_symlink=True)
        rknn = resolve_user_path(data.get("vision_rknn_path"), "视觉 RKNN 输出路径", reject_symlink=True)
        if len({output, onnx, rknn}) != 3:
            raise ValueError("RKLLM、ONNX 和 RKNN 必须使用三个不同的输出路径。")
        if any(model == path or model in path.parents for path in (onnx, rknn)):
            raise ValueError("输出文件不能放在模型目录内，以免覆盖源模型。")
        config.update({"vision_model_name": vision["script_name"], "vision_model_type": vision["model_type"], "vision_onnx_path": str(onnx), "vision_rknn_path": str(rknn), "vision_batch_size": batch, "vision_height": height, "vision_width": width})

    outputs = output_paths(config)
    validate_output_paths(outputs)
    sources = [(model, "模型目录")]
    if calibration is not None:
        sources.append((calibration, "校准数据"))
    validate_output_sources(outputs, sources)
    config_path = RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S-%f}.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config, config_path


def process_group_exists(process: subprocess.Popen[str]) -> bool:
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def kill_process_group(process: subprocess.Popen[str], wait: bool = False) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        if not wait:
            return

    def force_kill() -> None:
        deadline = time.monotonic() + 5
        try:
            while time.monotonic() < deadline:
                if process.poll() is not None and not process_group_exists(process):
                    return
                try:
                    process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
                except (ChildProcessError, OSError):
                    break
        finally:
            try:
                if process_group_exists(process):
                    os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except (subprocess.TimeoutExpired, ChildProcessError):
                pass

    if wait:
        force_kill()
    else:
        threading.Thread(target=force_kill, daemon=True).start()


def job_unfinished(job: dict) -> bool:
    finished = job.get("finished")
    if finished is not None:
        return not finished.is_set()
    return job["process"].poll() is None


def append_job_log(job: dict, text: str) -> None:
    total = job.get("log_total", len(job["lines"]))
    for line in text.splitlines():
        job["lines"].append(line)
        total += 1
        if len(job["lines"]) > MAX_LOG_LINES:
            del job["lines"][:-MAX_LOG_LINES]
    job["log_total"] = total


def start_job(data: dict) -> str:
    global _job
    lines = []
    with _job_lock:
        if _job and job_unfinished(_job):
            raise ValueError("已有转换任务正在运行。")
        config, config_path = make_config(data)
        outputs = output_paths(config)
        before = {str(path): file_signature(path) for path in outputs}
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            lines.append("以下已有输出文件将被覆盖：")
            lines.extend(f"- {path}" for path in existing)
        command = [str(llm_python()), str(WORKER_PATH), "--config", str(config_path)]
        lines.append("$ " + shlex.join(command))
        try:
            process = subprocess.Popen(command, cwd=ROOT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        except Exception:
            config_path.unlink(missing_ok=True)
            raise
        _job = {
            "id": uuid4().hex,
            "process": process,
            "lines": lines,
            "config": config,
            "config_path": str(config_path),
            "status": "转换中",
            "status_code": "running",
            "stop_requested": False,
            "outputs_before": before,
            "output": "",
            "outputs": [],
            "finished": threading.Event(),
        }
        append_job_log(_job, "\n".join(lines))
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
                        append_job_log(_job, line.rstrip())
        code = process.wait()
        with _job_lock:
            if not _job or _job["process"] is not process:
                return
            config = _job["config"]
            required_outputs = output_paths(config)
            generated_outputs = changed_output_paths(required_outputs, _job["outputs_before"])
            missing_outputs = [str(path) for path in required_outputs if path not in generated_outputs]
            if _job["stop_requested"]:
                _job["status"] = "已停止"
                _job["status_code"] = "cancelled"
                append_job_log(_job, "转换任务已停止。")
            elif code == 0 and not missing_outputs:
                output = config["output_path"]
                _job["status"] = "转换完成"
                _job["status_code"] = "completed"
                _job["output"] = output
                _job["outputs"] = [str(path) for path in required_outputs]
                append_job_log(_job, f"配置文件：{_job['config_path']}")
            else:
                _job["status"] = f"失败（退出码 {code}）" if code else "转换失败"
                _job["status_code"] = "failed"
                if missing_outputs:
                    append_job_log(_job, "以下输出文件未生成：\n" + "\n".join(f"- {path}" for path in missing_outputs))
    except Exception as exc:
        with _job_lock:
            if _job and _job["process"] is process:
                if _job["stop_requested"]:
                    _job["status"] = "已停止"
                    _job["status_code"] = "cancelled"
                else:
                    _job["status"] = "转换失败"
                    _job["status_code"] = "failed"
                append_job_log(_job, f"任务异常：{exc}")
    finally:
        with _job_lock:
            if _job and _job["process"] is process:
                finished = _job.get("finished")
            else:
                finished = None
        if finished is not None:
            finished.set()


def job_snapshot(after: int = 0) -> dict:
    with _job_lock:
        if not _job:
            return {
                "job_id": "",
                "status": "空闲",
                "status_code": "idle",
                "log": "",
                "log_offset": 0,
                "output": "",
                "outputs": [],
                "output_dir": "",
            }
        lines = list(_job["lines"])
        total = _job.get("log_total", len(lines))
        available_from = max(0, total - len(lines))
        start = max(after, available_from)
        index = max(0, start - available_from)
        return {
            "job_id": _job["id"],
            "status": _job["status"],
            "status_code": _job["status_code"],
            "log": "\n".join(lines[index:]),
            "log_offset": total,
            "output": _job["output"],
            "outputs": list(_job["outputs"]),
            "output_dir": str(Path(_job["config"]["output_path"]).parent),
        }


def stop_job(wait: bool = False) -> dict:
    finished = None
    with _job_lock:
        job = _job
        is_unfinished = job is not None and job_unfinished(job)
        if not is_unfinished:
            if job is not None:
                finished = job.get("finished")
            job = None
        else:
            job["stop_requested"] = True
            job["status"] = "正在停止"
            job["status_code"] = "stopping"
            append_job_log(job, "正在停止转换及其子进程……")
    if job is None:
        if finished is not None:
            finished.wait(timeout=1)
        return job_snapshot()
    kill_process_group(job["process"], wait=wait)
    if wait:
        finished = job.get("finished")
        if finished is not None:
            finished.wait(timeout=5)
    return job_snapshot()


def probe_environment(python: Path, code: str) -> list[str]:
    if not python.is_file():
        return [f"环境缺失：{python}"]
    try:
        result = subprocess.run([str(python), "-c", code], capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired:
        return ["环境检查超时"]
    except (OSError, subprocess.SubprocessError) as exc:
        return ["环境检查失败", str(exc)]
    if result.returncode == 0:
        return [line for line in result.stdout.strip().splitlines() if line]
    details = result.stderr.strip().splitlines()
    return ["环境加载失败", details[-1] if details else "没有更多错误信息"]


def environment_snapshot() -> str:
    llm = llm_python()
    vision = vision_python()
    rows = [f"主机：{platform_label()}", f"界面 Python：{sys.version.split()[0]}", "界面：本地 HTML", "", f"语言模型环境：{llm}"]
    llm_code = (
        "try:\n"
        " from rkllm.api import RKLLM\n"
        " print('RKLLM：可用')\n"
        "except ModuleNotFoundError:\n"
        " print('RKLLM：缺失')\n"
        "except Exception:\n"
        " print('RKLLM：加载失败')\n"
        "try:\n"
        " import torch\n"
        " print('Torch：'+torch.__version__)\n"
        " print('CUDA：'+('可用' if torch.cuda.is_available() else '不可用'))\n"
        "except ModuleNotFoundError:\n"
        " print('Torch：缺失')\n"
        " print('CUDA：不可用')\n"
        "except Exception:\n"
        " print('Torch：加载失败')\n"
        " print('CUDA：不可用')\n"
        "try:\n"
        " import torchvision\n"
        " print('Torchvision：'+torchvision.__version__)\n"
        "except ModuleNotFoundError:\n"
        " print('Torchvision：缺失')\n"
        "except Exception:\n"
        " print('Torchvision：加载失败')\n"
        "try:\n"
        " import onnx\n"
        " print('ONNX：'+onnx.__version__)\n"
        "except ModuleNotFoundError:\n"
        " print('ONNX：缺失')\n"
        "except Exception:\n"
        " print('ONNX：加载失败')"
    )
    rows.extend(f"  {line}" for line in probe_environment(llm, llm_code))
    rows += ["", f"视觉转换环境：{vision}"]
    vision_code = (
        "try:\n"
        " from rknn.api import RKNN\n"
        " print('RKNN：可用')\n"
        "except ModuleNotFoundError:\n"
        " print('RKNN：缺失')\n"
        "except Exception:\n"
        " print('RKNN：加载失败')\n"
        "try:\n"
        " import torch\n"
        " print('Torch：'+torch.__version__)\n"
        "except ModuleNotFoundError:\n"
        " print('Torch：缺失')\n"
        "except Exception:\n"
        " print('Torch：加载失败')\n"
        "try:\n"
        " import onnx\n"
        " print('ONNX：'+onnx.__version__)\n"
        "except ModuleNotFoundError:\n"
        " print('ONNX：缺失')\n"
        "except Exception:\n"
        " print('ONNX：加载失败')"
    )
    rows.extend(f"  {line}" for line in probe_environment(vision, vision_code))
    return "\n".join(rows)


def validate_local_request(headers, port: int) -> None:
    local_hosts = {
        "localhost", "127.0.0.1", "[::1]",
        f"localhost:{port}", f"127.0.0.1:{port}", f"[::1]:{port}",
    }
    if headers.get("Host", "").strip().lower() not in local_hosts:
        raise ValueError("只允许从本机访问。")
    origin = headers.get("Origin")
    if origin:
        parsed = urllib.parse.urlsplit(origin)
        if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() not in local_hosts:
            raise ValueError("只允许从本机发起请求。")
    if headers.get("Sec-Fetch-Site", "").lower() == "cross-site":
        raise ValueError("已拒绝跨站请求。")


def select_directory(initial_path: str = "") -> str:
    initial = resolve_user_path(initial_path, "初始目录") if initial_path else Path.home()
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
        if self.headers.get_content_type().lower() != "application/json":
            raise ValueError("Content-Type 必须是 application/json。")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("请求长度无效。") from exc
        if length < 0 or length > MAX_JSON_BYTES:
            raise ValueError("请求数据过大。")
        data = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(data, dict):
            raise ValueError("请求数据必须是 JSON 对象。")
        return data

    def do_GET(self) -> None:
        try:
            validate_local_request(self.headers, self.server.server_address[1])
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            if path == "/":
                self.send_bytes(INDEX_PATH.read_bytes(), "text/html; charset=utf-8")
            elif path == "/app.js":
                self.send_bytes((GUI_DIR / "app.js").read_bytes(), "text/javascript; charset=utf-8")
            elif path == "/style.css":
                self.send_bytes((GUI_DIR / "style.css").read_bytes(), "text/css; charset=utf-8")
            elif path == "/api/config":
                self.send_json({"platform": platform_label(), "default_device": DEFAULT_DEVICE, "cuda": CUDA_AVAILABLE, "default_preset": DEFAULT_PRESET, "default_quantization": PRESETS[DEFAULT_PRESET]["default_quantization"], "presets": PRESETS, "quantizations": QUANTIZATIONS, "vision_models": VISION_MODELS})
            elif path == "/api/job":
                after_text = urllib.parse.parse_qs(parsed.query).get("after", ["0"])[0]
                try:
                    after = int(after_text)
                except ValueError as exc:
                    raise ValueError("日志偏移必须是整数。") from exc
                if after < 0:
                    raise ValueError("日志偏移不能为负数。")
                self.send_json(job_snapshot(after))
            else:
                self.send_json({"error": "页面不存在"}, 404)
        except ValueError as exc:
            self.send_json({"error": str(exc)}, 400)
        except Exception as exc:
            self.send_json({"error": f"服务器内部错误：{exc}"}, 500)

    def do_POST(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        try:
            validate_local_request(self.headers, self.server.server_address[1])
            if path == "/api/environment":
                self.read_json()
                self.send_json({"text": environment_snapshot()})
            elif path == "/api/start":
                self.send_json({"job_id": start_job(self.read_json())})
            elif path == "/api/stop":
                self.read_json()
                self.send_json(stop_job())
            elif path == "/api/select-directory":
                selected = select_directory(self.read_json().get("initial_path", ""))
                self.send_json({"path": selected})
            elif path == "/api/open-folder":
                target = resolve_user_path(self.read_json().get("path"), "路径")
                target = target if target.is_dir() else target.parent
                if not target.is_dir():
                    raise ValueError("目录不存在。")
                if platform_label() == "WSL2":
                    if not shutil.which("explorer.exe") or not shutil.which("wslpath"):
                        raise ValueError("WSL 下未找到 explorer.exe 或 wslpath。")
                    converted = subprocess.run(
                        ["wslpath", "-w", str(target)],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if converted.returncode != 0 or not converted.stdout.strip():
                        raise ValueError("无法将 WSL 路径转换为 Windows 路径。")
                    subprocess.Popen(
                        ["explorer.exe", converted.stdout.strip()],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                elif sys.platform == "linux":
                    if not shutil.which("xdg-open"):
                        raise ValueError("未找到 xdg-open，请安装桌面文件管理器。")
                    subprocess.Popen(["xdg-open", str(target)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                elif shutil.which("explorer.exe"):
                    subprocess.Popen(["explorer.exe", str(target)])
                else:
                    raise ValueError("未找到可用的文件管理器。")
                self.send_json({"ok": True})
            else:
                self.send_json({"error": "接口不存在"}, 404)
        except (ValueError, TypeError, OSError, subprocess.SubprocessError) as exc:
            self.send_json({"error": str(exc)}, 400)
        except Exception as exc:
            self.send_json({"error": f"服务器内部错误：{exc}"}, 500)


def main() -> None:
    port_text = os.environ.get("GRADIO_SERVER_PORT", os.environ.get("RKLLM_WORKBENCH_PORT", "7860"))
    try:
        port = int(port_text)
    except ValueError as exc:
        raise SystemExit(f"端口必须是整数：{port_text}") from exc
    if not 1 <= port <= 65535:
        raise SystemExit(f"端口必须在 1 到 65535 之间：{port}")
    cleanup_runs()
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as exc:
        raise SystemExit(f"无法监听端口 {port}：{exc}") from exc
    print(f"RKLLM 工作台已启动：http://127.0.0.1:{port}", flush=True)

    def handle_shutdown(signum, frame):
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, handle_shutdown)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        stop_job(wait=True)
        server.server_close()


if __name__ == "__main__":
    main()
