from __future__ import annotations

import os
import platform
import sys
from pathlib import Path


def is_wsl() -> bool:
    if platform.system() != "Linux":
        return False
    try:
        version = Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False
    return "microsoft" in version or "wsl" in version


def platform_label() -> str:
    if is_wsl():
        return "WSL2"
    if platform.system() == "Linux":
        return "原生 Linux"
    return platform.system()


def default_app_home() -> Path:
    configured = os.environ.get("RKLLM_WORKBENCH_HOME")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".local/share/rknn-llm-workbench"


def default_llm_python() -> Path:
    return default_app_home() / "micromamba/envs/rknn-llm-workbench-llm/bin/python"


def default_vision_python() -> Path:
    return default_app_home() / "micromamba/envs/rknn-llm-workbench-vision/bin/python"


def llm_python() -> Path:
    configured = os.environ.get("RKLLM_WORKBENCH_LLM_PYTHON")
    if configured is None:
        configured = os.environ.get("RKLLM_WORKBENCH_PYTHON")
    if configured:
        return Path(configured).expanduser()
    managed = default_llm_python()
    if managed.is_file():
        return managed
    return Path(sys.executable)


def vision_python() -> Path:
    configured = os.environ.get("RKLLM_WORKBENCH_VISION_PYTHON")
    if configured is None:
        configured = os.environ.get("RKLLM_WORKBENCH_PYTHON")
    if configured:
        return Path(configured).expanduser()
    managed = default_vision_python()
    if managed.is_file():
        return managed
    return Path(sys.executable)


def converter_python() -> Path:
    """兼容旧配置，转换主流程始终使用 LLM 环境。"""
    return llm_python()
