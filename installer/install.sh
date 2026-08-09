#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_HOME="${RKLLM_WORKBENCH_HOME:-$HOME/.local/share/rknn-llm-workbench}"
MAMBA_ROOT_PREFIX="$APP_HOME/micromamba"
MAMBA_BIN="$APP_HOME/bin/micromamba"
LLM_ENV_PREFIX="$MAMBA_ROOT_PREFIX/envs/rknn-llm-workbench-llm"
VISION_ENV_PREFIX="$MAMBA_ROOT_PREFIX/envs/rknn-llm-workbench-vision"
LLM_PYTHON="$LLM_ENV_PREFIX/bin/python"
VISION_PYTHON="$VISION_ENV_PREFIX/bin/python"

DISTRO_ID="unknown"
if [[ -r /etc/os-release ]]; then
  # 仅用于在缺少启动依赖时给出对应发行版的安装提示。
  . /etc/os-release
  DISTRO_ID="${ID:-unknown}"
fi

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "此安装器适用于 Linux 和 WSL2。" >&2
  exit 1
fi

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "RKLLM 工具包目前要求 Linux x86_64 主机。" >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "安装 micromamba 需要 curl。" >&2
  case "$DISTRO_ID" in
    arch|manjaro)
      echo "请运行：sudo pacman -S --needed curl" >&2
      ;;
    ubuntu|debian|linuxmint)
      echo "请运行：sudo apt install curl" >&2
      ;;
  esac
  exit 1
fi

mkdir -p "$APP_HOME/bin" "$MAMBA_ROOT_PREFIX"
if [[ ! -x "$MAMBA_BIN" ]]; then
  echo "正在下载 micromamba……"
  curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C "$APP_HOME/bin" bin/micromamba
  mv "$APP_HOME/bin/bin/micromamba" "$MAMBA_BIN"
  rmdir "$APP_HOME/bin/bin"
fi

export MAMBA_ROOT_PREFIX
create_environment() {
  local prefix="$1"
  local environment_file="$2"
  local label="$3"
  if [[ ! -x "$prefix/bin/python" ]]; then
    echo "正在创建${label}托管 Python 环境……"
    "$MAMBA_BIN" create -y -p "$prefix" -f "$environment_file"
  else
    "$MAMBA_BIN" install -y -p "$prefix" -f "$environment_file"
  fi
}

create_environment \
  "$LLM_ENV_PREFIX" \
  "$ROOT_DIR/installer/environment-linux-64.yml" \
  "语言模型"
create_environment \
  "$VISION_ENV_PREFIX" \
  "$ROOT_DIR/installer/environment-vision-linux-64.yml" \
  "视觉转换"

WHEEL="$(find "$ROOT_DIR/rkllm-toolkit/packages" -maxdepth 1 -type f \
  -name 'rkllm_toolkit-*-cp310-*-linux_x86_64.whl' -print -quit)"
if [[ -z "$WHEEL" ]]; then
  echo "未找到 CPython 3.10 版 RKLLM 工具包 wheel。" >&2
  exit 1
fi

echo "正在安装官方 RKLLM 依赖……"
"$LLM_PYTHON" -m pip install -r "$ROOT_DIR/rkllm-toolkit/packages/requirements.txt"
"$LLM_PYTHON" -m pip install --no-deps "$WHEEL"
"$LLM_PYTHON" -m pip install -r "$ROOT_DIR/gui/requirements.txt"
echo "正在安装视觉转换依赖……"
"$VISION_PYTHON" -m pip install -r "$ROOT_DIR/gui/requirements-visual.txt" \
  -i "https://mirrors.aliyun.com/pypi/simple"

echo "正在检查两个托管环境……"
"$LLM_PYTHON" -m pip check
"$VISION_PYTHON" -m pip check

export RKLLM_WORKBENCH_LLM_PYTHON="$LLM_PYTHON"
export RKLLM_WORKBENCH_VISION_PYTHON="$VISION_PYTHON"
export RKLLM_WORKBENCH_PYTHON="$LLM_PYTHON"
echo "正在使用 $LLM_PYTHON 启动 RKLLM 工作台"
exec "$LLM_PYTHON" "$ROOT_DIR/gui/app.py"
