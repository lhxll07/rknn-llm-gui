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

# micromamba 下载与环境创建使用的 conda-forge 镜像。
# 默认清华镜像（linux-64 和 noarch 都完整），中科大镜像兜底。
# 可通过环境变量覆盖：
#   RKLLM_WORKBENCH_MM_MIRROR            主镜像
#   RKLLM_WORKBENCH_MM_MIRROR_FALLBACK   备用镜像
MM_MIRROR_URL="${RKLLM_WORKBENCH_MM_MIRROR:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge}"
MM_FALLBACK_URL="${RKLLM_WORKBENCH_MM_MIRROR_FALLBACK:-https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge}"

mkdir -p "$APP_HOME/bin" "$MAMBA_ROOT_PREFIX"
if [[ ! -x "$MAMBA_BIN" ]]; then
  echo "正在下载 micromamba……"
  downloaded=""
  if command -v python3 >/dev/null 2>&1; then
    # 用 Python 标准库从镜像解析最新版本并解压官方 tar 包，
    # 不依赖系统 bzip2；镜像不可达时自动尝试下一个。
    if python3 - "$APP_HOME" "$MM_MIRROR_URL" \
      "$MM_FALLBACK_URL" \
      "https://mirrors.aliyun.com/anaconda/cloud/conda-forge" <<'PY'; then
import bz2
import io
import json
import os
import re
import sys
import tarfile
import urllib.request

root = sys.argv[1]
mirrors = sys.argv[2:]


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "rknn-llm-gui-installer"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def version_key(filename):
    match = re.match(
        r"micromamba-(\d+)\.(\d+)\.(\d+)(?:\.(\d+))?(?:-(\d+))?", filename
    )
    if not match:
        return (0, 0, 0, 0, 0)
    return tuple(int(part) if part else 0 for part in match.groups())


package_url = None
for base in mirrors:
    try:
        try:
            data = json.loads(
                bz2.decompress(fetch(base + "/linux-64/current_repodata.json.bz2"))
            )
        except Exception:
            data = json.loads(fetch(base + "/linux-64/current_repodata.json"))
        candidates = [
            name
            for name in data["packages"]
            if name.startswith("micromamba-") and name.endswith(".tar.bz2")
        ]
        if candidates:
            package_url = base + "/linux-64/" + max(candidates, key=version_key)
            break
    except Exception:
        continue
if package_url is None:
    raise SystemExit("错误：无法从任何镜像获取 micromamba，请检查网络后重试。")

os.makedirs(os.path.join(root, "bin"), exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(fetch(package_url)), mode="r:bz2") as archive:
    archive.extract("bin/micromamba", root)
PY
      downloaded="1"
    fi
  fi
  if [[ -z "$downloaded" ]]; then
    echo "镜像下载失败，尝试 GitHub 官方静态二进制……" >&2
    if ! curl -fsSL --connect-timeout 15 --max-time 180 \
      https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64 \
      -o "$MAMBA_BIN"; then
      echo "micromamba 下载失败。请检查网络，或设置 RKLLM_WORKBENCH_MM_MIRROR 指向可用的 conda-forge 镜像后重试。" >&2
      exit 1
    fi
  fi
  chmod +x "$MAMBA_BIN"
fi

export MAMBA_ROOT_PREFIX
create_environment() {
  local prefix="$1"
  local environment_file="$2"
  local label="$3"
  if [[ ! -x "$prefix/bin/python" ]]; then
    echo "正在创建${label}托管 Python 环境……"
    "$MAMBA_BIN" create -y -p "$prefix" -f "$environment_file" \
      --override-channels -c "$MM_MIRROR_URL" -c "$MM_FALLBACK_URL"
  else
    "$MAMBA_BIN" install -y -p "$prefix" -f "$environment_file" \
      --override-channels -c "$MM_MIRROR_URL" -c "$MM_FALLBACK_URL"
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
