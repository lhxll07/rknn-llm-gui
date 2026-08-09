#!/usr/bin/env bash
set -euo pipefail

APP_HOME="${RKLLM_WORKBENCH_HOME:-$HOME/.local/share/rknn-llm-workbench}"
APP_HOME="${APP_HOME%/}"

if [[ "$APP_HOME" != /* || "$APP_HOME" == "/" || "$APP_HOME" == "$HOME" ]]; then
  echo "卸载路径不安全：$APP_HOME" >&2
  exit 1
fi

case "$APP_HOME" in
  */rknn-llm-workbench) ;;
  *)
    echo "为避免误删，卸载路径必须以 rknn-llm-workbench 结尾：$APP_HOME" >&2
    exit 1
    ;;
esac

if [[ ! -e "$APP_HOME" && ! -L "$APP_HOME" ]]; then
  echo "未找到托管安装目录：$APP_HOME"
  exit 0
fi

if [[ "${1:-}" != "--yes" ]]; then
  printf "将删除托管运行环境：%s\n确认卸载？[y/N] " "$APP_HOME"
  read -r answer
  if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
    echo "已取消卸载。"
    exit 0
  fi
fi

rm -rf -- "$APP_HOME"
echo "已卸载 RKLLM 工作台托管运行环境（包括语言和视觉环境）。"
echo "源码、模型、转换输出和 pip 缓存未删除。"
