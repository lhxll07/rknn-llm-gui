# RKLLM 工作台

一个基于本地 HTML 界面的 RKLLM 模型转换工具。

## 当前功能

- 选择本地模型目录
- 选择 RK3588 或 RK3576 目标平台预设
- 自由选择 W8A8、W8A8 / G128、W4A16 或 W4A16 / G128 量化方式
- 支持 Qwen2.5-VL、Qwen3-VL 和 Qwen3.5 视觉模型转换
- 指定校准数据 JSON
- 实时增量查看转换日志
- 同时只运行一个转换任务

模型转换由官方 `rkllm_toolkit` 完成。界面不会上传模型文件。
视觉模式使用项目内置的多模态校准样本，并同时生成视觉 RKNN 和语言 RKLLM 文件。
启动时自动保留最近 10 份任务配置和最近 3 个视觉转换临时目录；模型输出不会被自动删除。

## 安装

### Linux 或 WSL2

安装器支持 Arch Linux、Ubuntu 和其他 Linux x86_64 系统。

Arch Linux 先安装 `curl`：

```bash
sudo pacman -S --needed curl
```

然后在项目根目录运行：

```bash
bash installer/install.sh
```

如果之前安装过旧版单环境安装器，建议先运行卸载脚本，再重新安装，
这样可以同时清理旧环境和新环境。

安装器会在项目专用目录中安装 micromamba，并创建两个 Python 3.10 环境：

- 语言环境（LLM）：界面、RKLLM、ONNX 导出和校准数据生成
- 视觉环境（Vision）：RKNN Toolkit2 和视觉 RKNN 转换

两个环境由项目自动管理，不会修改你已有的 Conda 或 micromamba 环境。
这是因为 RKLLM 要求 Torch 2.6，而 RKNN Toolkit2 要求 Torch 不高于 2.4，
两者不能安装在同一个环境中。视觉环境还会固定兼容版本的 setuptools，
因为当前 RKNN Toolkit2 仍使用 `pkg_resources`。

### Windows + WSL2

请先安装 WSL2 和一个 Linux 发行版，然后在 PowerShell 中运行：

```powershell
.\installer\install.ps1
```

如果使用 Arch WSL：

```powershell
$env:RKLLM_WORKBENCH_WSL_DISTRO = "Arch"
.\installer\install.ps1
```

电脑上只有一个 WSL 发行版时，安装器会自动使用它；同时安装多个发行版时，
默认使用 Ubuntu，或通过 `RKLLM_WORKBENCH_WSL_DISTRO` 指定发行版。

## 手动运行

如果已经准备好两个独立的 Python 环境，可以手动安装。语言环境安装：

```bash
python -m pip install -r rkllm-toolkit/packages/requirements.txt
python -m pip install --no-deps rkllm-toolkit/packages/rkllm_toolkit-1.3.0-cp310-cp310-linux_x86_64.whl
python -m pip install -r gui/requirements.txt
```

视觉环境单独安装视觉依赖：

```bash
python -m pip install -r gui/requirements-visual.txt \
  -i https://mirrors.aliyun.com/pypi/simple
```

启动界面时设置两个 Python 路径：

```bash
export RKLLM_WORKBENCH_LLM_PYTHON=/路径/llm/bin/python
export RKLLM_WORKBENCH_VISION_PYTHON=/路径/vision/bin/python
python gui/app.py
```

然后打开 `http://127.0.0.1:7860`。

## 卸载

先在运行界面的终端按 `Ctrl+C` 停止程序，然后在项目根目录运行：

```bash
bash installer/uninstall.sh
```

Windows + WSL2 用户可以在 PowerShell 中运行：

```powershell
.\installer\uninstall.ps1
```

卸载会删除项目托管的 micromamba、语言环境、视觉环境和已安装依赖，
不会删除源码、模型、转换输出或 pip 缓存，也不会影响你已有的 Conda 或 micromamba 环境。

## 注意事项

官方转换工具目前只提供 CPython 3.10、Linux x86_64 wheel。
因此原生 Arch x86_64 和 Arch WSL2 可以转换；ARM Linux 主机暂不支持转换。
