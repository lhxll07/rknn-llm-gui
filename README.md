# RKLLM 工作台

一个基于 Gradio 的本地 RKLLM 模型转换界面，面向希望在瑞芯微 NPU 上部署大语言模型和视觉语言模型的用户。

本项目基于官方 [rknn-llm](https://github.com/airockchip/rknn-llm) 仓库，增加了图形界面、自动安装器和视觉模型转换流程。模型文件始终在本机处理，不会上传到服务器。

## 功能

- 通过浏览器图形界面完成模型转换
- 支持文本模型和视觉模型两种模式
- 提供 RK3588、RK3576 目标平台预设
- 可选择 W8A8、W8A8 / G128、W4A16、W4A16 / G128 量化方式
- 支持 Qwen2.5-VL、Qwen3-VL、Qwen3.5 视觉模型
- 自动生成视觉 ONNX、视觉 RKNN 和语言 RKLLM 文件
- 运行日志固定在窗口内，并支持滚动和复制
- 支持 Arch Linux、Ubuntu 和 WSL2
- 安装器自动管理项目专用 Python 环境

## 快速安装

### Linux

要求：Linux x86_64 和 `curl`。Arch Linux 用户先安装 `curl`：

```bash
sudo pacman -S --needed curl
```

在项目根目录执行：

```bash
bash installer/install.sh
```

安装器会自动完成以下工作：

1. 安装项目专用的 micromamba。
2. 创建语言转换环境和视觉转换环境。
3. 安装官方 RKLLM Toolkit、RKNN Toolkit2 和 Gradio。
4. 启动工作台。

浏览器打开：

```text
http://127.0.0.1:7860
```

### Windows + WSL2

先安装 WSL2 和一个 Linux 发行版，然后在 PowerShell 中进入项目目录并执行：

```powershell
.\installer\install.ps1
```

电脑中有多个 WSL 发行版时，可以指定目标发行版：

```powershell
$env:RKLLM_WORKBENCH_WSL_DISTRO = "Arch"
.\installer\install.ps1
```

安装完成后，直接在 Windows 浏览器打开 `http://127.0.0.1:7860`。

## 使用流程

### 文本模型

1. 将转换类型设为“文本模型”。
2. 选择 Hugging Face 模型目录。
3. 选择目标平台预设。
4. 选择量化方式。
5. 确认校准数据和输出文件路径。
6. 点击“开始转换”。

默认校准数据位于：

```text
examples/rkllm_api_demo/export/data_quant.json
```

### 视觉模型

1. 将转换类型设为“视觉模型”。
2. 选择模型类型和模型目录。
3. 选择目标平台。
4. 设置 ONNX、RKNN 和 RKLLM 输出路径。
5. 点击“开始转换”。

视觉模式会按顺序执行：

```text
模型 -> ONNX -> RKNN
              -> 多模态校准数据 -> RKLLM
```

转换过程会使用项目内置的多模态校准样本。生成的临时文件默认位于 `gui/runs/`，该目录已加入 Git 忽略规则。

## 为什么需要两个环境

官方 RKLLM Toolkit 和 RKNN Toolkit2 对 Torch 的要求不兼容：

| 环境 | 用途 | 关键版本 |
| --- | --- | --- |
| 语言环境 | 界面、ONNX 导出、校准数据、RKLLM | Torch 2.6.0、Torchvision 0.21.0 |
| 视觉环境 | 视觉 RKNN 转换 | RKNN Toolkit2 2.3.2、Torch 不高于 2.4.0 |

因此不要把两个工具包安装到同一个 Conda 或 micromamba 环境中。项目安装器会自动创建并使用两个独立环境，也不会修改你已有的环境。

## 支持范围

### 转换主机

- 原生 Linux x86_64
- Arch Linux x86_64
- Ubuntu x86_64
- Windows + WSL2

官方 RKLLM 工具包目前不支持 ARM Linux 主机直接转换。

### 目标平台

当前界面提供：

- RK3588
- RK3576

官方工具包的其他平台和模型能力，以官方仓库说明为准。

## 手动安装

通常不需要手动安装。确实需要手动管理环境时，请准备两个独立的 Python 3.10 环境。

语言环境：

```bash
python -m pip install -r rkllm-toolkit/packages/requirements.txt
python -m pip install --no-deps rkllm-toolkit/packages/rkllm_toolkit-1.3.0-cp310-cp310-linux_x86_64.whl
python -m pip install -r gui/requirements.txt
```

视觉环境：

```bash
python -m pip install -r gui/requirements-visual.txt \
  -i https://mirrors.aliyun.com/pypi/simple
```

启动时指定两个 Python：

```bash
export RKLLM_WORKBENCH_LLM_PYTHON=/路径/语言环境/bin/python
export RKLLM_WORKBENCH_VISION_PYTHON=/路径/视觉环境/bin/python
python gui/app.py
```

更多安装细节见 [GUI 说明](gui/README.md)。

## 卸载

停止工作台后，在项目根目录执行：

```bash
bash installer/uninstall.sh
```

Windows + WSL2 用户执行：

```powershell
.\installer\uninstall.ps1
```

卸载只删除项目托管的 micromamba、语言环境和视觉环境，不会删除源码、模型、转换输出或 pip 缓存，也不会影响用户已有的 Conda 或 micromamba 环境。

## 常见问题

### 出现 `torch 2.4.0` 和 `torch 2.6.0` 冲突

说明两个工具包被安装到了同一个环境。不要手动反复升降 Torch，运行项目卸载器后重新执行安装器即可。

### 出现 `No module named pkg_resources`

这是 RKNN Toolkit2 与新版 setuptools 的兼容问题。项目已经固定视觉环境的 setuptools 版本；重新运行安装器即可。

### 出现 fast path 或 `TracerWarning`

这类信息通常是警告，不代表转换失败。以日志中是否生成 ONNX、RKNN 或 RKLLM 文件为准。

## 官方资料

- [官方 rknn-llm 仓库](https://github.com/airockchip/rknn-llm)
- [官方 RKNN Toolkit2 仓库](https://github.com/airockchip/rknn-toolkit2)
- [官方 RKLLM 版本记录](CHANGELOG.md)
- [项目宣传页和录制说明](promo/README.md)

本项目遵循仓库中的原有许可证。具体内容见 [LICENSE](LICENSE)。
