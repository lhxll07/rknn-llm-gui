# RKLLM 工作台

一个给瑞芯微（Rockchip）NPU 开发板准备大模型 / 视觉大模型的一站式转换工具。

RK3588、RK3576 这类开发板只能运行一种叫做 **RKLLM** 的特殊模型格式。这个项目帮你把 Hugging Face 上的开源大模型，通过网页界面一键转换成开发板能直接跑的 `.rkllm` 文件。

整个转换过程都在你自己的电脑上完成，**模型文件不会被上传到任何服务器**。

> 转换引擎基于官方 [rknn-llm](https://github.com/airockchip/rknn-llm) 项目（v1.3.0），本项目在其基础上增加了图形界面和自动安装器。

## 它能做什么

- 把 Hugging Face 上的文本大模型转换成 RKLLM 格式
- 把 Qwen2.5-VL、Qwen3-VL、Qwen3.5 等视觉模型转换成"视觉 RKNN + 语言 RKLLM"两个文件
- 支持 RK3588、RK3576 两种目标平台预设
- 支持 W8A8、W4A16 等多种量化方式，自动完成模型压缩
- 全程图形界面操作，实时显示转换日志

## 你需要准备什么

- 一台 x86_64（Intel / AMD）电脑，建议内存 16 GB 以上、预留 20 GB 以上磁盘空间
- Linux 系统，或 Windows 系统（通过 WSL2 安装 Linux）
- 一个想转换的大模型（在 Hugging Face 下载到本地，模型文件夹即可）
- 网络连接（首次安装需要下载依赖）

### 支持的系统

| 系统 | 是否支持 | 说明 |
| --- | --- | --- |
| Linux x86_64（Ubuntu / Arch 等） | ✅ 推荐 | 官方转换工具只提供 Linux x86_64 版本 |
| Windows + WSL2 | ✅ 支持 | 通过 PowerShell 一键安装 |
| ARM Linux 主机 | ❌ 不支持 | 官方转换工具目前没有 ARM 版本 |

## 安装教程

### 方式一：Windows + WSL2

**第 1 步：安装 WSL2**

在 Windows PowerShell（管理员）中执行：

```powershell
wsl --install
```

安装完成后重启电脑，然后在开始菜单中打开 **Ubuntu**，按提示完成初始化（设置用户名和密码）。

**第 2 步：运行安装器**

1. 把本项目文件夹放到一个方便的位置，例如：

   ```text
   C:\Users\你的用户名\Documents\rknn-llm-gui
   ```

2. 打开 PowerShell，进入项目目录：

   ```powershell
   cd C:\Users\你的用户名\Documents\rknn-llm-gui
   ```

3. 运行安装命令：

   ```powershell
   .\installer\install.ps1
   ```

首次安装需要下载 micromamba 和两个 Python 环境，**根据网速可能需要 10~30 分钟**，请耐心等待，不要关闭窗口。

**第 3 步：打开工作台**

安装完成后会自动启动界面，浏览器访问：

```text
http://127.0.0.1:7860
```

> 电脑里装了多个 WSL 发行版时，先指定用哪个，再运行安装命令：
>
> ```powershell
> $env:RKLLM_WORKBENCH_WSL_DISTRO = "Ubuntu"
> .\installer\install.ps1
> ```

### 方式二：Linux（Ubuntu / Arch）

**第 1 步：安装 curl**

Ubuntu / Debian：

```bash
sudo apt install curl
```

Arch Linux：

```bash
sudo pacman -S --needed curl
```

**第 2 步：运行安装脚本**

```bash
cd rknn-llm-gui
bash installer/install.sh
```

安装完成后会自动启动工作台，浏览器打开 `http://127.0.0.1:7860`。

> 工作台运行在当前终端里，关闭终端或按 `Ctrl+C` 会停止服务。下次想再打开，重新运行安装脚本即可（环境已经装好，会直接复用并启动界面，不会重新下载）。

## 使用教程

### 界面说明

工作台分为左右两栏：

- 左边：转换设置（模型目录、目标平台、量化方式等）
- 右边：状态、转换日志、输出文件

第一次使用可以点左下角的 **"检查环境"**，确认两个 Python 环境都正常后再开始转换。

### 转换文本模型（以 Qwen 为例）

1. 在 [Hugging Face](https://huggingface.co) 搜索并下载一个模型到本地，例如 [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)，下载时选择 **下载整个模型文件夹**（不要只下载单个文件）。
2. 打开工作台，"转换类型" 保持默认的 **"文本模型"**。
3. 在 **"模型目录"** 填写模型文件夹的完整路径。也可以点下面的文件浏览器，选中模型目录里的任意一个文件，路径会自动填好。
4. **"目标平台预设"**：开发板是 RK3588 就选 `RK3588`，是 RK3576 就选 `RK3576`。
5. **"量化方式"**：使用默认值即可（RK3588 默认 W8A8，RK3576 默认 W4A16）。
6. **"校准数据 JSON"**：保持默认路径 `examples/rkllm_api_demo/export/data_quant.json`。
7. **"输出 RKLLM 文件"**：保持默认 `gui/runs/model.rkllm`，或改成你想要的路径。
8. 点击 **"开始转换"**，右侧日志会实时显示进度。
9. 转换完成后，"输出文件" 一栏会出现生成的 `.rkllm` 文件路径，把它拷贝到开发板即可使用。

### 转换视觉模型（Qwen2.5-VL 等）

1. 在 Hugging Face 下载视觉模型到本地，例如 [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)。
2. 打开工作台，把 **"转换类型"** 切换为 **"视觉模型"**。
3. 选择 **"视觉模型类型"**（Qwen2.5-VL / Qwen3-VL / Qwen3.5），并填写 **"模型目录"**。
4. 设置目标平台和量化方式（同上）。
5. **"视觉 ONNX 输出"** 和 **"视觉 RKNN 输出"** 使用默认路径即可。
6. 点击 **"开始转换"**。视觉模式会自动完成四步：导出 ONNX → 转 RKNN → 生成多模态校准数据 → 转换 RKLLM。
7. 完成后会得到两个文件：**视觉 RKNN**（处理图像）和**语言 RKLLM**（生成对话），配合板端示例程序一起部署。

> 转换生成的临时文件和结果默认放在 `gui/runs/` 目录下，该目录已被 Git 忽略，不会提交到代码仓库。

## 常见问题

### 下次怎么打开工作台？

重新运行安装脚本即可，已装好的环境会被自动检测并复用：

```powershell
.\installer\install.ps1
```

```bash
bash installer/install.sh
```

### 安装时下载慢或报网络错误

安装脚本默认使用国内 conda-forge 镜像（清华 + 中科大），通常不需要额外设置。如果仍失败，可以手动指定镜像源：

```bash
export RKLLM_WORKBENCH_MM_MIRROR="https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge"
export RKLLM_WORKBENCH_MM_MIRROR_FALLBACK="https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge"
```

设置后再运行安装脚本。

### 提示 `torch 2.4.0 和 torch 2.6.0 冲突`

说明两个工具包被装进了同一个环境。运行卸载脚本后重新安装即可——项目会创建两个互相隔离的 Python 环境（一个负责转 RKLLM，一个负责转 RKNN，两者要求的 PyTorch 版本不同，不能混装）。

### 提示 `No module named pkg_resources`

这是 RKNN Toolkit2 与新版 setuptools 的兼容问题。重新运行安装器即可，项目已把视觉环境的 setuptools 固定到兼容版本。

### 日志里出现 fast path 或 TracerWarning

这些是警告，不是错误。只要最后生成了 ONNX / RKNN / RKLLM 文件，就说明转换成功。

### 转换到一半失败了

先看右侧日志的最后几行。常见原因：模型路径填错、磁盘空间不足、内存不足。修改设置后重新点击"开始转换"即可。

### 从 Hugging Face 下载模型太慢

可以设置镜像地址后再下载：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

然后在浏览器或下载工具中正常下载。

## 卸载

先停止工作台，然后在项目根目录运行：

Linux / WSL2：

```bash
bash installer/uninstall.sh
```

Windows：

```powershell
.\installer\uninstall.ps1
```

卸载只删除项目自己安装的 micromamba 和两个 Python 环境，**不会删除你的模型、转换结果和源代码**。

## 进阶：手动运行

通常不需要手动操作，安装器会自动管理环境。想自己管理环境时，需要准备两个独立的 Python 3.10 环境，具体步骤见 [gui/README.md](gui/README.md)。

## 相关资源

- 官方 [rknn-llm](https://github.com/airockchip/rknn-llm) 项目
- 官方 [RKNN Toolkit2](https://github.com/airockchip/rknn-toolkit2) 项目
- 项目更新记录：[CHANGELOG.md](CHANGELOG.md)
- 性能测试数据：[benchmark.md](benchmark.md)

本项目遵循仓库原有许可证，详见 [LICENSE](LICENSE)。
