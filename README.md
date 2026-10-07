# RKLLM 工作台

一个给瑞芯微（Rockchip）NPU 开发板准备大模型 / 视觉大模型的一站式转换工具。

RK3588、RK3576 这类开发板使用官方 **RKLLM Runtime** 推理时，需要 `.rkllm` 格式的模型。这个项目帮你把本地开源模型，通过网页界面转换成这种格式，再部署到开发板。

整个转换过程都在你自己的电脑上完成，**模型文件不会被上传到任何服务器**。

> 网页工作台的转换引擎基于官方 [rknn-llm](https://github.com/airockchip/rknn-llm) 项目（v1.3.1）。仓库另提供独立转换器与开放 NPU 后端的实验代码，当前尚未接入网页界面，支持范围见下文。

## 它能做什么

- 把 Hugging Face 上的文本大模型转换成 RKLLM 格式
- 把 Qwen2.5-VL、Qwen3-VL、Qwen3.5 等视觉模型转换成"视觉 RKNN + 语言 RKLLM"两个文件
- 支持 RK3588、RK3576 两种目标平台预设
- 支持 W8A8、W4A16 等多种量化方式，自动完成模型压缩
- 全程图形界面操作，实时显示转换日志

## 网页工作台需要准备什么

- 一台 x86_64（Intel / AMD）电脑，建议内存 16 GB 以上、预留 20 GB 以上磁盘空间
- Linux 系统，或 Windows 系统（通过 WSL2 安装 Linux）
- 一个想转换的大模型（在 Hugging Face 下载到本地，模型文件夹即可）
- 网络连接（首次安装需要下载依赖）

### 支持的系统

| 系统 | 网页工作台 | 独立转换器（实验） |
| --- | --- | --- |
| Linux x86_64（Ubuntu / Arch 等） | ✅ 推荐，使用官方 Toolkit | 已验证有限模型配置 |
| Windows + WSL2 | ✅ 支持，通过 PowerShell 安装 | 可在 WSL2 的 Linux 环境中使用 |
| ARM64 Linux 主机 | 尚不支持官方转换流程 | 已在 RK3588 ARM64 Linux 上验证；其他 ARM64 主机尚未单独验证 |
| 原生 Windows（无需 WSL2） | 尚不支持 | 尚未适配验证，不能视为已支持 |

## 独立转换与 NPU 研究

以下为截至 **2026-10-07** 的实验进展，可通过命令行使用。

### 独立 RKLLM 转换器

[native_converter](native_converter/README.md) 使用 Python + NumPy 自行读取 Safetensors、变换权重并写入 `.rkllm`。当前支持有限 Llama 配置，以及指定 **Qwen3.5-0.8B 的语言部分**，输出为 **RK3588 单核 FP16**。

使用时先从配置匹配的官方参考文件提取 `.rkplan`，再用计划转换相同结构、相同分词器的新权重。计划保留元数据与 NPU 程序，不包含参考模型权重；转换主机无需安装或运行官方 Toolkit，也无需带有 NPU。首次取得参考文件仍依赖官方编译产物，生成的 `.rkllm` 仍由官方 Runtime 推理。

- 两种 Llama 小模型配置、四组 BPE 权重的独立输出与官方 1.3.1 逐字节一致；单层和双层随机小模型已通过 ARM64 转换及板端推理对照。
- Qwen3.5-0.8B 语言部分已完成 ARM64 转换、RK3588 NPU 加载和推理验证。该路径包含预分词标记修正及少量 FP32 舍入差异，完整文件不与官方参考逐字节一致，也尚未做完整质量评测。
- 尚未实现新模型结构的独立 NPU 编译、W8A8/W4A16 量化、RK3576、多核或完整多模态转换。

安装与命令见 [转换器说明](native_converter/README.md)。研究记录见 [逆向报告](doc/NATIVE_CONVERTER_REVERSE_ENGINEERING.md)、[Llama 真机验证](doc/NATIVE_CONVERTER_BOARD_VALIDATION.md) 和 [Qwen3.5 验证](doc/NATIVE_CONVERTER_QWEN35_VALIDATION.md)。参考模型和计划属于本地研究材料，未随仓库分发，需要自行准备匹配文件。

### 开放 NPU 矩阵后端

[native_npu](native_npu/README.md) 基于开源 [iwagumi](https://github.com/fukumori/iwagumi)（Apache-2.0），按矩阵形状生成寄存器命令，通过 RKNPU 内核驱动直接提交 RK3588 NPU。指令编码与驱动通信由 iwagumi 提供，这条路径无需官方 Toolkit、RKLLM/RKNN 用户态 Runtime 或官方参考程序。

在 RK3588、内核 6.1.115 / RKNPU 0.9.8 上，Qwen3.5 真实权重投影与随机小矩阵共 **11 组数值对照全部通过、17 次实际 NPU 提交成功**，包含单 token 对应的矩阵运算。最大绝对误差约 `2.38e-7`，对照使用相同 FP16 操作数的 CPU FP32 累加结果。

当前只完成基础矩阵后端，尚未运行完整语言模型，也不能生成新的完整 RKLLM 程序。每次调用仍重新上传权重，没有常驻权重缓存或整模型加速结论；板端仍需要 RKNPU 内核驱动。构建步骤见 [后端说明](native_npu/README.md)，证据与限制见 [NPU 阶段报告](doc/NATIVE_NPU_COMPILER_RESEARCH.md)。

另提供仅依赖 Python 标准库的 [NPU 程序解析工具](tools/inspect_npu_program.py) 和恢复的 [FlatBuffers schema](native_converter/rkllm_program.fbs)，可分析程序中的张量、任务、地址重定位及寄存器命令。

### ARM／原生 Windows 转换的当前边界

现阶段的跨平台转换方案是 **独立权重转换 + 匹配的参考程序**。ARM64 已有实机验证；原生 Windows 适配与验证、网页集成仍待完成。若要转换全新模型结构并完全摆脱官方参考程序，还需要实现完整 NPU 程序生成。转换主机可以用 CPU 完成这项工作，无需重写完整板端推理运行时。

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

第一次使用可以点击顶部或右侧的 **"检查环境"**，确认两个 Python 环境都正常后再开始转换。

### 转换文本模型（以 Qwen 为例）

1. 在 [Hugging Face](https://huggingface.co) 搜索并下载一个模型到本地，例如 [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)，下载时选择 **下载整个模型文件夹**（不要只下载单个文件）。
2. 打开工作台，"转换类型" 保持默认的 **"文本模型"**。
3. 在 **"模型目录"** 填写模型文件夹的完整路径。也可以点击选择按钮，在目录选择器中选中模型目录。
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
> 工作台启动时会保留最近 10 份任务配置和最近 3 个视觉转换临时目录，其余临时内容会自动清理；生成的模型结果不会被删除。

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
export RKLLM_WORKBENCH_PYPI_MIRROR="https://pypi.org/simple"
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
- 上游 RKLLM SDK 更新记录：[CHANGELOG.md](CHANGELOG.md)
- 性能测试数据：[benchmark.md](benchmark.md)

本项目遵循仓库原有许可证，详见 [LICENSE](LICENSE)。
