# ARM64 Linux 主机转换支持调研

后续进展：已实现有限范围的 [独立转换器原型](../native_converter/README.md)，转换阶段不调用官方 Toolkit，但复用匹配参考文件中的 NPU 程序。四组 BPE 模型输出已通过官方文件对照，单层及双层小模型也通过了 ARM64 真机转换和 RK3588 NPU 推理；见 [逆向报告](NATIVE_CONVERTER_REVERSE_ENGINEERING.md) 与 [真机验证报告](NATIVE_CONVERTER_BOARD_VALIDATION.md)。指定 Qwen3.5-0.8B 的语言部分也已通过 ARM64 转换和 NPU 推理，见 [Qwen3.5 报告](NATIVE_CONVERTER_QWEN35_VALIDATION.md)。下文保留初始调研记录，完整工作台的架构限制仍然存在。

调研日期：2026-10-07。范围：在其他 ARM64 Linux 主机上运行本工作台并完成模型转换。
本次核对了项目代码、内置 RKLLM wheel、官方 SDK 文档、官方公开 README 和 PyPI 发布元数据；未修改转换或安装代码，未在 ARM64 真机上执行转换。

## 结论

ARM64 Linux 支持可以分阶段增加，但当前不能承诺原生完成整个 RKLLM 转换流程。

| 能力 | 调研结论 | 依据与限制 |
| --- | --- | --- |
| 网页界面与 Python HTTP 服务 | 代码层面没有明显 CPU 架构障碍 | 使用 Python 标准库；安装器与转换环境探测需要拆分，尚未真机验证 |
| 视觉 ONNX 转 RKNN | 有官方 ARM64 转换包，具备原生适配基础 | RKNN Toolkit2 2.3.2 已发布 aarch64 wheel；仍需验证依赖、实际模型与主机资源 |
| 模型导出 ONNX、准备校准数据 | 具备 ARM64 依赖基础，需按模型验证 | PyTorch / ONNX 有 ARM64 wheel；不宜照搬 RKLLM 的整套依赖 |
| 文本模型转 RKLLM | 当前内置工具包不能原生运行 | RKLLM 1.3.1 wheel 及其中转换核心均为 x86_64 二进制 |
| 完整视觉大模型转换 | 仍受 RKLLM 部分限制 | 流程最后还要生成语言部分的 `.rkllm` |
| x86_64 仿真转换 | 可作为实验方向，兼容性和性能未证实 | QEMU / Box64 可研究；尚无本次调研验证过的 RKLLM 完整转换结果 |

针对“在 ARM64 主机上本地完成转换”的目标，建议先验证 x86_64 CPU 仿真能否生成可用 RKLLM 文件，再决定是否增加实验模式。原生完整转换需要官方提供 ARM64 RKLLM 转换器或可构建的核心源码。

## 已核实的证据

### RKLLM 的主要障碍是二进制架构

官方公开 README 当前标注最新版本为 v1.3.1，并区分 PC 上的 Toolkit 转换与开发板上的 Runtime 推理。

项目内置的安装包：

```text
rkllm-toolkit/packages/rkllm_toolkit-1.3.1-cp310-cp310-linux_x86_64.whl
```

读取其 `WHEEL` 元数据得到：

```text
Root-Is-Purelib: false
Tag: cp310-cp310-linux_x86_64
```

直接读取包内 ELF 头，检查到的 33 个 `.so` 均为 `e_machine = 62`（x86-64），包括：

```text
rkllm/api/rkllm_base.cpython-310-x86_64-linux-gnu.so
rkllm/base/converter.cpython-310-x86_64-linux-gnu.so
rkllm/base/quantizer.cpython-310-x86_64-linux-gnu.so
rkllm/lib/ekml.so
rkllm/lib/librkllmc.so
```

AArch64 的 ELF 架构编号为 183。因此放开 `uname -m` 检查、改 wheel 文件名或修改 pip 平台标签，都无法让这些库被 ARM64 Python 原生加载。已检查的发布内容中没有足够的转换核心源码，无法据此直接交叉编译 ARM64 转换器。

官方 SDK 中文文档第 10–12 页也将 x86_64 Toolkit wheel 与 aarch64 / armhf Runtime 库分开列出。仓库中的 ARM Runtime 用于 Rockchip 板端推理，不能替代 Toolkit 执行模型转换；它也不是适用于任意 ARM 主机的通用推理库。

官方当前 README 支持 Python 3.10 / 3.11 / 3.12；本项目安装器选择 Python 3.10。增加其他 Python 版本不能解决 CPU 架构限制。

### RKNN Toolkit2 已有 ARM64 完整转换包

PyPI 的 2.3.2 发布元数据明确包含：

```text
rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

同一版本还发布了 Python 3.6–3.12 的 aarch64 wheel。CPython 3.10 ARM64 包上传于 2025-04-09，大小为 37,641,091 字节。

这是 `rknn-toolkit2` 转换包。`rknn-toolkit-lite2` 的用途是板端加载和运行已转换的模型，不能用来替代 ONNX → RKNN 转换。

该版本 PyPI 元数据要求 `torch>=1.10.1,<=2.4.0`；项目 RKLLM 依赖固定 `torch==2.6.0`。ARM64 适配仍需保留独立语言与视觉环境。

进一步检查 PyPI 元数据确认，下列版本有 CPython 3.10 的 ARM64 wheel：

| 依赖 | ARM64 wheel |
| --- | --- |
| torch 2.4.0 | manylinux2014_aarch64 |
| torch 2.6.0 | manylinux_2_28_aarch64 |
| onnx 1.18.0 | manylinux_2_17_aarch64 |
| onnxruntime 1.20.1 | manylinux_2_27 / 2_28_aarch64 |

这说明视觉转换主要依赖具有 ARM64 发布基础，不代表完整依赖组合已验证。上述 Linux wheel 对 glibc 版本有要求，不能据此承诺兼容所有 ARM64 Linux 发行版。

另一个安装注意点：项目语言依赖中的 `auto_gptq==0.7.1` 在所查 PyPI 发布元数据中没有 CPython 3.10 ARM64 wheel，仅提供源码包等发布内容。独立 ONNX 导出环境应按脚本实际依赖设计，避免无条件安装整套 RKLLM 依赖。

## 当前项目需要改动的位置

| 文件 | 当前行为 | ARM64 适配需要处理的内容 |
| --- | --- | --- |
| `installer/install.sh` | 拒绝非 x86_64；下载 linux-64 micromamba / x86_64 二进制；只匹配 x86_64 RKLLM wheel | 按实际执行模式选择架构与环境，分别处理界面、原生视觉和仿真语言转换 |
| `gui/runtime.py` | 返回本机语言 / 视觉 Python 路径 | 区分原生执行和仿真执行；单个 Python 路径不足以描述容器命令 |
| `gui/server.py` | 以语言环境 Python 启动 worker，使用同一方式探测 CUDA 和依赖 | 统一构建执行命令、报告架构与能力；调整仿真环境的探测超时和停止任务方式 |
| `gui/worker.py` | 文本转换直接导入 RKLLM；视觉转换串联四个步骤 | 让 RKLLM 步骤在可用执行环境中运行；独立视觉转换需要新的任务入口 |
| `gui/index.html` / `gui/app.js` | 仅提供文本与完整视觉模型转换 | 显示主机实际支持的操作；实验模式说明应与实际验证结果对应 |
| `README.md` / `gui/README.md` | 统一说明工作台要求 x86_64 主机 | 按界面、RKNN、RKLLM 和执行方式分别说明支持情况 |

当前安装器的两个 YAML 文件主要声明 Python 3.10 与 pip；它们的文件名包含 `linux-64`，但主要架构限制实际来自 shell 下载地址、wheel 选择和转换核心。

网页服务本身没有明显架构依赖，但启动时会探测语言环境中的 PyTorch，环境检查也假定本机能够运行对应 Python。单独安装界面时需要明确这些能力的可用状态。

## 可选实现路线

### 原生 ARM64 界面与独立视觉转换

可以增加轻量界面安装方式，以及独立 ONNX → RKNN 任务。完整视觉大模型转换仍需要语言部分的 RKLLM 生成能力。

维护成本：中等。工作主要是拆分能力、任务入口和安装方式，现有日志与任务管理可以部分复用。此方案无法单独满足完整文本模型本地转换目标。

### ARM64 主机上的 x86_64 CPU 仿真

QEMU Linux user-mode 可执行其他 CPU 架构的用户态程序，并转换系统调用、信号和线程操作。可先用一个包含 x86_64 Python、RKLLM 和依赖的 Linux 环境做概念验证。

Docker `--platform=linux/amd64` 只选择镜像架构；ARM64 主机还需要 QEMU / binfmt 或其他能够执行 x86_64 程序的机制。容器本身不能解决 CPU 指令集差异。

第一轮可以将转换 worker 整体放入 amd64 环境，先验证 RKLLM 可用性。后续再评估原生 ARM64 RKNN 步骤与仿真 RKLLM 步骤的拆分价值。Box64 官方 README 确认其可在 ARM 等非 x86_64 Linux 主机上执行 x86_64 程序，并结合原生系统库和动态指令翻译；可作为另一个候选，但本次没有获得已验证的 RKLLM 转换兼容性证据。其其他应用的性能描述不能直接套用于 RKLLM。

应先采用 CPU 模式，重点核查：

- RKLLM、本机计算库与 PyTorch 对 CPU 指令特性和动态库的要求是否满足。
- 模型加载、量化、导出是否完成；import 成功不足以证明转换可用。
- 多线程、内存映射、进程退出与任务取消是否正常。
- 转换耗时与峰值内存是否符合目标 ARM64 主机的资源条件。
- 输出模型在对应 Rockchip 开发板上能否加载并正常推理。

维护成本：高且尚不确定。当前没有 ARM64 转换实测数据，不能给出确定的减速倍数、GPU 支持或可支持模型大小。若验证通过，应首先标记为实验支持。

### ARM64 控制界面配合 x86_64 远程转换

可复用现有 x86_64 worker，在 ARM64 主机上操作界面，通过 SSH 或任务服务进行转换。该路线能绕过 Toolkit 架构限制，但需要另一台计算机，不符合纯 ARM64 本地转换目标，作为备选方案保留。

最低成本的使用方式是 x86_64 主机启动现有工作台，ARM64 主机通过 SSH 端口转发访问。此时填写的模型与输出路径都属于 x86_64 主机。当前服务仅监听 `127.0.0.1` 且校验 Host / Origin，不能直接假定局域网访问已经可用。

若要做正式远程后端，还需处理文件传输、路径归属、结果获取、断线恢复和任务停止，并更新现有“模型文件不会上传”的产品说明。

维护成本：简单端口转发低；正式远程任务后端中等至高。

### 官方 ARM64 RKLLM 转换器

如果后续取得与 v1.3.1 或兼容版本对应的 ARM64 Toolkit wheel，可优先转为原生方案，复用现有双环境流程。

需要向官方核实是否存在未公开的 ARM64 转换包及其支持范围。本次未联系官方，也未确认私有 SDK 下载渠道中的内容。

## 逆向分析的初步可行性

进一步对内置 wheel 做了只读 ELF 静态检查，没有运行转换库：

- wheel 公开了少量 Python 文件，包括 `rkllm/api/rkllm.py`；关键实现仍位于二进制中。
- `rkllm_base`、`converter`、`quantizer` 的模块初始化导出包含 `__pyx_module_is_main_*`，表明使用了 Cython；检查的库均无 `.symtab` 和 `.debug_info` / `.debug_line` / `.debug_str`。这会增加恢复函数结构、变量含义和调用关系的难度。
- `librkllmc.so` 约 7.7 MB，无上述完整符号表或调试段，但仍有 517 个动态导出符号，其中包含 C++ 模板和运行库符号。这些可以提供部分分析线索，不能当成完整编译接口或源码。
- `ekml.so` 只有 14,432 字节，导出 `f32tofkl_ekml` 和 `fkltof32_ekml`，适合选作边界较小的数值转换分析对象；本次尚未验证这些函数的具体算法。

反汇编和反编译可以帮助理解局部实现、重建数据格式或编写行为兼容的代码，但不能自动恢复原始源码，也不能直接生成 ARM64 版本。完整替代需要重建模型适配、量化行为、目标硬件部署构建和导出格式，并验证板端 Runtime 的兼容性。当前静态检查无法估算完整重写的成本或确认能否成功。

若目标是尽快支持 ARM64，优先验证仿真执行现有转换器。逆向宜先限定在一个小型辅助库或明确的格式问题，依据分析和行为验证结果再扩大范围。

## 建议的下一步验证

1. 记录目标 ARM64 Linux 主机的 CPU、发行版、glibc、内存与可用磁盘；仿真结果必须对应具体硬件。
2. 在隔离的 x86_64 Linux 执行环境中验证 Python 3.10、CPU PyTorch 2.6、RKLLM 1.3.1 的加载和基本计算。
3. 选择官方支持的小型文本模型，使用固定校准数据、量化方式与参数执行完整转换；同时记录同配置 x86_64 基准。
4. 记录时间、峰值内存、日志、退出码和生成文件，再在对应 Rockchip 开发板上验证加载与推理。
5. 基础转换通过后检查取消任务、重复转换和失败清理；再验证完整视觉流程。
6. 根据实际结果决定增加仿真实验模式、先提供原生视觉功能，或等待官方 ARM64 RKLLM 工具。

在完成上述验证之前，项目支持说明宜维持“完整 RKLLM 转换要求 x86_64 执行环境”。

## 来源与调研边界

- [官方 RKLLM README](https://github.com/airockchip/rknn-llm/blob/main/README.md)：本次读取其 raw 内容，最新版本标注 v1.3.1，说明 Toolkit 与 Runtime 分工。
- [官方 RKLLM 变更记录](https://github.com/airockchip/rknn-llm/blob/main/CHANGELOG.md)：本次读取其 raw 内容。
- 项目附带 [RKLLM SDK 中文文档](Rockchip_RKLLM_SDK_CN_1.3.1.pdf)：第 10–12 页的 Toolkit wheel 与 Runtime 架构说明。
- 项目内置 RKLLM 1.3.1 CPython 3.10 wheel：本次读取 WHEEL 与全部 `.so` 的 ELF 头。
- [官方 RKNN Toolkit2 README](https://github.com/airockchip/rknn-toolkit2/blob/master/README.md)：本次读取其 raw 内容，区分 Toolkit2 与 Lite2 的职责。
- [RKNN Toolkit2 2.3.2 PyPI 元数据](https://pypi.org/pypi/rknn-toolkit2/2.3.2/json)：ARM64 发布文件、上传时间和依赖要求。
- [PyTorch 2.4.0](https://pypi.org/pypi/torch/2.4.0/json)、[PyTorch 2.6.0](https://pypi.org/pypi/torch/2.6.0/json)、[ONNX 1.18.0](https://pypi.org/pypi/onnx/1.18.0/json)、[ONNX Runtime 1.20.1](https://pypi.org/pypi/onnxruntime/1.20.1/json)、[AutoGPTQ 0.7.1](https://pypi.org/pypi/auto-gptq/0.7.1/json)：本次仅核查发布文件，未安装。
- [QEMU user-mode 官方文档](https://www.qemu.org/docs/master/user/main.html)：跨架构执行机制及线程、系统调用限制。
- [Box64 官方 README](https://github.com/ptitSeb/box64/blob/main/README.md)：本次重试后读取其 raw 内容，确认 ARM Linux 上的 x86_64 程序执行机制；未验证 RKLLM 兼容性。

GitHub API 的包目录和 issue 搜索请求在重试后仍发生连接重置或拒绝，因此本报告没有引用 issue 讨论来断言仿真成功或官方未来计划。官方 raw README 与 PyPI 元数据访问成功。公开资料与内置包的核查不能排除尚未公开的官方 ARM64 RKLLM 发布内容。
