# NPU 编译与开放运行路径：第一阶段

日期：2026-10-07。目标是进一步减少官方依赖，将研究重点从权重文件转向 NPU 命令生成与执行。

## 已完成的两部分

1. 恢复 RKLLM 程序内部的 FlatBuffers 字段、任务结构和寄存器命令格式，提供不导入 SDK 的独立解析器。
2. 基于开源 iwagumi 建立直接矩阵后端，在同一块 RK3588 上执行真实 Qwen 权重投影，不使用 Toolkit、官方用户态 Runtime 或缓存 RKLLM 程序。

第二部分是新的开放运行路径；现有 `.rkllm` 转换流程仍复用官方参考程序并使用官方 Runtime 推理。此次没有完成整个 Qwen 模型的开放运行时，也没有声称重写了完整 RKLLM 编译器。

## 程序格式恢复

[program.py](../native_converter/program.py) 使用 Python 标准库读取观察到的 RKLLM 1.3.1 程序，支持从 `.rkplan`、FP16 `.rkllm` 或独立 `program.bin` 读取。源码包含范围检查，输出模式只用于分析，不执行解析得到的命令。

外部结构为：

```text
256 字节 RLLM 头
  u64 magic = 0x4d4c4c52
  u64 version = 1
  u64 model_bytes
  u64 task_bytes
  u64 regcmd_bytes
FlatBuffers Model（标识 RKRK）
40 字节驱动任务数组
64 位寄存器命令数组
```

后两段语义由实际内容确认。每个任务匹配内核 `rknpu_task` 的八个 uint32 字段和一个 uint64 命令地址字段；文件中的命令地址为相对偏移，加载执行时需要分配、重定位。任务中可以检查使能掩码、寄存器数量、命令偏移和中断配置。

每个寄存器命令的观察布局是：

```text
bits 63..48: target
bits 47..16: value
bits 15..0 : register
```

例如地址表的 A、B、C 重定位项对应 CNA `0x1070`、CNA `0x1110`、DPU `0x4020`。地址表中的命令偏移以 64 位 word 为单位。这些只是已观察的命令字段；target 中的全部控制位与所有寄存器语义尚未完整恢复。

字段与类型通过观察官方生成访问器和构建器的调用恢复；完整 schema 记录在 [rkllm_program.fbs](../native_converter/rkllm_program.fbs)。解析结果逐项对照 SDK 暴露的字段，包括 Model、Operator、MatMulParams、Tensor、JobSubmit、地址结构和 DomainManage。标准库解析也在关闭 site-packages 的 `python3 -S` 下通过，解析阶段不需要 Toolkit 或 FlatBuffers 包。

| 程序 | 算子 | 张量 | 地址表 | 驱动任务 | 寄存器 words |
| --- | --- | --- | --- | --- | --- |
| 既有 tiny Llama | 10 | 34 | 32 | 236 | 24832 |
| Qwen3.5-0.8B FP16 | 159 | 481 | 479 | 4393 | 392264 |

Qwen 程序的三段长度分别为 568624、175720、3138112 字节，总长 3882712 字节。这里的 NPU 算子数量不等于模型层数，也不意味着所有模型操作都在 NPU 上执行。

对照记录：[native-npu-program-verification.json](native-npu-program-verification.json)。复现解析：

```bash
python3 -S tools/inspect_npu_program.py /path/to/model.rkplan \
  --output /path/to/new-program-report.json
```

`--commands` 可额外导出每个寄存器命令。此次字段恢复和解析只证明能读懂结构，并不等于已能生成完整、可被官方 Runtime 接受的新模型程序。

## 绕过官方编译器的矩阵路径

研究中找到两个开源实现：

- [mtx512/rk3588-npu](https://github.com/mtx512/rk3588-npu)：较早的 RK3588 逆向与矩阵乘法实现，README 实测内核为 5.10。
- [fukumori/iwagumi](https://github.com/fukumori/iwagumi)：面向 vendor 6.1.115 / RKNPU 0.9.8 的开放引擎，包含矩阵布局、指令编码、设备内存和提交代码。本次选择它，因为内核与用户开发板一致。

固定使用 iwagumi 提交 `8fafdbd28cce6eabb0ec2072bb549beb1035a141`，Apache-2.0。上游源代码保存于被忽略的研究目录，构建包包含其 LICENSE / NOTICE。本项目添加了 [native_npu/backend.py](../native_npu/backend.py)、可复现构建脚本和 Qwen 投影验证脚本；底层指令编码与驱动通信应归功于 iwagumi。

本次实际运行流程：

```mermaid
flowchart LR
    A[Safetensors 权重和输入激活] --> B[NumPy 读取与 FP16 转换]
    B --> C[OpenNPU：输出通道分块]
    C --> D[iwagumi：布局、寄存器编码、内存重定位]
    D --> E[RKNPU DRM ioctl]
    E --> F[RK3588 NPU]
    F --> G[FP32 矩阵结果]
```

不需要 `.rkllm` 文件或官方程序缓存。构建后的 `libiwagumi.so` ELF 依赖只有 `libm.so.6`、`libgomp.so.1`、`libc.so.6`，没有链接官方用户态 Runtime。内核仍使用现有 RKNPU 驱动。

## 实机数值与执行证据

开发板仍为 ROCK 5T / RK3588，内核 `6.1.115-vendor-rk35xx`。实验使用新目录 `/home/lhx/rk3588-open-npu-LO8vDpfz/`，未安装系统软件。交叉编译只构建开源引擎及工具。

先运行上游小矩阵示例，FP16 和 W8A8 均通过数值检查，跟踪记录 2 次成功 NPU 提交。

随后运行本项目的 [probe_open_npu.py](../tools/probe_open_npu.py)，以真实 Qwen3.5-0.8B 权重和随机输入激活检查：

| 权重 / 输入 | K | N | M |
| --- | --- | --- | --- |
| 随机小矩阵 | 64 | 64 | 1、2、16 |
| 第 0 层线性注意力 QKV 投影 | 1024 | 6144 | 1、16 |
| 第 0 层 FFN gate | 1024 | 3584 | 1、16 |
| 第 0 层 FFN down | 3584 | 1024 | 1、16 |
| 第 3 层完整注意力 Q 投影，按头拆出 Q | 1024 | 2048 | 1、16 |

共 11 组全部通过，N 按每块 2048 分块，共 17 次成功矩阵提交。`strace` 在 `/dev/dri/card1` 上确认 17 次 `ioctl(..., 0xc0686441, ...) = 0`，与调用成功数一致。进程映射及文件打开跟踪均没有 `librknnrt` / `librkllmrt`。

FP16 操作数先确定，然后以 CPU FP32 累加作参考。容差为 `atol=5e-4, rtol=2e-3`，实际最大绝对误差 `2.384185791015625e-7`。这个对照检查打包、命令生成和硬件计算，不评价原始 FP32 模型变成 FP16 后的整体精度。

M=1 的成功说明单 token 对应的矩阵可以运行在开放 NPU 路径；目前没有把它组装成完整语言模型逐 token 解码。上游 GGML 适配器的默认 M=1 CPU 路由是另外一层策略，本次使用独立矩阵 API，没有通过该适配器。

报告：[native-npu-board-verification.json](native-npu-board-verification.json)。原始证据包为被忽略的 `gui/runs/native-converter-research/open-npu-board-evidence.tar.gz`。本次跟踪下 Qwen 矩阵调用约 40–124 ms，含每次内存分配、权重转置、打包、上传和跟踪开销；它不代表整模型速度，也不构成比 CPU 更快的证据。

## 依赖减少到了哪里

| 路径 | 权重处理 | NPU 程序 | 用户态执行 | 当前状态 |
| --- | --- | --- | --- | --- |
| 已有独立 RKLLM 转换 | 本项目 | 复用官方参考 | 官方 RKLLM Runtime | 已验证整个 0.8B 语言模型 |
| 新开放矩阵后端 | 本项目 | 开源引擎按形状生成 | 开源 iwagumi + 内核驱动 | 已验证真实权重矩阵，尚未运行完整模型 |

源码与构建步骤见 [native_npu/README.md](../native_npu/README.md)。此次得到一个可复现、无需官方程序缓存的 NPU 基础层。

## 下一阶段的工程重点

首先实现常驻 NPU 权重与可复用的执行计划，避免逐次分配、重排和上传；否则基础运算正确也无法说明整模型加速。再用 Qwen 的一个完整 block 验证逐层结果：矩阵留在开放 NPU 后端，RMSNorm、RoPE、Softmax、卷积与 Gated DeltaNet 状态更新先用独立 CPU 实现，记录所有跨设备拷贝。

完整 block 对照通过后，再做 24 层调度、KV/线性注意力状态缓存、分词与采样，以及预填充/单 token 解码的精度和速度对照。量化、多核和更多算子融合属于后续优化。这条路线能够逐步替代官方用户态执行，但本次还未完成这些阶段。
