# 实验性 RK3588 开放矩阵后端

这个目录通过 Python + NumPy 调用开源 [iwagumi](https://github.com/fukumori/iwagumi) 的 C API，直接在 RK3588 上执行 FP16 矩阵乘法。iwagumi 从矩阵形状、内存地址和布局生成寄存器命令，再调用内核 RKNPU DRM 接口。运行时不使用官方 Toolkit、RKLLM/RKNN Runtime 或提取的 `.rkplan`。

**这是基础算子后端，尚未成为完整语言模型运行时。** 指令编码和内核通信由 iwagumi 提供，不能把这些开源成果说成本项目独立重写的完整编译器。仍需要板上可用的 RKNPU 内核驱动。

固定上游版本：`8fafdbd28cce6eabb0ec2072bb549beb1035a141`，Apache-2.0。上游源代码放在被 Git 忽略的研究目录，未复制进本项目；构建包保留上游 LICENSE 和 NOTICE。

## 当前接口与实测

`OpenNPU.matmul(A, W)` 接收 `A[M,K]` 和 `W[N,K]`，返回 `A @ W.T` 的 FP32 结果。输入转为 FP16，并按输出通道分块，默认每块 2048 列。错误或不支持的形状直接报错，验证程序不会把 CPU 回退当成 NPU 成功。

2026-10-07 在 ROCK 5T / RK3588、内核 6.1.115、RKNPU 0.9.8 上验证：

- 3 组随机小矩阵，M 为 1 / 2 / 16。
- Qwen3.5-0.8B 的 QKV、FFN gate、FFN down、完整注意力 Q 投影，每种 M 为 1 / 16。
- 11 组均通过，与相同 FP16 操作数的 CPU FP32 累加对照，最大绝对误差约 `2.38e-7`。
- 跟踪确认 17 次成功 `RKNPU_SUBMIT`，没有打开官方 Runtime；进程映射也没有官方 Runtime。

这里使用真实模型权重和合成输入激活。M=1 证明单 token 对应的矩阵运算可以直接提交 NPU，不代表已经完成整模型的逐 token 解码。详见 [阶段报告](../doc/NATIVE_NPU_COMPILER_RESEARCH.md) 和 [板端记录](../doc/native-npu-board-verification.json)。

## 构建与使用

准备固定版本的开源引擎：

```bash
git clone https://github.com/fukumori/iwagumi.git /path/to/iwagumi
git -C /path/to/iwagumi checkout --detach 8fafdbd28cce6eabb0ec2072bb549beb1035a141
```

在 ARM64 Linux 上用已有 CMake、C 编译器和 libdrm 头文件构建。输出目录必须尚不存在：

```bash
python3 tools/build_open_npu.py \
  --source /path/to/iwagumi --output /path/to/new-open-npu-bundle
```

也可使用现有 AArch64 交叉编译器；`--drm-include` 指向含 `drm/drm.h` 的目录：

```bash
python3 tools/build_open_npu.py \
  --source /path/to/iwagumi --output /path/to/new-open-npu-bundle \
  --cross-compiler aarch64-linux-gnu-gcc --drm-include /usr/include
```

将构建包、项目的 `native_npu/`、`native_converter/`、验证脚本和原始模型目录放到开发板。依赖仅为 Python 3.10+、NumPy 和构建出的开放引擎；不需要官方转换环境。按自己的目录设置库路径和锁文件：

```bash
export LD_LIBRARY_PATH=/path/to/new-open-npu-bundle/lib
export NPU_LOCK_PATH=/path/to/writable-directory/npu.lock
python3 tools/probe_open_npu.py \
  --library /path/to/new-open-npu-bundle/lib/libiwagumi.so.1 \
  --model /path/to/Qwen3.5-0.8B --report /path/to/new-probe-report.json
```

省略 `--model` 时仅验证 3 组小矩阵。所有数值对照通过且未映射官方 Runtime 才写出报告。`successful_matmul_submits` 是调用成功数，实际驱动提交次数需要配合 `strace` 核对。

从 Python 直接使用：

```python
from pathlib import Path
import numpy as np
from native_npu.backend import OpenNPU

with OpenNPU(Path("/path/to/libiwagumi.so.1")) as npu:
    a = np.ones((1, 1024), dtype=np.float16)
    w = np.ones((2048, 1024), dtype=np.float16)
    output = npu.matmul(a, w)
```

当前原型只提供 FP16 API。K 必须为 32 的倍数，N 为 16 的倍数，单块 N 不超过 4096，K 不超过 8192，M×K 不超过 131072。条件检查依赖固定上游的硬件边界，但本次实际形状覆盖有限。

每次调用仍重新分配设备缓冲并上传权重，没有常驻权重缓存、跨调用执行计划缓存、K 分块累加、完整模型调度或整模型推理速度结论。这些是下一阶段的主要工作。
