# 独立转换器的 RK3588 真机验证

验证日期：2026-10-07。结果：**单层和双层 BPE 随机 Llama 模型均完成 ARM64 独立转换、Runtime 加载和 NPU 推理对照。** 转换阶段仅运行 Python 和 NumPy；NPU 程序仍由匹配参考文件提供，推理阶段使用官方 Runtime。

## 硬件及环境

| 项目 | 实测值 |
| --- | --- |
| 开发板 | Radxa ROCK 5T，RK3588 |
| 系统 | Armbian 26.8.3，Debian 13 / trixie |
| 架构、内核 | aarch64，6.1.115-vendor-rk35xx |
| Python / NumPy | 3.13.5 / 2.3.4 |
| glibc | 2.41 |
| RKLLM Runtime / RKNPU 驱动 | 1.3.1 / 0.9.8 |
| 模型目标 | FP16，1 个 NPU 核心，最大上下文 128 |

系统原本没有 NumPy 或编译器。验证使用独立目录、无 pip 的 Python venv、解压到该环境中的官方 NumPy ARM64 wheel，以及本机交叉编译的 AArch64 验证程序。NumPy、Runtime 和系统调用工具 strace 均放在独立验证目录中。

## 验证步骤与结果

对每个结构，仅用种子 3407 的官方结果提取计划；板端转换种子 20261007 的不同权重。开发板没有导入或执行官方转换 Toolkit。

1. 板端读取 Safetensors，独立生成 `.rkllm`。
2. 使用 `cmp` 检查整个文件与对应官方参考一致。
3. 依次加载官方文件及板端新文件，输入 token ID `[4, 5]`。
4. 使用 `top_k=1`、重复惩罚 1、忽略 EOS，分别生成 8 个 token；检查成功返回、完成回调、token 范围和两次生成一致。

| 结构：隐藏维 / FFN 维 / 层数 | 源权重 | 文件大小 | ARM64 文件一致 | Runtime 加载 | 生成一致 |
| --- | --- | --- | --- | --- | --- |
| 128 / 256 / 1 | BF16，2 个分片 | 693604 字节 | 是 | 是 | 是，8 token |
| 256 / 768 / 2 | FP16，10 个分片 | 3832636 字节 | 是 | 是 | 是，8 token |

单层 token 序列：`[154, 97, 123, 231, 20, 5, 113, 148]`。

双层 token 序列：`[20, 44, 33, 33, 29, 71, 39, 29]`。

完整 SHA256 和板端返回值见 [native-converter-board-verification.json](native-converter-board-verification.json)；四组 x86_64 BPE 对照见 [native-converter-bpe-verification.json](native-converter-bpe-verification.json)。

## 真实 NPU 调用证据

对单层参考与新文件的两次推理执行 `strace -f -yy -X raw -e trace=openat,ioctl`。Runtime 使用 `/dev/dri/card1`；其 sysfs 驱动链接为 `/sys/bus/platform/drivers/RKNPU`。

跟踪记录包含该设备的 606 次 ioctl，其中 **132 次 `0xc0686441` 请求成功返回 0**。该请求的类型为 DRM 的 `0x64`，命令编号为 `0x41`，结构大小为 104 字节；对应 RKNPU 的任务提交入口 `DRM_IOCTL_RKNPU_SUBMIT`，用于提交 NPU 计算任务。

命令定义可核对 [Rockchip 内核 rknpu_ioctl.h](https://github.com/rockchip-linux/kernel/blob/develop-6.1/drivers/rknpu/include/rknpu_ioctl.h)：`RKNPU_SUBMIT = 0x01`，通过 `DRM_COMMAND_BASE + RKNPU_SUBMIT` 构建请求。实际跟踪示例：

```text
ioctl(3</dev/dri/card1<char 226:1>>, 0xc0686441, ...) = 0
```

因此已有实际 NPU 提交证据。Runtime 同时启用 CPU 4–7 做辅助处理，日志中的 CPU 数量不等于模型使用的 NPU 核心数。

## 真机暴露的问题与修复

初始 WordLevel 随机样本的官方结果与独立输出一致，但两者都无法被 Runtime 加载。它的元数据将分词器标记为 GPT-2，却没有 `tokenizer.ggml.merges`。改用完整 ByteLevel BPE 后加载成功；当前转换器会提前拒绝缺失该字段的参考与旧计划。

修复后的完整验证包已在板端再次通过。旧计划的板端校验返回退出码 1，并确认没有创建输出文件。

ByteLevel BPE 还会生成未拼完整 UTF-8 字符的 token，此时回调状态为 `RKLLM_RUN_WAITING`。初版验证程序只统计 `RKLLM_RUN_NORMAL`，误报生成数量不足；修复后两种状态都计入 token，对照通过。

## 适用边界

本报告中的随机小模型用于验证兼容性，尚未覆盖实际预训练模型的效果或性能。这里验证的能力为 RK3588 单核 FP16 Llama 的计划复用、权重转换和 Runtime 推理。后续 Qwen3.5-0.8B 的独立验证见 [Qwen3.5 报告](NATIVE_CONVERTER_QWEN35_VALIDATION.md)；量化、多核、新结构的完整 NPU 编译及网页界面集成仍未实现。
