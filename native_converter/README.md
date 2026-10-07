# 实验性 RKLLM 转换器

这个原型用 Python 和 NumPy 读取本地 Safetensors 权重，完成 Llama 或指定 Qwen3.5-0.8B 语言模型的张量映射、权重变换、RK3588 FP16 矩阵布局转换，以及 `.rkllm` 文件写入。转换过程离线运行，不导入或执行官方 RKLLM Toolkit，也不依赖 PyTorch、Transformers 或 Safetensors Python 包。

**NPU 指令编译器尚未重写。** 使用前需要从结构、分词器及构建参数匹配的官方 `.rkllm` 参考文件提取一份 `.rkplan`。计划包含元数据和 NPU 程序，不包含参考模型的权重。之后可转换相同结构、相同分词器的新权重，例如已合并的微调模型。新结构需要重新取得匹配计划。

后续 [native_npu](../native_npu/README.md) 提供另一条开放矩阵路径，由开源 iwagumi 按形状生成寄存器命令，不使用官方程序缓存或用户态 Runtime。它已经验证真实 Qwen 权重矩阵，但尚未实现完整模型执行，也未生成新的完整 RKLLM 程序。程序内部格式恢复及硬件证据见 [NPU 阶段报告](../doc/NATIVE_NPU_COMPILER_RESEARCH.md)。

## 当前支持范围

| 项目 | 范围 |
| --- | --- |
| 参考格式 | 官方 RKLLM Toolkit 1.3.1 |
| 目标设备 | RK3588，1 个 NPU 核心 |
| 模型 | 普通 `LlamaForCausalLM`，SiLU，独立 `lm_head`，无偏置；指定 Qwen3.5-0.8B 的语言部分 |
| 输出 | FP16；Embedding 为 FP16，归一化权重为 FP32 |
| 输入权重 | FP32 / FP16 / BF16 Safetensors，支持分片索引 |
| 矩阵尺寸 | 输出维 N 为 16 的倍数，输入维 K 为 32 的倍数；未实现补齐 |
| 分词器 | 必须有 `tokenizer.json`；已验证 ByteLevel BPE，完整词表与参考一致，GPT-2 元数据必须含 merges |
| RoPE | Llama 默认 RoPE；Qwen3.5-0.8B 的既定 partial mRoPE；参数须与计划相同 |
| 上下文 | 沿用参考构建值，32–16384 且为 32 的倍数 |
| 依赖 | Python 3.10+、NumPy 1.26+；见 `requirements.txt` |

转换主机已验证 Linux x86_64 和 RK3588 上的 ARM64 Linux；转换本身无需 NPU。原生 Windows 尚未适配验证，当前工作台的 Windows 安装方式仍为 WSL2。这个命令行原型尚未接入网页界面。

配置中的结构、RoPE、归一化参数和特殊 token ID 必须与计划一致。`tokenizer.json`、`tokenizer_config.json`、`special_tokens_map.json`、`generation_config.json` 的存在情况及文件哈希也必须一致；仅改变 JSON 排版也会触发拒绝。

Qwen 路径仅支持本次实测的 Qwen3.5-0.8B 配置，处理共享 Embedding / 输出权重以及混合注意力，明确排除视觉和 MTP 张量。准备计划时记录辅助张量的名称和形状，转换时拒绝不匹配的额外张量。Qwen 计划还校验完整 `config.json` 内容，以及 `chat_template.jinja`、`merges.txt`、`vocab.json` 的哈希。

不支持其他 Qwen 配置、完整多模态转换、W8A8/W4A16、RK3576、多 NPU 核心、直接加载 LoRA 适配器、Llama 共享 Embedding、Llama RoPE 缩放、滑动窗口或 `.bin` 权重。权重文件须位于模型目录内；当前不接受指向目录外的 HF 缓存符号链接。

## 使用

在项目根目录执行，或将整个 `native_converter/` 目录放在 Python 搜索路径中。可单独创建环境，无需运行工作台安装器：

```bash
python3 -m venv .venv-native
source .venv-native/bin/activate
python -m pip install -r native_converter/requirements.txt
```

先取得与本地模型配置和分词器一致的官方 **RK3588、单核、无量化、1.3.1** 参考文件。官方 API 的已验证构建参数为 `do_quantization=False`、`optimization_level=0`、`target_platform="rk3588"`、`num_npu_core=1`、`max_context=128`。首次生成该参考仍需能运行官方工具的主机；已有参考时无需重新运行 Toolkit。

一次性提取计划：

```bash
python -m native_converter prepare \
  --reference /path/to/reference-fp16.rkllm \
  --model /path/to/reference-hf-model \
  --output /path/to/llama.rkplan
```

将 `native_converter/`、计划及新模型目录放到转换主机上，再独立转换：

```bash
python -m native_converter convert \
  --plan /path/to/llama.rkplan \
  --model /path/to/matching-hf-model \
  --output /path/to/new-model.rkllm
```

检查外层目录：

```bash
python -m native_converter inspect /path/to/new-model.rkllm
```

`prepare` 与 `convert` 都拒绝覆盖已有输出。失败会清理本次创建的未完成文件。转换结果中的 `runtime_verified: false` 表示命令只生成文件，不执行板端验证。

## 已验证与待验证

2026-10-07 的对照使用仓库内准确的 Toolkit 1.3.1 wheel 作为参考，独立转换在只有 NumPy 的系统 Python 进程中执行。两种小模型配置、两组种子各自生成官方及独立输出，**四份完整文件逐字节一致**；覆盖 FP32、BF16、FP16、分片权重和非恒定归一化权重。不同权重的输出也确实不同。

验证记录：[逆向报告](../doc/NATIVE_CONVERTER_REVERSE_ENGINEERING.md)、[BPE 对照结果](../doc/native-converter-bpe-verification.json)。复现工具为 [verify_native_converter.py](../tools/verify_native_converter.py)，需要具备官方依赖的 x86_64 Python 3.10 环境；它只在对照阶段使用 Toolkit。

单层和双层 BPE 随机小模型已在 ROCK 5T / RK3588 上通过 ARM64 独立转换、Runtime 1.3.1 加载和 8 个 token 的贪心生成对照。系统调用跟踪确认 RKNPU 的 132 次成功任务提交，见 [真机验证报告](../doc/NATIVE_CONVERTER_BOARD_VALIDATION.md)。当前仍不能承诺支持任意 Llama 配置。

Qwen3.5-0.8B 原始 ModelScope 权重也已完成 ARM64 转换与真实 NPU 推理。SSM 的 `-exp(A_log)` 使用 Float64 计算后舍入为 FP32，288 个值中 12 个与 Torch/MKL 参考相差 1 ULP；其余权重和完整 NPU 程序一致。它还暴露了官方参考的 `default` 预分词标记问题：直接输入原始 token ID 可以恢复正确的算术结果。当前 `prepare` 会对已识别的源分词正则使用 `qwen2` 标记，旧的 `default` Qwen 计划会被拒绝。此修正与 FP32 舍入差异意味着完整文件不再与官方文件逐字节一致。验证经过和具体限制见 [Qwen3.5 报告](../doc/NATIVE_CONVERTER_QWEN35_VALIDATION.md)。这不是完整质量评测。

初始 WordLevel 样本的官方输出缺少 GPT-2 分词器 merges 元数据，Runtime 无法加载；当前转换器会拒绝这种参考及旧计划。验证程序已修正 UTF-8 等待状态的 token 统计，ByteLevel BPE 的等待事件也计入生成结果。

## RK3588 板端验证材料

本次研究的本地小模型、参考及计划保存在 `gui/runs/native-converter-research/bpe-board-fixtures/`，被 Git 忽略。它们使用随机权重，适合验证兼容性，生成内容不具有对话能力。默认验证包使用 `h128-f256-l1-seed20261007` 样本，计划来自同结构的另一组权重。

新检出项目没有这些本地材料时，在已有官方依赖的 Python 环境中生成并保留 BPE 样本：

```bash
/path/to/toolkit-dependencies/python tools/verify_native_converter.py \
  --portable-python /path/to/numpy-only/python \
  --keep-fixtures gui/runs/native-converter-research/bpe-board-fixtures \
  --report gui/runs/native-converter-research/bpe-verification.json
```

准备便携验证包，默认从上述本地目录读取材料，也可指定自己的匹配模型、计划、参考：

```bash
python3 tools/prepare_native_board_bundle.py \
  --output gui/runs/native-converter-research/rkllm-native-board-check.tar.gz
```

在 RK3588 Linux 上解压后执行：

```bash
tar -xzf rkllm-native-board-check.tar.gz
cd rkllm-native-board-check
bash run-board-check.sh
```

开发板需已有 Python 3.10+、NumPy 和可用 RKNPU 驱动。默认包从源码构建验证程序，还需 `g++`；可在准备包时通过 `--smoke-binary` 提供预先交叉编译的 AArch64 程序。包自带仓库中的 Runtime 和对应头文件，脚本不安装系统依赖。

预编译程序的 glibc / libstdc++ 版本需要与开发板兼容；在板端从源码构建可避免本机交叉编译环境过新的问题。

脚本创建独立结果目录，在板端重新转换并与官方文件比较，然后分别加载参考文件和新文件，用 token ID `[4, 5]` 做贪心生成，检查完成状态、token 范围、数量及一致性。退出码为 0 时写出 `board-report.json`；失败日志保留在结果目录。此验证仍使用官方 Runtime，转换 Toolkit 不参与板端执行。
