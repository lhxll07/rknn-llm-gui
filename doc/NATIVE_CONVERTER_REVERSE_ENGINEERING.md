# RKLLM 转换器逆向记录

日期：2026-10-07。目标：尝试恢复 RKLLM 转换过程，探索无需 x86_64 官方 Toolkit 的 ARM64 Linux 转换。

## 当前成果与边界

已实现 [native_converter](../native_converter/README.md)：自行读取 Safetensors、映射 Llama 张量、转换 FP16 矩阵布局并写出 RKLLM 文件。转换阶段仅需 Python 和 NumPy；四组小模型的完整输出与仓库 Toolkit 1.3.1 一致。

NPU 程序从配置匹配的官方参考文件中提取，尚未自行编译。该原型可以更换同结构模型的全部权重，不能直接支持没有计划的新结构，也没有实现量化。单层和双层 BPE 小模型已通过 ARM64 真机转换和 RK3588 NPU 推理，见 [真机验证报告](NATIVE_CONVERTER_BOARD_VALIDATION.md)。网页工作台和安装器尚未接入该原型。

## 分析方法

先检查内置 wheel 的 ELF 架构、导出符号和 Cython 字符串，再用确定性随机小模型运行官方转换器，比较改变权重或模型尺寸后的输出。独立实现只依赖恢复出的目录和矩阵布局规则；官方包用于生成参考并验证行为。

仓库 wheel 中转换相关的 33 个 `.so` 均为 x86_64。关键 Cython 扩展和 `librkllmc.so` 没有完整符号表及调试段；后者保留 `rknn_llm_build` 等动态导出。`ekml.so` 保留 `f32tofkl_ekml` / `fkltof32_ekml`，但本原型没有重写或使用其量化算法。

## 外层文件格式

观察到的 FP16 文件采用类似 GGUF v3 的外层目录；头部魔数及每个字符串分别用固定 256 字节循环掩码 XOR，数值和权重负载不掩码。

```text
掩码处理的 "GGUF"
u32 version = 3
u64 tensor_count
u64 metadata_count
有类型的元数据
张量目录：名称、倒序维度、类型、相对偏移
32 字节对齐
Embedding 和归一化张量
"RLLM" NPU 程序块
打包后的 FP16 矩阵权重
"rkllm-toolkit version: 1.3.1"
```

矩阵的目录偏移不直接指向其实际数据。它们重复指向普通张量区的当前位置，矩阵负载集中放在 NPU 程序之后。将文件当作普通 GGUF 按目录偏移读取矩阵会得到错误数据。

程序块头部前五个小端 `u64` 为魔数 `0x4D4C4C52`、版本 `1` 和三个区段长度；在已观察样本中总长度为 `256 + 后三个长度之和`。偏移 256 处可解析出 FlatBuffers Model，包含算子、张量、地址寄存器和 JobSubmit 信息。尚未完整恢复各区段与硬件指令的语义。

后续已恢复访问器暴露的字段并实现标准库解析器，确认三段为模型描述、40 字节任务数组、64 位寄存器命令数组。此外建立了基于开源 iwagumi 的直接 NPU 矩阵路径，真实 Qwen 投影无需官方参考程序或用户态 Runtime，见 [NPU 阶段报告](NATIVE_NPU_COMPILER_RESEARCH.md)。

单层小模型中，三个长度为 34760、9440、198656，总程序长度为 243112 字节；FlatBuffers Model 有 10 个算子、34 个张量和 32 个地址寄存器项。程序生成路径先构造构建信息，再调用 `librkllmc.so` 的 `rknn_llm_build`。仓库 ARM `librknnrt.so` 的矩阵 API 导出不能直接替代这一入口。

## 权重映射与重排

| Safetensors 名称 | RKLLM 名称 |
| --- | --- |
| `model.embed_tokens.weight` | `token_embd.weight` |
| `model.layers.i.self_attn.q_proj.weight` | `blk.i.attn_q.weight` |
| `model.layers.i.self_attn.k_proj.weight` | `blk.i.attn_k.weight` |
| `model.layers.i.self_attn.v_proj.weight` | `blk.i.attn_v.weight` |
| `model.layers.i.self_attn.o_proj.weight` | `blk.i.attn_output.weight` |
| `model.layers.i.mlp.gate_proj.weight` | `blk.i.ffn_gate.weight` |
| `model.layers.i.mlp.up_proj.weight` | `blk.i.ffn_up.weight` |
| `model.layers.i.mlp.down_proj.weight` | `blk.i.ffn_down.weight` |
| `model.layers.i.input_layernorm.weight` | `blk.i.attn_norm.weight` |
| `model.layers.i.post_attention_layernorm.weight` | `blk.i.ffn_norm.weight` |
| `model.norm.weight` | `output_norm.weight` |
| `lm_head.weight` | `output.weight` |

逻辑矩阵形状为 `[N, K]`。Q 和 K 先按相应注意力头数做 RoPE 排列变换：

```python
matrix.reshape(heads, 2, head_dim // 2, K).transpose(0, 2, 1, 3).reshape(N, K)
```

随后所有矩阵转为小端 FP16，并按照观察到的 RK3588 单核布局打包：

```python
matrix.astype("<f2").reshape(N // 16, 16, K // 32, 32).transpose(0, 2, 1, 3)
```

Embedding 保持普通 FP16 数组，归一化权重为 FP32。当前没有为无法整除的 N/K 实现补齐；布局结论仅针对已验证的 FP16 单核路径。

## 对照证据

参考通过解压仓库精确的 `rkllm_toolkit-1.3.1-cp310-cp310-linux_x86_64.whl` 并隔离导入生成，没有使用本机已有的 1.3.0 包。转换在另一个只有 NumPy 的 `/usr/bin/python3` 进程中执行。每种结构只从第一组权重的官方结果提取计划，第二组权重复用同一计划。

| 隐藏维 / FFN 维 / 层数 | 种子 | 源类型 | 分片数 | 输出字节 | 完整文件一致 |
| --- | --- | --- | --- | --- | --- |
| 128 / 256 / 1 | 3407 | FP32 | 4 | 677860 | 是 |
| 128 / 256 / 1 | 20261007 | BF16 | 2 | 677860 | 是 |
| 256 / 768 / 2 | 3407 | FP32 | 14 | 3800508 | 是 |
| 256 / 768 / 2 | 20261007 | FP16 | 10 | 3800508 | 是 |

这覆盖了非恒定归一化权重，避免只用全 1 参数掩盖读取错误。不同权重产生不同输出，说明转换器确实替换了权重。另检查了保留已有输出、拒绝归一化参数变化和拒绝截断容器。原有工作台的 20 个测试在原型加入后通过。

上表为最初 WordLevel 样本的文件对照，完整 SHA256 见历史记录 [native-converter-verification.json](native-converter-verification.json)。这些文件虽然一致，官方参考也未能通过 Runtime 加载，因此不能据此声称 WordLevel 支持。

改用完整 ByteLevel BPE 后，四组新输出仍与官方一致；单层样本大小为 693604 字节，双层为 3832636 字节，完整结果见 [native-converter-bpe-verification.json](native-converter-bpe-verification.json)。单层 BF16 源权重、双层 FP16 源权重都完成了板端验证。转换器新增了缺少 GPT-2 merges 元数据的拒绝检查，验证程序新增对应回归检查，并正确统计 `RKLLM_RUN_WAITING` 状态的生成 token。

复现入口为 [tools/verify_native_converter.py](../tools/verify_native_converter.py)；需要 x86_64 的官方依赖环境与单独 NumPy Python：

```bash
/path/to/toolkit-dependencies/python tools/verify_native_converter.py \
  --portable-python /path/to/numpy-only/python \
  --report /path/to/verification.json
```

文件对照和小模型硬件验证尚未覆盖 ARM64 性能、实际大模型、Tokenizer 全部行为，也没有证明完整独立编译可行。随机小模型用来检查格式、布局和推理兼容性，不能用于评价生成质量。

## 下一阶段

RK3588 上的 NumPy 转换、文件一致性、Runtime 加载及 token 推理已完成；便携材料由 [prepare_native_board_bundle.py](../tools/prepare_native_board_bundle.py) 生成。NPU 执行证据与硬件环境见 [真机验证报告](NATIVE_CONVERTER_BOARD_VALIDATION.md)。下一步可扩展实际模型验证，并研究独立 NPU 编译。

进一步脱离参考计划需要恢复 NPU 算子构建、地址规划、目标指令和上下文布局。新形状、补齐、多核和量化仍待研究；这些能力不能由外层格式恢复或当前四组对照直接推出。后续已单独恢复指定 Qwen3.5-0.8B 的语言权重变换，并完成 ARM64 / NPU 实测，见 [Qwen3.5 验证报告](NATIVE_CONVERTER_QWEN35_VALIDATION.md)；它仍然复用配置匹配的官方 NPU 程序。
