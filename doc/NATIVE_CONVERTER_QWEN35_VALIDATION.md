# Qwen3.5-0.8B 独立转换与 RK3588 验证

日期：2026-10-07。源模型为用户提供的 [ModelScope Qwen/Qwen3.5-0.8B](https://www.modelscope.cn/models/Qwen/Qwen3.5-0.8B)。

## 结果

已使用 Python + NumPy 独立生成语言部分的 RKLLM 文件，并在 Radxa ROCK 5T / RK3588 上重新转换、加载和推理。最终 ARM64 文件与本机生成文件的 SHA256 相同，普通文本输入的中文问答和两条算术探针通过。最终算术推理的系统调用跟踪记录了 **495 次成功 RKNPU 任务提交**。

转换不运行 RKLLM Toolkit、PyTorch 或 Transformers；推理仍使用官方 Runtime 1.3.1。NPU 指令编译器仍未实现，计划复用准确配置的官方参考程序。首次生成参考文件使用仓库的 Toolkit 1.3.1 x86_64 wheel。

| 项目 | 本次结果 |
| --- | --- |
| 原始权重 | BF16 为主的 Safetensors，1,746,942,600 字节 |
| 权重 SHA256 | `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` |
| 模型 | 24 层；18 层线性注意力、6 层完整注意力；隐藏维 1024 |
| 输出 | RK3588、1 个 NPU 核心、FP16，最大上下文 128 |
| 文件大小 | 2,030,366,996 字节，约 1.89 GiB |
| 输出张量目录 | 327 项 |
| 缓存 NPU 程序 | 3,882,712 字节 |
| 最终文件 SHA256 | `c8e36d3439330e301238e1c00f83858c8699d89d1b503f962be69450a4afcfc0` |
| ARM64 转换耗时 | 本次记录 19.879 秒；权重已在系统缓存中，未进行完整性能评测 |
| 开发板 | aarch64，内核 6.1.115-vendor-rk35xx，Python 3.13.5，NumPy 2.3.4 |
| 推理环境 | Runtime 1.3.1，RKNPU 驱动 0.9.8 |

完整文件与官方参考有两类明确差异：预分词类型的修正，以及 12 个 SSM FP32 指数值的 1 ULP 舍入差异。其余张量数据、矩阵布局、NPU 程序和文件尾一致。

## 实现

Checkpoint 共 488 个张量：320 个语言张量、153 个视觉张量、15 个 MTP 张量。只转换语言部分。视觉和 MTP 张量按明确的名字前缀排除，其名称、形状被记入计划并在转换时核对，未知额外张量会被拒绝。

[qwen35.py](../native_converter/qwen35.py) 实现专用配置检查、名字映射、源形状检查和变换。现有 Llama 路径继续使用原本的规则。

| 源张量 / 操作 | RKLLM 处理 |
| --- | --- |
| `model.language_model.embed_tokens.weight` | FP16 Embedding；也作为共享输出矩阵的源数据 |
| 完整注意力 `q_proj.weight` `[4096,1024]` | 按 8 个头，每头 256 行 Q + 256 行 gate，拆成两个 `[2048,1024]` 矩阵 |
| 完整注意力 Q / K | 保持源行序；不使用 Llama 的 Q/K 重排 |
| 输入、FFN、最终、Q/K RMSNorm | FP32 转换后加 1 |
| 线性注意力 gated RMSNorm | 保留原始权重，不加 1 |
| `linear_attn.A_log` | `-exp(A_log)`，Float64 指数计算后舍入为 FP32 |
| `linear_attn.conv1d.weight` | `[6144,1,4]` 去掉中间维，普通 FP32 存储 |
| `in_proj_a` / `in_proj_b` | 对应 `ssm_alpha` / `ssm_beta`，普通 FP32 矩阵 |
| QKV / Z / out / FFN / 输出投影 | FP16 NPU 矩阵，布局为 `N/16, K/32, 16, 32` |

参考中的 288 个 `ssm_a` 数值有 12 个与 Float64 指数舍入结果相差 1 ULP。直接调用 NumPy FP32 `exp` 会有更多差异，因此使用 Float64 路径。没有在计划中缓存这些参考权重来伪造逐字节一致。完整载荷逐字节对照工具仅允许这 12 类 SSM 位置的至多 1 ULP 变化，其他载荷变化会报错。

## 定位并修正预分词标记

官方 1.3.1 导出的 `tokenizer.ggml.pre` 为 `default`。源 tokenizer 的正则按单个数字切分；Runtime 的默认预分词行为与之不同。

| 输入 | 原始模型 CPU / FP32 | 官方 RKLLM 文本输入 | 最终独立 RKLLM 文本输入 |
| --- | --- | --- | --- |
| 法国首都，只回答城市 | 未做 CPU 对照 | 巴黎 | 巴黎 |
| `17 + 25`，只输出结果 | 42 | **32** | 42 |
| `1 加 1`，只输出数字 | 2 | 2 | 2 |

早期独立文件沿用 `default`，它与官方参考在以上三个探针的 token 序列完全一致，也同样答错 `17 + 25`。随后把原始分词器生成的完整聊天 token 序列直接送入独立文件的 NPU 推理，得到 `[19,17,248046]`，即 `42` + EOS。这将问题定位到内置文本分词路径。

当前 `prepare` 对明确识别的 Qwen3.5 源预分词配置，把计划中的 `tokenizer.ggml.pre` 修正为 `qwen2`；普通文本算术输入随即恢复 `42`。`convert` 拒绝早期含 `default` 的 Qwen 计划，避免继续生成已知有问题的模型。原始文件未被覆盖。

该修正仅经过本次中文与数字探针验证，尚未做完整 tokenizer 等价性评测，尤其不能据此保证组合音标、所有 Unicode 边界或多模态特殊 token 的分词行为完全等价。

## 真实 NPU 证据

跟踪使用独立解包的 `strace`，未安装系统软件：

```bash
strace -f -yy -X raw -e trace=openat,ioctl -o corrected-arithmetic.trace \
  ./rkllm_text_probe_v2 native-qwen2.rkllm 128 248320 \
  '计算 17 + 25，只输出结果。'
```

最终文件的这次推理记录了 `/dev/dri/card1` 上 495 次 `ioctl(..., 0xc0686441, ...) = 0`。请求号对应 `DRM_IOCTL_RKNPU_SUBMIT`，设备驱动为 RKNPU。这说明运行实际提交了 NPU 任务；模型的其他部分仍可能使用 CPU。

[rkllm_text_probe.cpp](../tools/rkllm_text_probe.cpp) 收集普通与 UTF-8 等待状态的 token，检查初始化、推理、结束、销毁和 token 范围。可输入文本，也可用 `tokens:ID,ID,...` 输入完整聊天序列，后者会清空 Runtime 自动模板。

## 本次材料与复现

工作目录为 `gui/runs/native-converter-research/`，大文件由 Git 忽略：

- `qwen35-0.8b-modelscope/`：下载的原始模型，含 ModelScope 文件哈希清单。
- `qwen35-0.8b-rk3588-fp16-reference.rkllm`：未经修改的官方参考。
- `qwen35-0.8b-rk3588-fp16-qwen2.rkplan`：最终修正计划，不含模型权重。
- `qwen35-0.8b-rk3588-fp16-native-qwen2.rkllm`：最终独立转换结果。
- `qwen35-board-evidence.tar.gz`：板端日志、报告、系统调用跟踪。

板端最终模型位于 `/home/lhx/rkllm-qwen35-check-f74cS19p/native-qwen2.rkllm`。实验放在独立目录内，既有用户模型和系统依赖未被覆盖。

下载新快照时必须使用尚不存在的目录；下载器核对每个文件的大小与 SHA256：

```bash
python3 tools/download_modelscope_snapshot.py Qwen/Qwen3.5-0.8B \
  --output /path/to/new-qwen35-model
```

有官方依赖的 x86_64 Python 3.10 环境先生成参考。脚本将仓库内准确的 1.3.1 wheel 解包到临时目录并从该目录导入，使用 CPU 加载、无量化、优化等级 0、RK3588 单核和上下文 128：

```bash
/path/to/toolkit-dependencies/python tools/export_qwen35_reference.py \
  --model /path/to/new-qwen35-model --output /path/to/new-reference.rkllm
```

其后 `prepare` 和 `convert` 仅需 NumPy，亦可在 ARM64 执行：

```bash
python3 -m native_converter prepare \
  --reference /path/to/new-reference.rkllm --model /path/to/new-qwen35-model \
  --output /path/to/new-qwen35.rkplan
python3 -m native_converter convert \
  --plan /path/to/new-qwen35.rkplan --model /path/to/new-qwen35-model \
  --output /path/to/new-qwen35.rkllm
python3 tools/compare_qwen35_exports.py \
  --reference /path/to/new-reference.rkllm --native /path/to/new-qwen35.rkllm \
  --report /path/to/new-comparison.json
```

报告：[文件和 CPU 对照](native-converter-qwen35-verification.json)。此报告还包含保存的板端验证数据。

## 限制

仅验证这个原始 checkpoint 和配置，不代表其他 Qwen3.5 尺寸或微调权重已完成验证。没有完整指令编译器，计划仍来自官方参考；推理 Runtime 也仍为官方二进制。此次没有转换视觉 RKNN、实现 MTP、量化、多核、扩大上下文或接入工作台界面。三条短探针用于验证兼容性和已发现的分词问题，不能替代完整精度、长文本和性能评测。
