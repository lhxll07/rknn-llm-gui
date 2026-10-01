# multimodal model demo
1. This demo demonstrates how to deploy multimodal model. The Vision + Projector component and the Audio encoder are exported as RKNN models using the `rknn-toolkit2`, while the LLM component is exported as an RKLLM model using the `rkllm-toolkit`.
2. The open-source models used in this demo are available at: [Qwen2-VL-2B](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct), [Qwen2-VL-7B](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct), [Gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it). 

## 1. Requirements
```text
rkllm-toolkit>=1.3.0
rknn-toolkit2>=2.3.2
```

rknn-toolkit2 installation guide：

pip install rknn-toolkit2 -i https://mirrors.aliyun.com/pypi/simple

## 2. HuggingFace Demo

```bash
# modify the modelpath in infer.py
cd examples/multimodal_model_demo
python infer.py
# expect results:
["The image depicts an astronaut in a white spacesuit, reclining on a green chair with his feet up. He is holding a green beer bottle in his right hand. The astronaut is on a moon-like surface, with the Earth visible in the background. The scene is set against a backdrop of stars and the moon's surface, creating a surreal and whimsical atmosphere."]
```

## 3. Model Conversion
- ### convert to onnx

1. Export the Vision + Projector component of the Qwen2-VL model to an ONNX model using the `export/export_vision_qwen2.py` script.

2. Since RKNN currently supports only `float32`, if the data type is restricted when loading weights, you need to set the `"use_flash_attn"` parameter in `config.json` to `false`.

```bash
# First time to generate cu_seqlens and rotary_pos_emb, need to set 'step' to 1
python export/export_vision_qwen2.py --step 1 --path /path/to/Qwen2-VL-model --batch 1 --height 392 --width 392
# Second time to export onnx model, need to set 'step' to any number except 1
python export/export_vision_qwen2.py --step 0 --path /path/to/Qwen2-VL-model --savepath /path/to/save/qwen2-vl-vision.onnx --batch 1 --height 392 --width 392
```

Note: if change the batch, height and width, the cu_seqlens and rotary_pos_emb must be re-generated.

3. If you want to convert models such as Qwen2.5-VL, Qwen3-VL, MiniCPM-V-2_6, SmolVLM, DeepSeekOCR, Qwen3.5 or InternVL3, please use the script export/export_vision.py instead.

4. When converting the DeepSeekOCR vision model to onnx, you need to set the `antialias` attribute to `False` for all `F.interpolate` calls in the `deepencoder.py` file.

```bash
1、pip install onnx==1.18.0
2、python export/export_vision.py --path=/path/to/DeepSeek-OCR --model_name=deepseekocr --height=448 --width=448
3、python export/export_vision_rknn.py --path=./onnx/deepseekocr_vision.onnx --model_name=deepseekocr --height=448 --width=448
```

5. The code for converting Qwen3-VL vision to RKNN is as follows.

```bash
1、pip install transformers==4.57.0
2、python export/export_vision.py --path=/path/to/Qwen3-VL --model_name=qwen3-vl --height=448 --width=448
3、python export/export_vision_rknn.py --path=./onnx/qwen3-vl_vision.onnx --model_name=qwen3-vl --height=448 --width=448
```

6. The code for converting Gemma-4-E2B-it vision to RKNN is as follows.
```bash
1、pip install transformers==5.8.0 onnx==1.22.0
2、python export/export_vision_gemma4.py --model_path /path/to/gemma-4-E2B-it --export_vision_path ./onnx/gemma-4-E2B-it-vision.onnx
3、python export/export_vision_rknn.py --path=./onnx/gemma-4-E2B-it-vision.onnx --model_name=gemma-4 --height=448 --width=448
```

7. The code for converting audio tower + projector of the Gemma-4-E2B-it model to RKNN is as follows.

```bash
1、pip install transformers==5.8.0 onnx==1.22.0
2、python export/export_audio_gemma4.py --model_path /path/to/gemma-4-E2B-it --export_audio_path ./onnx/gemma-4-E2B-it-audio.onnx
3、python export/export_audio_rknn.py --path ./onnx/gemma-4-E2B-it-audio.onnx --target-platform rk3588
```

- ### convert to rknn

1. After successfully exporting the ONNX model, you can use the `export/export_vision_rknn.py` script along with the `rknn-toolkit2` tool to convert the ONNX model to an RKNN model.

```bash
python export/export_vision_rknn.py --path /path/to/save/qwen2-vl-vision.onnx --target-platform rk3588
```

2. Convert the audio ONNX model to an RKNN model using the `export/export_audio_rknn.py` script.

```bash
python export/export_audio_rknn.py --path ./onnx/gemma-4-E2B-it-audio.onnx --target-platform rk3588
```

- ### convert to rkllm

1. We collected 20 image-text examples from the MMBench_DEV_EN dataset, stored in `data/datasets.json` and `data/datasets`. To use these data, you first need to create `input_embeds` for quantizing the RKLLM model. Run the following code to generate `data/inputs.json`.

```bash
python data/make_input_embeds_for_quantize.py --path /path/to/Qwen2-VL-model --model_type qwen2vl
```

2. Use the following code to export the RKLLM model.

```bash
python export/export_rkllm.py --path /path/to/Qwen2-VL-model --target-platform rk3588 --num_npu_core 3 --quantized_dtype w8a8 --device cpu --savepath /path/to/save/qwen2-vl-llm_rk3588.rkllm
```

3. When converting DeepSeekOCR, since the model supports only an older version of the transformers library, you need to replace the original file with `modeling_deepseekv2.py` from the `export` directory.

## 4. C++ Demo

In the `deploy` directory, we provide example code for board-side inference. This code demonstrates the process of "image/audio input to image/audio features," where an input image or audio is processed to output its corresponding features. These features are then used by the RKLLM model for multimodal content inference.

### 1. Compile and Build
Users can directly compile the example code by running the `deploy/build-linux.sh` or `deploy/build-android.sh` script (replacing the cross-compiler path with the actual path). This will generate an `install/demo_Linux_aarch64` folder in the `deploy` directory, containing the executables `imgenc`, `audioenc`, `demo`, and the `lib` folder.

**'img_start', 'img_end', 'img_content', 'audio_start', 'audio_end' and 'audio_content' in src/main.cpp should be set specially.** Pass `""` for a modality (image or audio) and its encoder model path to disable that modality.

```bash
cd deploy
# for linux
./build-linux.sh
# for android
./build-android.sh
# push install dir to device
adb push ./install/demo_Linux_aarch64 /data
# push model file to device
adb push qwen2-vl-vision_rk3588.rknn /data/models
adb push qwen2-vl-llm_rk3588.rkllm /data/models
```

### 2. Run Demo
Enter the `/data/demo_Linux_aarch64` directory on the board and run the example using the following code

```bash
adb shell
cd /data/demo_Linux_aarch64
# export lib path
export LD_LIBRARY_PATH=./lib
# soft link models dir
ln -s /data/models .
# run imgenc
./imgenc models/qwen2-vl-vision_rk3588.rknn demo.jpg 3
# run audioenc (Gemma-4 audio encoder)
./audioenc models/gemma-4-E2B-it-audio_rk3588.rknn demo.wav 3
# run demo(Multimodal Example, image only)
./demo demo.jpg models/qwen2-vl-vision_rk3588.rknn "" "" models/qwen2-vl-llm_rk3588.rkllm 2048 4096 3 rk3588 "<|vision_start|>" "<|vision_end|>" "<|image_pad|>"
# run demo(Multimodal Example, image + audio)
./demo demo.jpg models/gemma-4-E2B-it-vision_rk3588.rknn demo.wav models/gemma-4-E2B-it-audio_rk3588.rknn models/gemma-4-E2B-it_fp16_rk3588.rkllm 2048 4096 3 rk3588 "<|image>" "<image|>" "<|image|>" "<|audio>" "<audio|>" "<|audio|>"
```

Note: max_context_len must be larger than text-token-num+image-token-num+audio-token-num+max_new_tokens

The user can view the relevant runtime logs in the terminal and obtain the `img_vec.bin` (image features) and `audio_vec.bin` (audio features) files in the current directory, which contain the features corresponding to the input image/audio.

Multimodal Example

```text
user: <image>What is in the image?
robot: The image depicts an astronaut on the moon, enjoying a beer. The background shows the Earth and stars, creating a surreal and futuristic scene.
```

Audio Example (Gemma-4 audio encoder)

```text
user: <audio>What did the audio say?
robot: The audio says "..." 
```

Pure Text Example
```text
user: 把这句话翻译成英文: RK3588是新一代高端处理器，具有高算力、低功耗、超强多媒体、丰富数据接口等特点
robot: The RK3588 is a new generation of high-end processors with high computational power, low power consumption, strong multimedia capabilities, and rich data interfaces.
```