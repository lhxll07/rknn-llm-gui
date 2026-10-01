import torch
import numpy as np
import os
import sys
import argparse
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig


class gemma4_vision(torch.nn.Module):
    def __init__(self, vlm, in_h, in_w, patch_size):
        super(gemma4_vision, self).__init__()
        self.vision_tower = vlm.model.vision_tower
        self.embed_vision = vlm.model.embed_vision
        self.patch_size = patch_size
        
        # 根据要求，宽高必须是48的倍数对齐
        align_size = 48
        align_h = (in_h + align_size - 1) // align_size * align_size
        align_w = (in_w + align_size - 1) // align_size * align_size
        
        self.in_h = align_h
        self.in_w = align_w

    def forward(self, pixel, pixel_position_ids):
        pixel = pixel.float() 
        
        B, C, H, W = pixel.shape
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        
        # 将 4D Tensor 转为 Patches
        # 1. Reshape -> [B, C, num_patches_h, patch_size, num_patches_w, patch_size]
        patched_image = pixel.reshape(B, C, num_patches_h, self.patch_size, num_patches_w, self.patch_size)
        # 2. Permute -> [B, num_patches_h, num_patches_w, patch_size, patch_size, C]
        patched_image = patched_image.permute(0, 2, 4, 3, 5, 1).contiguous()
        # 3. Reshape 展平 -> [B, num_patches_h * num_patches_w, patch_size * patch_size * C]
        patched_image = patched_image.reshape(B, num_patches_h * num_patches_w, -1)

        last_hidden_state = self.vision_tower(patched_image, pixel_position_ids).last_hidden_state
        # 调用原始vision model，传入patch和位置ids
        return self.embed_vision(inputs_embeds=last_hidden_state)


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--model_path', type=str, default='google/gemma-4-E2B-it', help='model path', required=False)
    argparse.add_argument('--export_vision_path', type=str, default='onnx/gemma-4-E2B-it-vision.onnx', help="export vision onnx model path", required=False)
    argparse.add_argument('--batch_size', type=int, default=1, help='batch size', required=False)
    argparse.add_argument('--height', type=int, default=448, help='image height', required=False)
    argparse.add_argument('--width', type=int, default=448, help='image width', required=False)
    argparse.add_argument('--device', type=str, default="cpu", help='cpu or cuda', required=False)
    args = argparse.parse_args()

    PATCH_SIZE = 16

    path = args.model_path
    savepath = args.export_vision_path
    device_type = args.device
    os.makedirs(os.path.dirname(savepath), exist_ok=True)

    in_h = args.height
    in_w = args.width
    custom_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './'))
    print(f"Custom gemma4 module path: {custom_path}")

    # Load our custom modeling_gemma4 module using importlib
    import importlib.util
    modeling_spec = importlib.util.spec_from_file_location(
        'transformers.models.gemma4.modeling_gemma4',
        os.path.join(custom_path, 'modeling_gemma4.py')
    )
    modeling_module = importlib.util.module_from_spec(modeling_spec)
    sys.modules['transformers.models.gemma4.modeling_gemma4'] = modeling_module
    modeling_spec.loader.exec_module(modeling_module)
    print(f"Loaded custom modeling_gemma4 from: {modeling_module.__file__}")
    
    import transformers.models.gemma4 as _gemma4_pkg
    for _name in getattr(modeling_module, '__all__', []):
        if hasattr(modeling_module, _name):
            setattr(_gemma4_pkg, _name, getattr(modeling_module, _name))
    import transformers.models.auto as _auto_pkg
    for _attr in dir(_auto_pkg):
        _mapping = getattr(_auto_pkg, _attr, None)
        if hasattr(_mapping, '_modules') and isinstance(getattr(_mapping, '_modules', None), dict):
            _mapping._modules.pop('gemma4', None)

    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(path, 
                                                    config = config,
                                                    dtype=torch.float32, # 注意此处的数据类型，由于 rknn 目前仅支持 float32 ，因此需要指定；若是在加载权重时限制了数据类型，需要自行修改config.json中的 "use_flash_attn" 参数为 false
                                                    low_cpu_mem_usage=True,
                                                    attn_implementation="eager",
                                                    trust_remote_code=True)
    model = gemma4_vision(model, in_h, in_w, PATCH_SIZE)

    in_h = model.in_h
    in_w = model.in_w

    num_patches_h = in_h // PATCH_SIZE
    num_patches_w = in_w // PATCH_SIZE
    num_patches = num_patches_h * num_patches_w

    # 构建 Dummy Inputs: 图像和位置编码
    fake_input = torch.randint(0, 255, (1, 3, in_h, in_w), dtype=torch.float32)
    
    # 构造 pixel_position_ids: shape [1, max_patches, 2]
    y, x = torch.meshgrid(torch.arange(num_patches_h), torch.arange(num_patches_w), indexing='ij')
    fake_pos_ids = torch.stack([x, y], dim=-1).reshape(1, -1, 2).to(torch.int64)

    # 推理一次
    out = model(fake_input, fake_pos_ids)

    # 导出ONNX
    torch.onnx.export(
        model,
        (fake_input, fake_pos_ids),
        savepath,
        input_names=['pixel', 'pixel_position_ids'],
        output_names=['last_hidden_state'],
        dynamic_axes={
            'pixel': {2: 'height', 3: 'width'},
            'pixel_position_ids': {1: 'max_patches'}},
        opset_version=19)
        
    print(f"Exported to {savepath}")
