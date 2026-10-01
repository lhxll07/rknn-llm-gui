import torch
import os
import sys
from transformers import AutoModelForCausalLM, AutoConfig


class Gemma4AudioWrapper(torch.nn.Module):
    def __init__(self, audio_tower, embed_audio):
        super().__init__()
        self.audio_tower = audio_tower
        self.embed_audio = embed_audio

    def forward(self, input_features, input_features_mask):
        outputs = self.audio_tower(
            input_features=input_features,
            attention_mask=input_features_mask,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state   # [B, T, D]
        output_mask = outputs.attention_mask        # [B, T]

        hidden_states = self.embed_audio(hidden_states)

        return hidden_states


if __name__ == '__main__':
    from argparse import ArgumentParser
    argparse = ArgumentParser()
    argparse.add_argument("--model_path", type=str, help='model path', default='google/gemma-4-E2B-it', required=False)
    argparse.add_argument("--export_audio_path", type=str, default='onnx/gemma-4-E2B-it-audio.onnx', help="export audio onnx model path", required=False)
    args = argparse.parse_args()

    path = args.model_path
    savepath = args.export_audio_path
    os.makedirs(os.path.dirname(savepath), exist_ok=True)

    BATCH_SIZE = 1
    MAX_MEL_FRAMES = 756
    MEL_BINS = 128
    
    # Get the custom module path
    custom_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'./'))
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

    kwargs = {
        'trust_remote_code': True,
        'torch_dtype': torch.float32,
    }
    config = AutoConfig.from_pretrained(path, **kwargs)

    # RKNN runs the audio tower internally in FP16. The stock attention_invalid_logits_value (-1e9) overflows FP16 to -inf
    config.audio_config.attention_invalid_logits_value = -3e4
    kwargs['config'] = config
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs).eval()

    # export audio model
    audio_tower = model.model.audio_tower
    embed_audio = model.model.embed_audio

    audio_tower.eval()

    wrapper = Gemma4AudioWrapper(audio_tower, embed_audio).eval()

    input_features = torch.zeros(
        BATCH_SIZE, MAX_MEL_FRAMES, MEL_BINS, dtype=torch.float32
    )
    input_features_mask = torch.ones(
        BATCH_SIZE, MAX_MEL_FRAMES, dtype=torch.float32
    )

    # ====== 导出 ONNX ======
    torch.onnx.export(
        wrapper,
        (input_features, input_features_mask),
        savepath,
        input_names=["input_features", "input_features_mask"],
        output_names=["hidden_states"],
        opset_version=19)
            
    print(f"Exported to {savepath}")
    
