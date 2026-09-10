import torch
from transformers import BitsAndBytesConfig

def get_device_map():
    if torch.cuda.is_available():
        return {"": torch.cuda.current_device()}
    elif torch.backends.mps.is_available():
        return {"": "mps"}
    return "auto"

def get_quantization_config(use_4bit: bool):
    if not use_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

def load_model_for_training(model_id, use_4bit, device_map):
    from transformers import AutoModelForCausalLM
    from peft import prepare_model_for_kbit_training

    quant_config = get_quantization_config(use_4bit)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    return model

def load_model_for_inference(model_id, use_4bit, device_map):
    from transformers import AutoModelForCausalLM

    quant_config = get_quantization_config(use_4bit)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    return model