# merge_lora_adapters.py
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel
import os
import safetensors.torch

def load_lora_weights(adapter_path):
    """Load LoRA weights from safetensors format"""
    weight_file = os.path.join(adapter_path, "adapter_model.safetensors")
    if os.path.exists(weight_file):
        return safetensors.torch.load_file(weight_file)
    else:
        # Fallback to .bin format
        weight_file = os.path.join(adapter_path, "adapter_model.bin")
        return torch.load(weight_file)

def merge_adapters(paths, weights):
    """Merge multiple LoRA adapters with weights"""
    print("Loading adapters...")
    adapters = [load_lora_weights(path) for path in paths]
    
    # Weighted average
    merged = {}
    for key in adapters[0].keys():
        merged[key] = sum(w * adapter[key] for w, adapter in zip(weights, adapters))
    
    return merged

def main():
    models = {
        "wikipedia": "./gpt2-lora-wikipedia-final",
        "qa": "./gpt2-lora-qa-final"
    }
    
    # Adjust weights as desired
    weights = [0.6, 0.4]  # 60% Wikipedia knowledge, 40% Q&A ability
    
    print(f"Merging with weights: {weights}")
    print(f"Models: {list(models.keys())}")
    
    # Merge
    merged_weights = merge_adapters(list(models.values()), weights)
    
    # Save
    output_path = "./gpt2-lora-merged"
    os.makedirs(output_path, exist_ok=True)
    
    # Save as safetensors
    safetensors.torch.save_file(
        merged_weights, 
        os.path.join(output_path, "adapter_model.safetensors")
    )
    
    # Copy config
    import shutil
    shutil.copy(
        os.path.join(models["wikipedia"], "adapter_config.json"),
        os.path.join(output_path, "adapter_config.json")
    )
    
    # Copy tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(models["wikipedia"])
    tokenizer.save_pretrained(output_path)
    
    print(f"\nMerged model saved to: {output_path}")
    print("\nThis model combines:")
    print("  - Wikipedia factual knowledge (60%)")
    print("  - Q&A instruction following (40%)")

if __name__ == "__main__":
    main()
