# test_wikipedia_model.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel
import torch

def generate_text(model, tokenizer, prompt, max_length=200):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    print("Loading Wikipedia-trained model...\n")
    
    # Load base model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-lora-wikipedia-final")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(model, "./gpt2-lora-wikipedia-final")
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test prompts
    prompts = [
        "The theory of relativity is",
        "Python is a programming language that",
        "The Roman Empire was",
        "Machine learning is the study of",
        "The human brain consists of"
    ]
    
    print("=" * 80)
    print("WIKIPEDIA MODEL GENERATION EXAMPLES")
    print("=" * 80)
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 80)
        text = generate_text(model, tokenizer, prompt, max_length=150)
        print(text)
        print("-" * 80)

if __name__ == "__main__":
    main()
