# finetune_gpt2_lora_peft.py
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
import time

def load_shakespeare():
    import requests
    print("Downloading Shakespeare...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    return requests.get(url).text

def prepare_data(text, tokenizer, max_length=128):
    print("Tokenizing...")
    tokens = tokenizer.encode(text)
    print(f"Total tokens: {len(tokens):,}")
    
    sequences = []
    for i in range(0, len(tokens) - max_length, max_length // 2):
        sequences.append(tokens[i:i + max_length])
    
    sequences = torch.tensor(sequences, dtype=torch.long)
    split = int(0.9 * len(sequences))
    
    print(f"Train: {split:,} | Val: {len(sequences)-split:,}")
    
    # Create dataset
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, sequences):
            self.sequences = sequences
        
        def __len__(self):
            return len(self.sequences)
        
        def __getitem__(self, idx):
            return {
                'input_ids': self.sequences[idx],
                'labels': self.sequences[idx]
            }
    
    return TextDataset(sequences[:split]), TextDataset(sequences[split:])

def generate(model, tokenizer, prompt, max_len=200, device='cuda'):
    model.eval()
    ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    out = model.generate(
        ids, 
        max_length=max_len, 
        temperature=0.8, 
        top_k=50, 
        top_p=0.95,
        do_sample=True, 
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Load model and tokenizer
    print("Loading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    original_params = sum(p.numel() for p in model.parameters())
    print(f"Original parameters: {original_params:,}\n")
    
    # Configure LoRA
    print("Adding LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,                          # LoRA rank
        lora_alpha=16,                # LoRA alpha
        lora_dropout=0.1,
        target_modules=['c_attn'],    # Apply to attention
        bias="none"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load data
    text = load_shakespeare()
    train_dataset, val_dataset = prepare_data(text, tokenizer, 128)
    
    # Test BEFORE training
    print("\n" + "=" * 80)
    print("BEFORE TRAINING:")
    print("=" * 80)
    model.to(device)
    print(generate(model, tokenizer, "ROMEO:", 150, device))
    print("=" * 80)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./gpt2-lora-shakespeare",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=3e-4,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=500,
        warmup_steps=100,
        fp16=True,  # Use mixed precision on RTX 3090
        report_to="none"
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Train
    print("\n" + "=" * 80)
    print("TRAINING WITH LoRA")
    print("=" * 80)
    start = time.time()
    trainer.train()
    print(f"\nTraining complete in {(time.time()-start)/60:.1f} minutes")
    
    # Test AFTER training
    print("\n" + "=" * 80)
    print("AFTER TRAINING:")
    print("=" * 80)
    
    for prompt in ["ROMEO:", "JULIET:", "First Citizen:"]:
        print(f"\n'{prompt}'")
        print("-" * 80)
        print(generate(model, tokenizer, prompt, 200, device))
        print("-" * 80)
    
    # Save
    model.save_pretrained("./gpt2-lora-shakespeare-final")
    print("\nSaved to: ./gpt2-lora-shakespeare-final")
    print("You now have GPT-2 fine-tuned with PROPER LoRA on Shakespeare!")

if __name__ == "__main__":
    main()
