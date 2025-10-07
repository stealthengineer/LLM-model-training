# finetune_gpt2_lora.py
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import time

print("Using device check...")
import subprocess
import sys

try:
    import transformers
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
    import transformers

print(f"Transformers version: {transformers.__version__}")

class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer"""
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        self.scaling = alpha / rank
        
    def forward(self, x):
        return (x @ self.lora_A @ self.lora_B) * self.scaling

def add_lora_to_model(model, rank=8, alpha=16):
    """Add LoRA layers to GPT-2"""
    print(f"Adding LoRA adapters (rank={rank}, alpha={alpha})...")
    
    lora_layers = []
    
    for block in model.transformer.h:
        hidden_size = block.attn.c_attn.weight.shape[0]
        
        block.attn.q_lora = LoRALayer(hidden_size, hidden_size, rank, alpha)
        block.attn.v_lora = LoRALayer(hidden_size, hidden_size, rank, alpha)
        
        lora_layers.extend([block.attn.q_lora, block.attn.v_lora])
    
    for param in model.parameters():
        param.requires_grad = False
    
    for lora_layer in lora_layers:
        for param in lora_layer.parameters():
            param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"Trainable parameters: {trainable:,} ({100*trainable/total:.2f}%)")
    print(f"Total parameters: {total:,}")
    
    return model, lora_layers

def load_shakespeare():
    """Load Shakespeare text"""
    import requests
    
    print("Downloading Shakespeare dataset...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = requests.get(url).text
    
    print(f"Downloaded {len(text):,} characters")
    return text

def prepare_data(text, tokenizer, max_length=128):
    """Tokenize and create training data"""
    print(f"Tokenizing text (max_length={max_length})...")
    
    tokens = tokenizer.encode(text)
    print(f"Total tokens: {len(tokens):,}")
    
    sequences = []
    for i in range(0, len(tokens) - max_length, max_length // 2):
        sequences.append(tokens[i:i + max_length])
    
    sequences = torch.tensor(sequences, dtype=torch.long)
    
    split_idx = int(0.9 * len(sequences))
    train_data = sequences[:split_idx]
    val_data = sequences[split_idx:]
    
    print(f"Train sequences: {len(train_data):,}")
    print(f"Val sequences: {len(val_data):,}")
    
    return train_data, val_data

def train_with_lora(
    model,
    train_data,
    val_data,
    lora_layers,
    n_epochs=3,
    batch_size=8,
    learning_rate=3e-4,
    device='cuda'
):
    """Fine-tune with LoRA"""
    print("\n" + "=" * 80)
    print("FINE-TUNING GPT-2 WITH LoRA")
    print("=" * 80)
    
    # Model already on device from main()
    
    optimizer = torch.optim.AdamW(
        [p for lora in lora_layers for p in lora.parameters()],
        lr=learning_rate
    )
    
    history = {'train_loss': [], 'val_loss': [], 'step': []}
    
    n_batches = len(train_data) // batch_size
    step = 0
    
    start_time = time.time()
    
    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []
        
        indices = torch.randperm(len(train_data))
        train_data = train_data[indices]
        
        for batch_idx in range(n_batches):
            batch = train_data[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(device)
            
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for lora in lora_layers for p in lora.parameters()], 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            step += 1
            
            if step % 100 == 0:
                val_loss = evaluate(model, val_data, batch_size, device)
                history['train_loss'].append(np.mean(epoch_losses[-50:]))
                history['val_loss'].append(val_loss)
                history['step'].append(step)
                
                elapsed = time.time() - start_time
                print(f"Step {step:>4} | Epoch {epoch+1}/{n_epochs} | "
                      f"Train: {history['train_loss'][-1]:.4f} | "
                      f"Val: {val_loss:.4f} | {elapsed:.0f}s")
                
                model.train()
        
        print(f"Epoch {epoch+1} complete | Avg Loss: {np.mean(epoch_losses):.4f}")
        print("-" * 80)
    
    print(f"\nTraining complete in {(time.time() - start_time)/60:.1f} minutes")
    
    return history

@torch.no_grad()
def evaluate(model, val_data, batch_size, device):
    """Evaluate model"""
    model.eval()
    losses = []
    
    n_batches = min(20, len(val_data) // batch_size)
    
    for i in range(n_batches):
        batch = val_data[i * batch_size:(i + 1) * batch_size].to(device)
        outputs = model(batch, labels=batch)
        losses.append(outputs.loss.item())
    
    return np.mean(losses)

def generate_text(model, tokenizer, prompt, max_length=200, device='cuda'):
    """Generate text"""
    model.eval()
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("\nLoading GPT-2 Small (117M parameters)...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    model, lora_layers = add_lora_to_model(model, rank=8, alpha=16)
    
    # Move to device AFTER adding LoRA
    model = model.to(device)
    
    text = load_shakespeare()
    train_data, val_data = prepare_data(text, tokenizer, max_length=128)
    
    print("\n" + "=" * 80)
    print("BEFORE FINE-TUNING:")
    print("=" * 80)
    sample = generate_text(model, tokenizer, "ROMEO:", max_length=150, device=device)
    print(sample)
    print("=" * 80)
    
    history = train_with_lora(
        model, train_data, val_data, lora_layers,
        n_epochs=3,
        batch_size=8,
        learning_rate=3e-4,
        device=device
    )
    
    print("\n" + "=" * 80)
    print("AFTER FINE-TUNING:")
    print("=" * 80)
    
    prompts = ["ROMEO:", "JULIET:", "First Citizen:"]
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 80)
        text = generate_text(model, tokenizer, prompt, max_length=200, device=device)
        print(text)
        print("-" * 80)
    
    torch.save({
        'lora_layers': [lora.state_dict() for lora in lora_layers],
        'history': history
    }, 'gpt2_lora_shakespeare.pt')
    
    print("\n✓ Model saved to: gpt2_lora_shakespeare.pt")
    print("✓ You now have GPT-2 fine-tuned on Shakespeare with LoRA!")

if __name__ == "__main__":
    main()
