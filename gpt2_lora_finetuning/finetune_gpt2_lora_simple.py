# finetune_gpt2_lora_simple.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import time

class LoRALinear(nn.Module):
    """Linear layer with LoRA"""
    def __init__(self, original_linear, rank=8, alpha=16):
        super().__init__()
        self.original = original_linear
        self.original.requires_grad_(False)
        
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = alpha / rank
        
    def forward(self, x):
        result = self.original(x)
        lora_result = (x @ self.lora_A @ self.lora_B) * self.scaling
        return result + lora_result

def add_lora_to_gpt2(model, rank=8, alpha=16, target_modules=['c_attn']):
    """Add LoRA to specific modules in GPT-2"""
    print(f"Adding LoRA (rank={rank})...")
    
    lora_modules = []
    
    for name, module in model.named_modules():
        # Target the attention projection layers
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Get parent module and attribute name
                *parent_path, attr_name = name.split('.')
                parent = model
                for p in parent_path:
                    parent = getattr(parent, p)
                
                # Replace with LoRA version
                lora_linear = LoRALinear(module, rank, alpha)
                setattr(parent, attr_name, lora_linear)
                lora_modules.append(lora_linear)
                print(f"  Added LoRA to: {name}")
    
    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"\nTrainable: {trainable:,} ({100*trainable/total:.2f}%)")
    print(f"Total: {total:,}")
    
    return model, lora_modules

def load_shakespeare():
    import requests
    print("Downloading Shakespeare...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    return requests.get(url).text

def prepare_data(text, tokenizer, max_length=128):
    print(f"Tokenizing...")
    tokens = tokenizer.encode(text)
    print(f"Tokens: {len(tokens):,}")
    
    sequences = []
    for i in range(0, len(tokens) - max_length, max_length // 2):
        sequences.append(tokens[i:i + max_length])
    
    sequences = torch.tensor(sequences, dtype=torch.long)
    split = int(0.9 * len(sequences))
    
    print(f"Train: {split:,} | Val: {len(sequences)-split:,}")
    return sequences[:split], sequences[split:]

def train(model, train_data, val_data, n_epochs=3, batch_size=8, lr=3e-4, device='cuda'):
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    
    history = {'train_loss': [], 'val_loss': [], 'step': []}
    n_batches = len(train_data) // batch_size
    step = 0
    start = time.time()
    
    for epoch in range(n_epochs):
        model.train()
        losses = []
        
        perm = torch.randperm(len(train_data))
        train_data = train_data[perm]
        
        for i in range(n_batches):
            batch = train_data[i*batch_size:(i+1)*batch_size].to(device)
            
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            
            losses.append(loss.item())
            step += 1
            
            if step % 100 == 0:
                val_loss = evaluate(model, val_data, batch_size, device)
                history['train_loss'].append(np.mean(losses[-50:]))
                history['val_loss'].append(val_loss)
                history['step'].append(step)
                
                print(f"Step {step:>4} | E{epoch+1}/{n_epochs} | "
                      f"Train: {history['train_loss'][-1]:.4f} | Val: {val_loss:.4f} | "
                      f"{time.time()-start:.0f}s")
                model.train()
        
        print(f"Epoch {epoch+1} complete | Loss: {np.mean(losses):.4f}")
        print("-" * 80)
    
    print(f"Done in {(time.time()-start)/60:.1f} min")
    return history

@torch.no_grad()
def evaluate(model, val_data, batch_size, device):
    model.eval()
    losses = []
    for i in range(min(20, len(val_data)//batch_size)):
        batch = val_data[i*batch_size:(i+1)*batch_size].to(device)
        losses.append(model(batch, labels=batch).loss.item())
    return np.mean(losses)

def generate(model, tokenizer, prompt, max_len=200, device='cuda'):
    model.eval()
    ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    out = model.generate(ids, max_length=max_len, temperature=0.8, top_k=50, top_p=0.95,
                        do_sample=True, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    print("Loading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Loaded: {sum(p.numel() for p in model.parameters()):,} params\n")
    
    model, lora_modules = add_lora_to_gpt2(model, rank=8, alpha=16)
    model = model.to(device)
    
    text = load_shakespeare()
    train_data, val_data = prepare_data(text, tokenizer, 128)
    
    print("\n" + "=" * 80)
    print("BEFORE TRAINING:")
    print("=" * 80)
    print(generate(model, tokenizer, "ROMEO:", 150, device))
    print("=" * 80)
    
    history = train(model, train_data, val_data, n_epochs=3, batch_size=8, lr=3e-4, device=device)
    
    print("\n" + "=" * 80)
    print("AFTER TRAINING:")
    print("=" * 80)
    for prompt in ["ROMEO:", "JULIET:", "First Citizen:"]:
        print(f"\n'{prompt}'")
        print("-" * 80)
        print(generate(model, tokenizer, prompt, 200, device))
        print("-" * 80)
    
    torch.save({'lora_modules': [m.state_dict() for m in lora_modules], 'history': history}, 'gpt2_lora.pt')
    print("\nSaved: gpt2_lora.pt")
    print("You now have GPT-2 fine-tuned with LoRA on Shakespeare!")

if __name__ == "__main__":
    main()
