# finetune_gpt2_lora_fixed.py
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import time

print("Using PyTorch and Transformers...")

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

def inject_lora_into_gpt2(model, rank=8, alpha=16):
    """Properly inject LoRA into GPT-2 by modifying forward hooks"""
    print(f"Injecting LoRA (rank={rank}, alpha={alpha})...")
    
    lora_params = []
    
    for i, block in enumerate(model.transformer.h):
        # Get the attention module
        attn = block.attn
        
        # Create LoRA layers for Q and V projections
        hidden_size = attn.c_attn.weight.shape[1]
        
        lora_q = LoRALayer(hidden_size, hidden_size, rank, alpha)
        lora_v = LoRALayer(hidden_size, hidden_size, rank, alpha)
        
        # Store as attributes
        attn.lora_q = lora_q
        attn.lora_v = lora_v
        
        lora_params.extend([lora_q, lora_v])
        
        # Monkey-patch the forward method to include LoRA
        original_forward = attn._attn
        
        def new_attn(self, query, key, value, attention_mask=None, head_mask=None):
            # Add LoRA to query and value
            query = query + self.lora_q(query)
            value = value + self.lora_v(value)
            
            # Call original attention
            return original_forward(query, key, value, attention_mask, head_mask)
        
        # Bind the new method
        import types
        attn._attn = types.MethodType(new_attn, attn)
    
    # Freeze all original parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze LoRA parameters
    for lora in lora_params:
        for param in lora.parameters():
            param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
    print(f"Total: {total:,}")
    
    return model, lora_params

def load_shakespeare():
    import requests
    print("Downloading Shakespeare...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    return requests.get(url).text

def prepare_data(text, tokenizer, max_length=128):
    print(f"Tokenizing (max_length={max_length})...")
    tokens = tokenizer.encode(text)
    print(f"Total tokens: {len(tokens):,}")
    
    sequences = []
    for i in range(0, len(tokens) - max_length, max_length // 2):
        sequences.append(tokens[i:i + max_length])
    
    sequences = torch.tensor(sequences, dtype=torch.long)
    
    split = int(0.9 * len(sequences))
    return sequences[:split], sequences[split:]

def train_with_lora(model, train_data, val_data, lora_params, n_epochs=3, batch_size=8, lr=3e-4, device='cuda'):
    print("\n" + "=" * 80)
    print("FINE-TUNING GPT-2 WITH LoRA")
    print("=" * 80)
    
    optimizer = torch.optim.AdamW([p for lora in lora_params for p in lora.parameters()], lr=lr)
    
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
            torch.nn.utils.clip_grad_norm_([p for lora in lora_params for p in lora.parameters()], 1.0)
            optimizer.step()
            
            losses.append(loss.item())
            step += 1
            
            if step % 100 == 0:
                val_loss = evaluate(model, val_data, batch_size, device)
                history['train_loss'].append(np.mean(losses[-50:]))
                history['val_loss'].append(val_loss)
                history['step'].append(step)
                
                print(f"Step {step:>4} | Epoch {epoch+1}/{n_epochs} | "
                      f"Train: {history['train_loss'][-1]:.4f} | Val: {val_loss:.4f} | "
                      f"{time.time()-start:.0f}s")
                model.train()
        
        print(f"Epoch {epoch+1} done | Loss: {np.mean(losses):.4f}")
        print("-" * 80)
    
    print(f"Training done in {(time.time()-start)/60:.1f} min")
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
    
    model, lora_params = inject_lora_into_gpt2(model, rank=8, alpha=16)
    model = model.to(device)
    
    text = load_shakespeare()
    train, val = prepare_data(text, tokenizer, 128)
    print(f"Train: {len(train):,} | Val: {len(val):,}\n")
    
    print("=" * 80)
    print("BEFORE TRAINING:")
    print("=" * 80)
    print(generate(model, tokenizer, "ROMEO:", 150, device))
    print("=" * 80)
    
    history = train_with_lora(model, train, val, lora_params, n_epochs=3, batch_size=8, lr=3e-4, device=device)
    
    print("\n" + "=" * 80)
    print("AFTER TRAINING:")
    print("=" * 80)
    for p in ["ROMEO:", "JULIET:", "First Citizen:"]:
        print(f"\n'{p}'")
        print("-" * 80)
        print(generate(model, tokenizer, p, 200, device))
        print("-" * 80)
    
    torch.save({'lora': [l.state_dict() for l in lora_params], 'history': history}, 'gpt2_lora.pt')
    print("\n✓ Saved: gpt2_lora.pt")

if __name__ == "__main__":
    main()
