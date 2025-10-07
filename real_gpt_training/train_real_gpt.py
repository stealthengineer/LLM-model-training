# train_real_gpt.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time
from pathlib import Path

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.W_o(context)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        x = x + self.dropout(self.attention(self.ln1(x), mask))
        x = x + self.dropout(self.feed_forward(self.ln2(x)))
        return x

class RealGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)
        
        x = self.dropout(self.token_embedding(input_ids) + self.position_embedding(positions))
        
        mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
        mask = mask.view(1, 1, seq_len, seq_len)
        
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate text"""
        self.eval()
        
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

def load_data():
    """Load prepared data"""
    print("Loading data...")
    train_data = torch.from_numpy(np.load('train_data.npy'))
    val_data = torch.from_numpy(np.load('val_data.npy'))
    
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    print(f"Train: {len(train_data):,} sequences")
    print(f"Val: {len(val_data):,} sequences")
    print(f"Vocab size: {tokenizer['vocab_size']}")
    
    return train_data, val_data, tokenizer

def train_model(
    model,
    train_data,
    val_data,
    n_epochs=10,
    batch_size=64,
    learning_rate=3e-4,
    device='cuda',
    eval_interval=500
):
    print("\n" + "=" * 80)
    print("TRAINING REAL GPT ON SHAKESPEARE")
    print("=" * 80)
    print(f"Device: {device.upper()}")
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {n_epochs}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 80)
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'step': [],
        'tokens_per_sec': []
    }
    
    n_batches = len(train_data) // batch_size
    step = 0
    start_time = time.time()
    total_tokens = 0
    
    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []
        
        # Shuffle
        indices = torch.randperm(len(train_data))
        train_data = train_data[indices]
        
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = batch_start + batch_size
            input_ids = train_data[batch_start:batch_end].long().to(device)
            
            logits = model(input_ids)
            
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, model.vocab_size),
                input_ids[:, 1:].reshape(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            total_tokens += batch_size * input_ids.shape[1]
            step += 1
            
            # Evaluation
            if step % eval_interval == 0:
                val_loss = evaluate(model, val_data, batch_size, device)
                elapsed = time.time() - start_time
                tokens_per_sec = total_tokens / elapsed
                
                history['train_loss'].append(np.mean(epoch_losses[-100:]))
                history['val_loss'].append(val_loss)
                history['step'].append(step)
                history['tokens_per_sec'].append(tokens_per_sec)
                
                print(f"Step {step:>6} | Epoch {epoch+1}/{n_epochs} | "
                      f"Train Loss: {history['train_loss'][-1]:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"{tokens_per_sec:.0f} tok/s")
                
                model.train()
        
        print(f"Epoch {epoch+1} complete | Avg Loss: {np.mean(epoch_losses):.4f}")
        print("-" * 80)
    
    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time/60:.1f} minutes")
    
    return history

@torch.no_grad()
def evaluate(model, val_data, batch_size, device):
    """Evaluate on validation set"""
    model.eval()
    losses = []
    
    n_batches = min(50, len(val_data) // batch_size)
    
    for i in range(n_batches):
        batch_start = i * batch_size
        batch_end = batch_start + batch_size
        input_ids = val_data[batch_start:batch_end].long().to(device)
        
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, model.vocab_size),
            input_ids[:, 1:].reshape(-1)
        )
        losses.append(loss.item())
    
    return np.mean(losses)

def generate_samples(model, tokenizer, device, num_samples=3):
    """Generate text samples"""
    print("\n" + "=" * 80)
    print("GENERATING SAMPLES")
    print("=" * 80)
    
    prompts = [
        "ROMEO:",
        "First Citizen:",
        "JULIET:"
    ]
    
    for prompt in prompts[:num_samples]:
        # Encode prompt
        context = torch.tensor(
            [[tokenizer['char_to_idx'][c] for c in prompt]], 
            dtype=torch.long, 
            device=device
        )
        
        # Generate
        generated = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=40)
        
        # Decode
        text = ''.join([tokenizer['idx_to_char'][i] for i in generated[0].tolist()])
        
        print(f"\nPrompt: '{prompt}'")
        print("-" * 80)
        print(text)
        print("-" * 80)

def visualize_training(model, history, device):
    """Create training visualization"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Loss curves
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(history['step'], history['train_loss'], 'b-', label='Train Loss', alpha=0.7)
    ax1.plot(history['step'], history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Progress on Shakespeare', fontweight='bold', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Speed
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(history['step'], history['tokens_per_sec'], 'g-', linewidth=2)
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Tokens/Second')
    ax2.set_title('Training Speed', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Model info
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    info_text = f"""
MODEL CONFIGURATION
{'=' * 40}

Parameters:          {model.count_parameters():,}
Model Dimension:     {model.d_model}
Attention Heads:     {model.n_heads}
Layers:              {model.n_layers}
Vocabulary:          {model.vocab_size} chars
Max Sequence:        {model.max_seq_len}

TRAINING RESULTS
{'=' * 40}

Final Train Loss:    {history['train_loss'][-1]:.4f}
Final Val Loss:      {history['val_loss'][-1]:.4f}
Avg Speed:           {np.mean(history['tokens_per_sec']):.0f} tok/s
Total Steps:         {history['step'][-1]:,}
    """
    
    ax3.text(0.1, 0.5, info_text, transform=ax3.transAxes,
            fontfamily='monospace', fontsize=9, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # GPU info
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        gpu_text = f"""
TRAINING ON: {gpu_name}
GPU Memory: {gpu_memory:.1f} GB | Model Size: {model.count_parameters() * 4 / (1024**2):.1f} MB (FP32)

This model learned the patterns and style of Shakespeare's writing!
It can now generate text that mimics Shakespeare's language.
        """
    else:
        gpu_text = "Trained on CPU"
    
    ax4.text(0.5, 0.5, gpu_text, transform=ax4.transAxes,
            ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.suptitle('Real GPT: Shakespeare Training Results', fontsize=16, fontweight='bold')
    plt.savefig('shakespeare_training_results.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    # Configuration - SCALED UP!
    CONFIG = {
        'd_model': 256,        # Embedding dimension
        'n_heads': 8,          # Attention heads  
        'n_layers': 6,         # Transformer layers
        'd_ff': 1024,          # Feed-forward dimension
        'max_seq_len': 128,    # Sequence length
        'dropout': 0.1,
        
        'batch_size': 64,
        'n_epochs': 10,
        'learning_rate': 3e-4,
        'eval_interval': 200
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load data
    train_data, val_data, tokenizer = load_data()
    
    # Create model
    model = RealGPT(
        vocab_size=tokenizer['vocab_size'],
        d_model=CONFIG['d_model'],
        n_heads=CONFIG['n_heads'],
        n_layers=CONFIG['n_layers'],
        d_ff=CONFIG['d_ff'],
        max_seq_len=CONFIG['max_seq_len'],
        dropout=CONFIG['dropout']
    )
    
    print(f"\nModel size: {model.count_parameters()/1e6:.1f}M parameters")
    
    # Train
    history = train_model(
        model, train_data, val_data,
        n_epochs=CONFIG['n_epochs'],
        batch_size=CONFIG['batch_size'],
        learning_rate=CONFIG['learning_rate'],
        device=device,
        eval_interval=CONFIG['eval_interval']
    )
    
    # Generate samples
    generate_samples(model, tokenizer, device)
    
    # Visualize
    visualize_training(model, history, device)
    
    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
        'history': history,
        'vocab_size': tokenizer['vocab_size']
    }, 'shakespeare_gpt.pt')
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print("Model saved to: shakespeare_gpt.pt")
    print("You now have a real GPT that writes like Shakespeare!")

if __name__ == "__main__":
    main()
