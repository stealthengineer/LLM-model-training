# mini_gpt_pytorch.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time
from typing import Optional

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking"""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections and reshape for multi-head
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        # Apply causal mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        
        # Concatenate heads and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.W_o(context)
        
        return output

class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation"""
    
    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))

class TransformerBlock(nn.Module):
    """Single transformer block with attention + FFN"""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model)
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Attention with residual
        x = x + self.attention(self.ln1(x), mask)
        
        # Feed-forward with residual
        x = x + self.feed_forward(self.ln2(x))
        
        return x

class MiniGPTPyTorch(nn.Module):
    """Complete GPT model in PyTorch"""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 32
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights with small random values"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Create position indices
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)
        
        # Embeddings
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        # Create causal mask (lower triangular)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
        mask = mask.view(1, 1, seq_len, seq_len)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_synthetic_data(vocab_size: int, seq_len: int, n_samples: int) -> torch.Tensor:
    """Generate synthetic random token sequences"""
    # Random tokens (excluding special tokens like 0, 1)
    data = torch.randint(2, vocab_size, (n_samples, seq_len))
    return data

def train_model(
    model: MiniGPTPyTorch,
    train_data: torch.Tensor,
    n_epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: str = 'cuda'
):
    """Train the model with real-time progress"""
    
    print("=" * 80)
    print("MINI-GPT PYTORCH TRAINING")
    print("=" * 80)
    print(f"Device: {device.upper()}")
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Dataset size: {len(train_data):,} sequences")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {n_epochs}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 80)
    
    # Move model to device
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'loss': [],
        'epoch': [],
        'tokens_per_sec': []
    }
    
    n_batches = len(train_data) // batch_size
    
    start_time = time.time()
    total_tokens = 0
    
    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []
        
        # Shuffle data
        indices = torch.randperm(len(train_data))
        train_data = train_data[indices]
        
        for batch_idx in range(n_batches):
            # Get batch
            batch_start = batch_idx * batch_size
            batch_end = batch_start + batch_size
            input_ids = train_data[batch_start:batch_end].to(device)
            
            # Forward pass
            logits = model(input_ids)
            
            # Compute loss (predict next token)
            # Shift targets: predict token i+1 from token i
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, model.vocab_size),
                input_ids[:, 1:].reshape(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update weights
            optimizer.step()
            
            # Track metrics
            epoch_losses.append(loss.item())
            total_tokens += batch_size * input_ids.shape[1]
            
            # Progress
            if batch_idx % 20 == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = total_tokens / elapsed
                avg_loss = np.mean(epoch_losses[-20:]) if len(epoch_losses) > 20 else loss.item()
                
                print(f"Epoch {epoch+1}/{n_epochs} | Batch {batch_idx}/{n_batches} | "
                      f"Loss: {avg_loss:.4f} | {tokens_per_sec:.0f} tokens/s")
        
        # Epoch summary
        avg_epoch_loss = np.mean(epoch_losses)
        elapsed = time.time() - start_time
        tokens_per_sec = total_tokens / elapsed
        
        history['loss'].append(avg_epoch_loss)
        history['epoch'].append(epoch)
        history['tokens_per_sec'].append(tokens_per_sec)
        
        print(f"\nEpoch {epoch+1} complete | Avg Loss: {avg_epoch_loss:.4f} | "
              f"Speed: {tokens_per_sec:.0f} tokens/s")
        print("-" * 80)
        
        # Learning rate decay
        if (epoch + 1) % 5 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9
            print(f"Learning rate decayed to {optimizer.param_groups[0]['lr']:.6f}")
    
    total_time = time.time() - start_time
    final_speed = total_tokens / total_time
    
    print(f"\n✓ Training complete in {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"✓ Final speed: {final_speed:.0f} tokens/second")
    print(f"✓ Total tokens processed: {total_tokens:,}")
    
    return history

def visualize_training(model, history, device):
    """Create comprehensive training visualization"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Loss curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(history['epoch'], history['loss'], 'b-', linewidth=2, marker='o', markersize=6)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss (PyTorch + GPU)', fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Add final loss annotation
    final_loss = history['loss'][-1]
    ax1.annotate(f'Final: {final_loss:.4f}', 
                xy=(history['epoch'][-1], final_loss),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                fontsize=10, fontweight='bold')
    
    # 2. Training speed
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(history['epoch'], history['tokens_per_sec'], 'g-', linewidth=2, marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Tokens/Second')
    ax2.set_title('Training Speed', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    avg_speed = np.mean(history['tokens_per_sec'])
    ax2.axhline(y=avg_speed, color='r', linestyle='--', alpha=0.5, 
               label=f'Avg: {avg_speed:.0f} tok/s')
    ax2.legend()
    
    # 3. Model architecture
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    arch_text = f"""
MODEL ARCHITECTURE
{'=' * 40}

Vocabulary Size:     {model.vocab_size}
Model Dimension:     {model.d_model}
Attention Heads:     {model.n_heads}
Transformer Layers:  {model.n_layers}
Max Sequence Length: {model.max_seq_len}

PARAMETERS
{'=' * 40}

Token Embeddings:    {model.token_embedding.weight.numel():,}
Position Embeddings: {model.position_embedding.weight.numel():,}
Transformer Blocks:  {sum(p.numel() for block in model.blocks for p in block.parameters()):,}
Output Head:         {model.lm_head.weight.numel():,}

Total Parameters:    {model.count_parameters():,}
Model Size (FP32):   {model.count_parameters() * 4 / (1024**2):.2f} MB
    """
    
    ax3.text(0.1, 0.5, arch_text, transform=ax3.transAxes,
            fontfamily='monospace', fontsize=9, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # 4. GPU info
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
        
        gpu_text = f"""
TRAINING SETUP
{'=' * 70}

Device:              {gpu_name}
Total GPU Memory:    {gpu_memory:.1f} GB
Memory Allocated:    {gpu_memory_allocated:.2f} GB
CUDA Version:        {torch.version.cuda}
PyTorch Version:     {torch.__version__}

Final Training Loss: {history['loss'][-1]:.4f}
Average Speed:       {np.mean(history['tokens_per_sec']):.0f} tokens/second
Total Epochs:        {len(history['epoch'])}

✓ Model trained successfully with GPU acceleration!
✓ Ready for inference and fine-tuning
        """
    else:
        gpu_text = f"""
TRAINING SETUP
{'=' * 70}

Device:              CPU
PyTorch Version:     {torch.__version__}

Final Training Loss: {history['loss'][-1]:.4f}
Average Speed:       {np.mean(history['tokens_per_sec']):.0f} tokens/second
Total Epochs:        {len(history['epoch'])}

⚠ Trained on CPU (GPU not available)
✓ Model trained successfully!
        """
    
    ax4.text(0.1, 0.5, gpu_text, transform=ax4.transAxes,
            fontfamily='monospace', fontsize=9, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.suptitle('Mini-GPT PyTorch Training Results', fontsize=16, fontweight='bold')
    plt.savefig('pytorch_training_results.png', dpi=150, bbox_inches='tight')
    plt.show()

def save_model(model, history, filepath='mini_gpt_pytorch.pt'):
    """Save model checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'vocab_size': model.vocab_size,
            'd_model': model.d_model,
            'n_heads': model.n_heads,
            'n_layers': model.n_layers,
            'max_seq_len': model.max_seq_len
        },
        'training_history': history,
        'total_parameters': model.count_parameters()
    }
    
    torch.save(checkpoint, filepath)
    print(f"\n✓ Model saved to: {filepath}")
    print(f"  File size: {torch.load(filepath, map_location='cpu', weights_only=True)}")

def main():
    """Main training pipeline"""
    
    # Configuration
    VOCAB_SIZE = 20
    D_MODEL = 64
    N_HEADS = 4
    N_LAYERS = 2
    MAX_SEQ_LEN = 32
    
    N_SAMPLES = 10000  # Synthetic data size
    BATCH_SIZE = 64
    N_EPOCHS = 20
    LEARNING_RATE = 0.001
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        print("\n🚀 GPU DETECTED!")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    else:
        print("\n⚠️  No GPU detected, using CPU")
    
    print("\n" + "=" * 80)
    
    # Create model
    model = MiniGPTPyTorch(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        max_seq_len=MAX_SEQ_LEN
    )
    
    # Generate synthetic data
    print(f"\nGenerating {N_SAMPLES:,} synthetic sequences...")
    train_data = create_synthetic_data(VOCAB_SIZE, MAX_SEQ_LEN, N_SAMPLES)
    print(f"✓ Dataset shape: {train_data.shape}")
    
    # Train
    history = train_model(
        model=model,
        train_data=train_data,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        device=device
    )
    
    # Visualize
    print("\nGenerating visualizations...")
    visualize_training(model, history, device)
    print("✓ Saved: pytorch_training_results.png")
    
    # Save model
    save_model(model, history, 'mini_gpt_pytorch.pt')
    
    print("\n" + "=" * 80)
    print("🎉 PYTORCH TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nYour RTX 3090 processed ~{np.mean(history['tokens_per_sec']):.0f} tokens/second")
    print("This is the power of GPU acceleration + PyTorch autograd!")

if __name__ == "__main__":
    main()
