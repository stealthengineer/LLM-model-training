# positional_encodings.py
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

class PositionalEncodings:
    """Implementations of different positional encoding methods"""
    
    def __init__(self, max_seq_len=512, d_model=128):
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
    def sinusoidal_encoding(self):
        """Original Transformer positional encoding (Vaswani et al.)"""
        position = np.arange(self.max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * 
                         -(np.log(10000.0) / self.d_model))
        
        pe = np.zeros((self.max_seq_len, self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def learned_encoding(self):
        """Learned positional embeddings (like GPT)"""
        # Initialize random embeddings (would be learned during training)
        np.random.seed(42)
        return np.random.randn(self.max_seq_len, self.d_model) * 0.1
    
    def rotary_encoding(self, seq_len=None):
        """Simplified RoPE (Rotary Position Embedding) - used in LLaMA"""
        if seq_len is None:
            seq_len = self.max_seq_len
            
        # Simplified version for demonstration
        position = np.arange(seq_len)[:, np.newaxis]
        dims = np.arange(0, self.d_model, 2)[np.newaxis, :] / self.d_model
        
        angles = position / np.power(10000, dims)
        
        # Create rotation matrix components
        cos_vals = np.cos(angles)
        sin_vals = np.sin(angles)
        
        # For simplicity, returning the angle components
        rope = np.zeros((seq_len, self.d_model))
        rope[:, 0::2] = sin_vals
        rope[:, 1::2] = cos_vals
        
        return rope
    
    def alibi_encoding(self, seq_len=None):
        """ALiBi (Attention with Linear Biases) - position-based attention penalty"""
        if seq_len is None:
            seq_len = self.max_seq_len
            
        # Create distance matrix
        positions = np.arange(seq_len)
        distances = positions[:, np.newaxis] - positions[np.newaxis, :]
        
        # Apply linear bias (simplified)
        # In practice, different heads get different slopes
        slopes = np.array([2**(-8/8 * (i+1)) for i in range(8)])  # 8 heads
        
        # Return distance matrix (would be added to attention scores)
        return distances
    
    def visualize_encodings(self):
        """Visualize different encoding methods"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Sinusoidal encoding
        sin_enc = self.sinusoidal_encoding()[:100, :64]  # First 100 positions, 64 dims
        im1 = axes[0, 0].imshow(sin_enc.T, cmap='coolwarm', aspect='auto')
        axes[0, 0].set_title('Sinusoidal Positional Encoding')
        axes[0, 0].set_xlabel('Position')
        axes[0, 0].set_ylabel('Dimension')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Learned encoding
        learned_enc = self.learned_encoding()[:100, :64]
        im2 = axes[0, 1].imshow(learned_enc.T, cmap='coolwarm', aspect='auto')
        axes[0, 1].set_title('Learned Positional Encoding')
        axes[0, 1].set_xlabel('Position')
        axes[0, 1].set_ylabel('Dimension')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # RoPE encoding
        rope_enc = self.rotary_encoding()[:100, :64]
        im3 = axes[1, 0].imshow(rope_enc.T, cmap='coolwarm', aspect='auto')
        axes[1, 0].set_title('Rotary Position Encoding (RoPE)')
        axes[1, 0].set_xlabel('Position')
        axes[1, 0].set_ylabel('Dimension')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # ALiBi distances
        alibi_dist = self.alibi_encoding(50)
        im4 = axes[1, 1].imshow(alibi_dist, cmap='coolwarm', aspect='auto')
        axes[1, 1].set_title('ALiBi Distance Matrix')
        axes[1, 1].set_xlabel('Position')
        axes[1, 1].set_ylabel('Position')
        plt.colorbar(im4, ax=axes[1, 1])
        
        plt.suptitle('Comparison of Positional Encoding Methods', fontsize=16)
        plt.tight_layout()
        plt.savefig('positional_encodings_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return sin_enc, learned_enc, rope_enc, alibi_dist
    
    def plot_attention_without_position(self, seq_len=10):
        """Show why positional encoding is necessary"""
        # Random attention scores without position
        np.random.seed(42)
        attention_no_pos = np.random.rand(seq_len, seq_len)
        attention_no_pos = attention_no_pos / attention_no_pos.sum(axis=-1, keepdims=True)
        
        # With sinusoidal position
        pos_enc = self.sinusoidal_encoding()[:seq_len, :seq_len]
        pos_similarity = np.dot(pos_enc, pos_enc.T)
        attention_with_pos = attention_no_pos + 0.1 * pos_similarity
        attention_with_pos = np.exp(attention_with_pos) / np.exp(attention_with_pos).sum(axis=-1, keepdims=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        im1 = ax1.imshow(attention_no_pos, cmap='Blues')
        ax1.set_title('Attention WITHOUT Position\n(Order doesn\'t matter!)')
        ax1.set_xlabel('Key Position')
        ax1.set_ylabel('Query Position')
        plt.colorbar(im1, ax=ax1)
        
        im2 = ax2.imshow(attention_with_pos, cmap='Blues')
        ax2.set_title('Attention WITH Position\n(Order matters!)')
        ax2.set_xlabel('Key Position')
        ax2.set_ylabel('Query Position')
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        plt.savefig('attention_position_importance.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def animate_3d_encoding(self):
        """Create 3D visualization of positional encoding"""
        positions = 20
        dims = 3
        
        # Get encoding for animation
        encoding = self.sinusoidal_encoding()[:positions, :dims]
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the 3D trajectory
        ax.plot(encoding[:, 0], encoding[:, 1], encoding[:, 2], 'b-', linewidth=2)
        
        # Mark positions
        for i in range(positions):
            ax.scatter(encoding[i, 0], encoding[i, 1], encoding[i, 2], 
                      s=50, c=f'C{i%10}', marker='o')
            ax.text(encoding[i, 0], encoding[i, 1], encoding[i, 2], f'  {i}', fontsize=8)
        
        ax.set_xlabel('Dimension 0')
        ax.set_ylabel('Dimension 1')
        ax.set_zlabel('Dimension 2')
        ax.set_title('Positional Encoding in 3D Space\n(Each position has unique coordinates)')
        
        plt.savefig('positional_3d.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return encoding

# Experiment runner
def run_experiments():
    print("Exploring Positional Encodings...")
    print("=" * 50)
    
    pe = PositionalEncodings(max_seq_len=512, d_model=128)
    
    # 1. Compare different encodings
    print("\n1. Visualizing different encoding methods...")
    sin_enc, learned_enc, rope_enc, alibi_dist = pe.visualize_encodings()
    print("   Saved: positional_encodings_comparison.png")
    
    # 2. Show importance of position
    print("\n2. Demonstrating why position matters...")
    pe.plot_attention_without_position()
    print("   Saved: attention_position_importance.png")
    
    # 3. 3D visualization
    print("\n3. Creating 3D position visualization...")
    encoding_3d = pe.animate_3d_encoding()
    print("   Saved: positional_3d.png")
    
    # Analysis
    print("\n" + "=" * 50)
    print("KEY INSIGHTS:")
    print("- Without position: 'dog bites man' = 'man bites dog'")
    print("- Sinusoidal: Smooth, allows extrapolation to longer sequences")
    print("- Learned: More flexible but limited to training length")
    print("- RoPE: Rotation-based, better for relative positions")
    print("- ALiBi: Direct bias on attention, no position embeddings needed")
    
    return pe

if __name__ == "__main__":
    pe = run_experiments()
