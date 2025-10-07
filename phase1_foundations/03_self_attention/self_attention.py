# self_attention.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class SelfAttention:
    """Build self-attention mechanism from scratch"""
    
    def __init__(self, d_model=64, n_heads=8):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        The core attention mechanism.
        Q, K, V: Query, Key, Value matrices
        """
        # Calculate attention scores
        scores = np.dot(Q, K.T) / np.sqrt(self.d_k)
        
        # Apply mask if provided (for causal attention)
        if mask is not None:
            scores = scores + mask * -1e9
        
        # Apply softmax
        attention_weights = self.softmax(scores)
        
        # Apply attention to values
        output = np.dot(attention_weights, V)
        
        return output, attention_weights
    
    def softmax(self, x):
        """Stable softmax"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def single_head_attention(self, X):
        """Single attention head"""
        seq_len, d_model = X.shape
        
        # Initialize weight matrices (would be learned in practice)
        np.random.seed(42)
        W_q = np.random.randn(d_model, self.d_k) * 0.1
        W_k = np.random.randn(d_model, self.d_k) * 0.1
        W_v = np.random.randn(d_model, self.d_k) * 0.1
        
        # Compute Q, K, V
        Q = np.dot(X, W_q)
        K = np.dot(X, W_k)
        V = np.dot(X, W_v)
        
        # Apply attention
        output, weights = self.scaled_dot_product_attention(Q, K, V)
        
        return output, weights, Q, K, V
    
    def multi_head_attention(self, X):
        """Multi-head attention"""
        seq_len, d_model = X.shape
        outputs = []
        all_weights = []
        
        for head in range(self.n_heads):
            np.random.seed(42 + head)
            W_q = np.random.randn(d_model, self.d_k) * 0.1
            W_k = np.random.randn(d_model, self.d_k) * 0.1
            W_v = np.random.randn(d_model, self.d_k) * 0.1
            
            Q = np.dot(X, W_q)
            K = np.dot(X, W_k)
            V = np.dot(X, W_v)
            
            output, weights = self.scaled_dot_product_attention(Q, K, V)
            outputs.append(output)
            all_weights.append(weights)
        
        # Concatenate heads
        multi_head_output = np.concatenate(outputs, axis=-1)
        
        # Final linear layer (would be learned)
        W_o = np.random.randn(self.d_model, self.d_model) * 0.1
        final_output = np.dot(multi_head_output, W_o)
        
        return final_output, all_weights
    
    def visualize_attention_computation(self):
        """Step-by-step visualization of attention"""
        # Simple example
        seq_len = 5
        d_model = 4
        
        # Create simple input
        X = np.random.randn(seq_len, d_model)
        
        # Compute Q, K, V
        Q = X  # Simplified - normally would be X @ W_q
        K = X  # Simplified
        V = X  # Simplified
        
        # Compute scores
        scores = np.dot(Q, K.T) / np.sqrt(d_model)
        attention_weights = self.softmax(scores)
        output = np.dot(attention_weights, V)
        
        # Visualize
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Q, K, V matrices
        im1 = axes[0, 0].imshow(Q, cmap='coolwarm', aspect='auto')
        axes[0, 0].set_title('Query (Q)')
        axes[0, 0].set_ylabel('Position')
        axes[0, 0].set_xlabel('Dimension')
        plt.colorbar(im1, ax=axes[0, 0])
        
        im2 = axes[0, 1].imshow(K, cmap='coolwarm', aspect='auto')
        axes[0, 1].set_title('Key (K)')
        axes[0, 1].set_ylabel('Position')
        axes[0, 1].set_xlabel('Dimension')
        plt.colorbar(im2, ax=axes[0, 1])
        
        im3 = axes[0, 2].imshow(V, cmap='coolwarm', aspect='auto')
        axes[0, 2].set_title('Value (V)')
        axes[0, 2].set_ylabel('Position')
        axes[0, 2].set_xlabel('Dimension')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Attention computation
        im4 = axes[1, 0].imshow(scores, cmap='Blues', aspect='auto')
        axes[1, 0].set_title('Attention Scores\n(Q @ K.T)')
        axes[1, 0].set_ylabel('Query Position')
        axes[1, 0].set_xlabel('Key Position')
        plt.colorbar(im4, ax=axes[1, 0])
        
        im5 = axes[1, 1].imshow(attention_weights, cmap='Blues', aspect='auto')
        axes[1, 1].set_title('Attention Weights\n(After Softmax)')
        axes[1, 1].set_ylabel('Query Position')
        axes[1, 1].set_xlabel('Key Position')
        plt.colorbar(im5, ax=axes[1, 1])
        
        im6 = axes[1, 2].imshow(output, cmap='coolwarm', aspect='auto')
        axes[1, 2].set_title('Output\n(Attention @ V)')
        axes[1, 2].set_ylabel('Position')
        axes[1, 2].set_xlabel('Dimension')
        plt.colorbar(im6, ax=axes[1, 2])
        
        plt.suptitle('Self-Attention: Step-by-Step Computation', fontsize=14)
        plt.tight_layout()
        plt.savefig('attention_computation.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return scores, attention_weights, output
    
    def causal_vs_bidirectional(self):
        """Compare causal vs bidirectional attention"""
        seq_len = 8
        
        # Create causal mask
        causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        
        # Random Q, K, V
        np.random.seed(42)
        Q = K = V = np.random.randn(seq_len, self.d_k)
        
        # Bidirectional attention
        _, bidirectional_weights = self.scaled_dot_product_attention(Q, K, V)
        
        # Causal attention
        _, causal_weights = self.scaled_dot_product_attention(Q, K, V, mask=causal_mask)
        
        # Visualize
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        im1 = ax1.imshow(bidirectional_weights, cmap='Blues', vmin=0, vmax=0.5)
        ax1.set_title('Bidirectional Attention\n(Can see all positions)')
        ax1.set_xlabel('Key Position')
        ax1.set_ylabel('Query Position')
        plt.colorbar(im1, ax=ax1)
        
        im2 = ax2.imshow(causal_weights, cmap='Blues', vmin=0, vmax=0.5)
        ax2.set_title('Causal Attention\n(Can only see past)')
        ax2.set_xlabel('Key Position')
        ax2.set_ylabel('Query Position')
        plt.colorbar(im2, ax=ax2)
        
        # Add diagonal line to show causality boundary
        ax2.plot([0, seq_len], [0, seq_len], 'r--', alpha=0.5, linewidth=2)
        
        plt.suptitle('Causal vs Bidirectional Attention Patterns', fontsize=14)
        plt.tight_layout()
        plt.savefig('causal_vs_bidirectional.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def multi_head_visualization(self):
        """Visualize what different heads learn"""
        seq_len = 10
        sentence = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.']
        
        # Create input
        X = np.random.randn(seq_len, self.d_model)
        
        # Get multi-head attention
        _, all_weights = self.multi_head_attention(X)
        
        # Plot different heads
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        
        for head in range(min(8, self.n_heads)):
            im = axes[head].imshow(all_weights[head], cmap='Blues', vmin=0, vmax=0.5)
            axes[head].set_title(f'Head {head+1}')
            axes[head].set_xticks(range(seq_len))
            axes[head].set_yticks(range(seq_len))
            axes[head].set_xticklabels(sentence, rotation=45, ha='right', fontsize=8)
            axes[head].set_yticklabels(sentence, fontsize=8)
            
        plt.suptitle('Multi-Head Attention: Different Heads Learn Different Patterns', fontsize=14)
        plt.tight_layout()
        plt.savefig('multi_head_patterns.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Different heads typically learn:")
        print("- Head 1-2: Local/positional patterns")
        print("- Head 3-4: Syntactic relationships")  
        print("- Head 5-6: Semantic relationships")
        print("- Head 7-8: Long-range dependencies")

def run_experiments():
    print("Exploring Self-Attention Mechanism...")
    print("=" * 50)
    
    attention = SelfAttention(d_model=64, n_heads=8)
    
    # 1. Visualize computation
    print("\n1. Visualizing attention computation...")
    attention.visualize_attention_computation()
    print("   Saved: attention_computation.png")
    
    # 2. Causal vs bidirectional
    print("\n2. Comparing causal vs bidirectional...")
    attention.causal_vs_bidirectional()
    print("   Saved: causal_vs_bidirectional.png")
    
    # 3. Multi-head patterns
    print("\n3. Visualizing multi-head attention...")
    attention.multi_head_visualization()
    print("   Saved: multi_head_patterns.png")
    
    print("\n" + "=" * 50)
    print("KEY INSIGHTS:")
    print("1. Attention = 'How much should I look at each position?'")
    print("2. Scaled by sqrt(d_k) to prevent gradient vanishing")
    print("3. Causal masking ensures autoregressive generation")
    print("4. Multiple heads capture different relationship types")

if __name__ == "__main__":
    run_experiments()
