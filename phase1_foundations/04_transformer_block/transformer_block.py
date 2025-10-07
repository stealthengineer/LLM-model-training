# transformer_block.py
import numpy as np
import matplotlib.pyplot as plt

class TransformerBlock:
    """Complete transformer block with all components"""
    
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout_rate=0.1):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.d_k = d_model // n_heads
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize all weight matrices"""
        # Multi-head attention weights
        self.W_q = np.random.randn(self.d_model, self.d_model) * np.sqrt(2.0 / self.d_model)
        self.W_k = np.random.randn(self.d_model, self.d_model) * np.sqrt(2.0 / self.d_model)
        self.W_v = np.random.randn(self.d_model, self.d_model) * np.sqrt(2.0 / self.d_model)
        self.W_o = np.random.randn(self.d_model, self.d_model) * np.sqrt(2.0 / self.d_model)
        
        # Feed-forward network weights
        self.W_ff1 = np.random.randn(self.d_model, self.d_ff) * np.sqrt(2.0 / self.d_model)
        self.b_ff1 = np.zeros(self.d_ff)
        self.W_ff2 = np.random.randn(self.d_ff, self.d_model) * np.sqrt(2.0 / self.d_ff)
        self.b_ff2 = np.zeros(self.d_model)
        
        # Layer norm parameters
        self.ln1_gamma = np.ones(self.d_model)
        self.ln1_beta = np.zeros(self.d_model)
        self.ln2_gamma = np.ones(self.d_model)
        self.ln2_beta = np.zeros(self.d_model)
    
    def layer_norm(self, x, gamma, beta, eps=1e-5):
        """Layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma * x_norm + beta
    
    def multi_head_attention(self, x, mask=None):
        """Multi-head attention mechanism"""
        batch_size, seq_len, _ = x.shape
        
        # Linear projections in batch from d_model => h x d_k
        Q = np.dot(x, self.W_q).reshape(batch_size, seq_len, self.n_heads, self.d_k)
        K = np.dot(x, self.W_k).reshape(batch_size, seq_len, self.n_heads, self.d_k)
        V = np.dot(x, self.W_v).reshape(batch_size, seq_len, self.n_heads, self.d_k)
        
        # Transpose for attention computation
        Q = Q.transpose(0, 2, 1, 3)  # (batch, heads, seq, d_k)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        # Attention scores
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores += mask * -1e9
            
        # Softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # Apply attention to values
        context = np.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        
        # Final linear projection
        output = np.dot(context, self.W_o)
        
        return output, attention_weights
    
    def feed_forward(self, x):
        """Position-wise feed-forward network"""
        # First linear layer with ReLU
        hidden = np.dot(x, self.W_ff1) + self.b_ff1
        hidden = np.maximum(0, hidden)  # ReLU activation
        
        # Second linear layer
        output = np.dot(hidden, self.W_ff2) + self.b_ff2
        
        return output
    
    def forward(self, x, mask=None):
        """Complete forward pass through transformer block"""
        # Multi-head attention with residual connection and layer norm
        attn_output, attn_weights = self.multi_head_attention(x, mask)
        x = self.layer_norm(x + attn_output, self.ln1_gamma, self.ln1_beta)
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.layer_norm(x + ff_output, self.ln2_gamma, self.ln2_beta)
        
        return x, attn_weights
    
    def visualize_block_architecture(self):
        """Visualize the transformer block architecture"""
        fig, ax = plt.subplots(figsize=(10, 12))
        
        # Component boxes
        components = [
            ('Input', 0.5, 0.1, 'lightblue'),
            ('Multi-Head\nAttention', 0.5, 0.25, 'lightgreen'),
            ('Add & Norm', 0.5, 0.35, 'lightyellow'),
            ('Feed Forward', 0.5, 0.5, 'lightcoral'),
            ('Add & Norm', 0.5, 0.6, 'lightyellow'),
            ('Output', 0.5, 0.75, 'lightblue')
        ]
        
        for name, x, y, color in components:
            rect = plt.Rectangle((x-0.15, y-0.03), 0.3, 0.06, 
                                facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(x, y, name, ha='center', va='center', fontsize=11, weight='bold')
        
        # Connections
        connections = [
            (0.5, 0.13, 0.5, 0.22),  # Input to MHA
            (0.5, 0.28, 0.5, 0.32),  # MHA to Add&Norm
            (0.5, 0.38, 0.5, 0.47),  # Add&Norm to FF
            (0.5, 0.53, 0.5, 0.57),  # FF to Add&Norm
            (0.5, 0.63, 0.5, 0.72),  # Add&Norm to Output
        ]
        
        for x1, y1, x2, y2 in connections:
            ax.arrow(x1, y1, x2-x1, y2-y1, head_width=0.02, head_length=0.015, 
                    fc='black', ec='black', linewidth=2)
        
        # Residual connections
        residuals = [
            (0.35, 0.13, 0.35, 0.35, 'Input'),
            (0.35, 0.38, 0.35, 0.60, 'After Attention')
        ]
        
        for x1, y1, x2, y2, label in residuals:
            ax.plot([x1, x1, x2, x2], [y1, y2, y2, y2], 'r--', linewidth=2, alpha=0.7)
            ax.text(x1-0.05, (y1+y2)/2, label, rotation=90, va='center', 
                   fontsize=9, color='red', alpha=0.7)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.85)
        ax.axis('off')
        ax.set_title('Transformer Block Architecture', fontsize=14, weight='bold')
        
        plt.tight_layout()
        plt.savefig('transformer_architecture.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def gradient_flow_visualization(self):
        """Visualize gradient flow with and without residuals"""
        seq_len = 10
        
        # Simulate gradient flow
        depths = range(1, 13)
        
        # Without residual connections
        grad_without_residual = [0.9 ** d for d in depths]
        
        # With residual connections
        grad_with_residual = [1.0 - 0.05 * d for d in depths]
        grad_with_residual = [max(0.4, g) for g in grad_with_residual]
        
        plt.figure(figsize=(10, 6))
        plt.plot(depths, grad_without_residual, 'r-', linewidth=2, 
                label='Without Residuals', marker='o')
        plt.plot(depths, grad_with_residual, 'g-', linewidth=2, 
                label='With Residuals', marker='s')
        
        plt.xlabel('Layer Depth', fontsize=12)
        plt.ylabel('Gradient Magnitude', fontsize=12)
        plt.title('Gradient Flow: Impact of Residual Connections', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0.1, color='black', linestyle=':', alpha=0.5)
        plt.text(6, 0.15, 'Vanishing gradient threshold', fontsize=10, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('gradient_flow.png', dpi=150, bbox_inches='tight')
        plt.show()

def run_experiments():
    print("Building Complete Transformer Block...")
    print("=" * 50)
    
    # Create transformer block
    transformer = TransformerBlock(d_model=512, n_heads=8, d_ff=2048)
    
    # Test with dummy data
    batch_size = 2
    seq_len = 10
    x = np.random.randn(batch_size, seq_len, 512)
    
    print("\n1. Forward pass through transformer block...")
    output, attention_weights = transformer.forward(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Attention shape: {attention_weights.shape}")
    
    print("\n2. Visualizing architecture...")
    transformer.visualize_block_architecture()
    print("   Saved: transformer_architecture.png")
    
    print("\n3. Analyzing gradient flow...")
    transformer.gradient_flow_visualization()
    print("   Saved: gradient_flow.png")
    
    print("\n" + "=" * 50)
    print("KEY COMPONENTS:")
    print("1. Multi-Head Attention: Parallel attention mechanisms")
    print("2. Layer Normalization: Stabilizes training")
    print("3. Residual Connections: Prevents gradient vanishing")
    print("4. Feed-Forward Network: Position-wise transformations")
    print("\nThis block can be stacked to create deep transformers!")

if __name__ == "__main__":
    run_experiments()
