# grouped_query_attention.py
import numpy as np
import matplotlib.pyplot as plt
import time

class MultiHeadAttention:
    """Standard multi-head attention (MHA)"""
    
    def __init__(self, d_model: int, n_heads: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Separate KV for each head
        np.random.seed(42)
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass"""
        batch_size, seq_len, _ = x.shape
        
        # Project and reshape
        Q = np.dot(x, self.W_q).reshape(batch_size, seq_len, self.n_heads, self.d_k)
        K = np.dot(x, self.W_k).reshape(batch_size, seq_len, self.n_heads, self.d_k)
        V = np.dot(x, self.W_v).reshape(batch_size, seq_len, self.n_heads, self.d_k)
        
        # Transpose for attention
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        # Attention
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        weights = self.softmax(scores)
        context = np.matmul(weights, V)
        
        # Concatenate and project
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        output = np.dot(context, self.W_o)
        
        return output
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def count_kv_parameters(self):
        """Count KV parameters"""
        return self.W_k.size + self.W_v.size

class MultiQueryAttention:
    """Multi-query attention (MQA) - single KV for all heads"""
    
    def __init__(self, d_model: int, n_heads: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Single KV shared across all heads
        np.random.seed(42)
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, self.d_k) * 0.02  # Smaller!
        self.W_v = np.random.randn(d_model, self.d_k) * 0.02  # Smaller!
        self.W_o = np.random.randn(d_model, d_model) * 0.02
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass"""
        batch_size, seq_len, _ = x.shape
        
        # Project Q with multiple heads
        Q = np.dot(x, self.W_q).reshape(batch_size, seq_len, self.n_heads, self.d_k)
        Q = Q.transpose(0, 2, 1, 3)
        
        # Project K, V once (shared across heads)
        K = np.dot(x, self.W_k)  # (batch, seq, d_k)
        V = np.dot(x, self.W_v)  # (batch, seq, d_k)
        
        # Expand K, V to match Q heads
        K = np.expand_dims(K, 1).repeat(self.n_heads, axis=1)  # (batch, heads, seq, d_k)
        V = np.expand_dims(V, 1).repeat(self.n_heads, axis=1)
        
        # Attention
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        weights = self.softmax(scores)
        context = np.matmul(weights, V)
        
        # Concatenate and project
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        output = np.dot(context, self.W_o)
        
        return output
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def count_kv_parameters(self):
        """Count KV parameters"""
        return self.W_k.size + self.W_v.size

class GroupedQueryAttention:
    """Grouped query attention (GQA) - KV shared within groups"""
    
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_k = d_model // n_heads
        
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.group_size = n_heads // n_kv_heads
        
        # KV for each group
        np.random.seed(42)
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, n_kv_heads * self.d_k) * 0.02
        self.W_v = np.random.randn(d_model, n_kv_heads * self.d_k) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass"""
        batch_size, seq_len, _ = x.shape
        
        # Project Q with all heads
        Q = np.dot(x, self.W_q).reshape(batch_size, seq_len, self.n_heads, self.d_k)
        Q = Q.transpose(0, 2, 1, 3)
        
        # Project K, V with fewer heads
        K = np.dot(x, self.W_k).reshape(batch_size, seq_len, self.n_kv_heads, self.d_k)
        V = np.dot(x, self.W_v).reshape(batch_size, seq_len, self.n_kv_heads, self.d_k)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        # Expand K, V to match Q heads (repeat each KV group_size times)
        K = np.repeat(K, self.group_size, axis=1)
        V = np.repeat(V, self.group_size, axis=1)
        
        # Attention
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        weights = self.softmax(scores)
        context = np.matmul(weights, V)
        
        # Concatenate and project
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        output = np.dot(context, self.W_o)
        
        return output
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def count_kv_parameters(self):
        """Count KV parameters"""
        return self.W_k.size + self.W_v.size

def visualize_attention_architectures():
    """Show the difference between MHA, MQA, and GQA"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    n_heads = 8
    
    # MHA - each head has its own KV
    ax1 = axes[0]
    for i in range(n_heads):
        # Query
        ax1.add_patch(plt.Rectangle((0, i), 0.8, 0.8, facecolor='lightblue', 
                                    edgecolor='black', linewidth=2))
        ax1.text(0.4, i+0.4, f'Q{i}', ha='center', va='center', fontsize=10)
        
        # Key
        ax1.add_patch(plt.Rectangle((2, i), 0.8, 0.8, facecolor='lightcoral', 
                                    edgecolor='black', linewidth=2))
        ax1.text(2.4, i+0.4, f'K{i}', ha='center', va='center', fontsize=10)
        
        # Value
        ax1.add_patch(plt.Rectangle((4, i), 0.8, 0.8, facecolor='lightgreen', 
                                    edgecolor='black', linewidth=2))
        ax1.text(4.4, i+0.4, f'V{i}', ha='center', va='center', fontsize=10)
        
        # Connections
        ax1.plot([0.8, 2], [i+0.4, i+0.4], 'k-', alpha=0.3)
        ax1.plot([2.8, 4], [i+0.4, i+0.4], 'k-', alpha=0.3)
    
    ax1.set_xlim(-0.5, 5.5)
    ax1.set_ylim(-0.5, n_heads)
    ax1.set_title('Multi-Head Attention (MHA)\n8 Q, 8 K, 8 V', fontweight='bold')
    ax1.axis('off')
    
    # MQA - single KV shared by all heads
    ax2 = axes[1]
    for i in range(n_heads):
        # Query
        ax2.add_patch(plt.Rectangle((0, i), 0.8, 0.8, facecolor='lightblue', 
                                    edgecolor='black', linewidth=2))
        ax2.text(0.4, i+0.4, f'Q{i}', ha='center', va='center', fontsize=10)
        
        # Connection to shared KV
        ax2.plot([0.8, 2], [i+0.4, 3.5], 'k-', alpha=0.3)
        ax2.plot([2.8, 4], [3.5, 3.5], 'k-', alpha=0.3)
    
    # Shared Key
    ax2.add_patch(plt.Rectangle((2, 3), 0.8, 0.8, facecolor='lightcoral', 
                                edgecolor='black', linewidth=3))
    ax2.text(2.4, 3.4, 'K', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Shared Value
    ax2.add_patch(plt.Rectangle((4, 3), 0.8, 0.8, facecolor='lightgreen', 
                                edgecolor='black', linewidth=3))
    ax2.text(4.4, 3.4, 'V', ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax2.set_xlim(-0.5, 5.5)
    ax2.set_ylim(-0.5, n_heads)
    ax2.set_title('Multi-Query Attention (MQA)\n8 Q, 1 K, 1 V', fontweight='bold')
    ax2.axis('off')
    
    # GQA - grouped KV
    ax3 = axes[2]
    n_groups = 2
    heads_per_group = n_heads // n_groups
    colors_kv = ['lightcoral', 'coral']
    colors_v = ['lightgreen', 'green']
    
    for i in range(n_heads):
        # Query
        ax3.add_patch(plt.Rectangle((0, i), 0.8, 0.8, facecolor='lightblue', 
                                    edgecolor='black', linewidth=2))
        ax3.text(0.4, i+0.4, f'Q{i}', ha='center', va='center', fontsize=10)
        
        # Connection to group KV
        group = i // heads_per_group
        kv_pos = group * 2 + 2
        ax3.plot([0.8, 2], [i+0.4, kv_pos+0.4], 'k-', alpha=0.3)
        ax3.plot([2.8, 4], [kv_pos+0.4, kv_pos+0.4], 'k-', alpha=0.3)
    
    # Group KVs
    for g in range(n_groups):
        y_pos = g * 2 + 2
        # Key
        ax3.add_patch(plt.Rectangle((2, y_pos), 0.8, 0.8, facecolor=colors_kv[g], 
                                    edgecolor='black', linewidth=2))
        ax3.text(2.4, y_pos+0.4, f'K{g}', ha='center', va='center', fontsize=11, fontweight='bold')
        
        # Value
        ax3.add_patch(plt.Rectangle((4, y_pos), 0.8, 0.8, facecolor=colors_v[g], 
                                    edgecolor='black', linewidth=2))
        ax3.text(4.4, y_pos+0.4, f'V{g}', ha='center', va='center', fontsize=11, fontweight='bold')
    
    ax3.set_xlim(-0.5, 5.5)
    ax3.set_ylim(-0.5, n_heads)
    ax3.set_title('Grouped Query Attention (GQA)\n8 Q, 2 K, 2 V', fontweight='bold')
    ax3.axis('off')
    
    plt.suptitle('Attention Architectures Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('attention_architectures.png', dpi=150, bbox_inches='tight')
    plt.show()

def compare_parameters_and_memory():
    """Compare parameter counts and memory usage"""
    d_model = 512
    n_heads = 8
    seq_len = 2048
    
    configs = [
        ('MHA\n8 heads', MultiHeadAttention(d_model, n_heads)),
        ('GQA-4\n4 groups', GroupedQueryAttention(d_model, n_heads, 4)),
        ('GQA-2\n2 groups', GroupedQueryAttention(d_model, n_heads, 2)),
        ('MQA\n1 group', MultiQueryAttention(d_model, n_heads))
    ]
    
    results = {
        'names': [],
        'kv_params': [],
        'kv_cache_memory': [],
        'param_reduction': []
    }
    
    mha_params = configs[0][1].count_kv_parameters()
    
    for name, model in configs:
        kv_params = model.count_kv_parameters()
        
        # KV cache memory (for inference with caching)
        # Memory = seq_len × (K + V dimensions) × 4 bytes
        if isinstance(model, MultiHeadAttention):
            kv_dims = d_model * 2  # Full KV for all heads
        elif isinstance(model, MultiQueryAttention):
            kv_dims = (d_model // n_heads) * 2  # Single KV
        else:  # GQA
            kv_dims = (d_model // n_heads) * model.n_kv_heads * 2
        
        cache_memory = (seq_len * kv_dims * 4) / (1024 ** 2)  # MB
        
        results['names'].append(name)
        results['kv_params'].append(kv_params / 1e6)  # Millions
        results['kv_cache_memory'].append(cache_memory)
        results['param_reduction'].append((1 - kv_params / mha_params) * 100)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. KV Parameters
    ax1 = axes[0]
    bars = ax1.bar(range(len(results['names'])), results['kv_params'], 
                   color=['red', 'orange', 'yellow', 'green'], alpha=0.7)
    ax1.set_xticks(range(len(results['names'])))
    ax1.set_xticklabels(results['names'])
    ax1.set_ylabel('Parameters (Millions)')
    ax1.set_title('KV Parameters')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, results['kv_params']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}M',
                ha='center', va='bottom', fontsize=9)
    
    # 2. KV Cache Memory
    ax2 = axes[1]
    bars = ax2.bar(range(len(results['names'])), results['kv_cache_memory'], 
                   color=['red', 'orange', 'yellow', 'green'], alpha=0.7)
    ax2.set_xticks(range(len(results['names'])))
    ax2.set_xticklabels(results['names'])
    ax2.set_ylabel('Memory (MB)')
    ax2.set_title(f'KV Cache Memory\n({seq_len} tokens)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, results['kv_cache_memory']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f} MB',
                ha='center', va='bottom', fontsize=9)
    
    # 3. Reduction percentage
    ax3 = axes[2]
    bars = ax3.bar(range(len(results['names'])), results['param_reduction'], 
                   color=['red', 'orange', 'yellow', 'green'], alpha=0.7)
    ax3.set_xticks(range(len(results['names'])))
    ax3.set_xticklabels(results['names'])
    ax3.set_ylabel('Reduction (%)')
    ax3.set_title('Parameter Reduction vs MHA')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, results['param_reduction']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('GQA: Parameter and Memory Efficiency', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('gqa_efficiency.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return results

def scaling_analysis():
    """Show how benefits scale with model size"""
    d_models = [128, 256, 512, 1024, 2048]
    n_heads = 8
    n_kv_heads = 2  # GQA with 2 groups
    
    mha_memory = []
    gqa_memory = []
    mqa_memory = []
    
    for d_model in d_models:
        # KV cache size (per token)
        mha_kv = d_model * 2 * 4  # bytes
        gqa_kv = (d_model // n_heads) * n_kv_heads * 2 * 4
        mqa_kv = (d_model // n_heads) * 2 * 4
        
        mha_memory.append(mha_kv)
        gqa_memory.append(gqa_kv)
        mqa_memory.append(mqa_kv)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Memory per token
    ax1.plot(d_models, mha_memory, 'ro-', label='MHA', linewidth=2)
    ax1.plot(d_models, gqa_memory, 'go-', label='GQA (2 groups)', linewidth=2)
    ax1.plot(d_models, mqa_memory, 'bo-', label='MQA', linewidth=2)
    ax1.set_xlabel('Model Dimension')
    ax1.set_ylabel('KV Cache per Token (bytes)')
    ax1.set_title('Memory Scaling with Model Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log', base=2)
    
    # Speedup factor
    speedup_gqa = [m / g for m, g in zip(mha_memory, gqa_memory)]
    speedup_mqa = [m / q for m, q in zip(mha_memory, mqa_memory)]
    
    ax2.plot(d_models, speedup_gqa, 'go-', label='GQA vs MHA', linewidth=2)
    ax2.plot(d_models, speedup_mqa, 'bo-', label='MQA vs MHA', linewidth=2)
    ax2.set_xlabel('Model Dimension')
    ax2.set_ylabel('Memory Bandwidth Reduction')
    ax2.set_title('Relative Memory Efficiency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('gqa_scaling.png', dpi=150, bbox_inches='tight')
    plt.show()

def run_experiments():
    print("Grouped Query Attention (GQA) Analysis")
    print("=" * 50)
    
    print("\n1. Visualizing attention architectures...")
    visualize_attention_architectures()
    print("   Saved: attention_architectures.png")
    
    print("\n2. Comparing parameters and memory...")
    results = compare_parameters_and_memory()
    print("   Saved: gqa_efficiency.png")
    
    print("\n3. Efficiency summary (d_model=512, 8 heads):")
    for i, name in enumerate(results['names']):
        print(f"   {name:15s}: {results['kv_params'][i]:.2f}M params, "
              f"{results['kv_cache_memory'][i]:.1f} MB cache, "
              f"{results['param_reduction'][i]:.1f}% reduction")
    
    print("\n4. Scaling analysis...")
    scaling_analysis()
    print("   Saved: gqa_scaling.png")
    
    print("\n" + "=" * 50)
    print("KEY INSIGHTS:")
    print("1. MHA: Full KV for each head (most memory)")
    print("2. GQA: KV shared within groups (balanced)")
    print("3. MQA: Single KV for all heads (least memory, some quality loss)")
    print("4. GQA trades off between MHA quality and MQA efficiency")
    print("5. Critical for large models: reduces memory bandwidth bottleneck")
    print("6. Used in: Llama 2, Mistral, Falcon")
    print("\nTrade-off: Slight quality loss for major memory savings")

if __name__ == "__main__":
    run_experiments()
