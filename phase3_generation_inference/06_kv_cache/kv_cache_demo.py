# kv_cache_demo.py
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List

class SimpleKVCache:
    """Simplified KV cache demonstration"""
    
    def __init__(self, max_seq_len: int, d_model: int):
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.cache_k = []
        self.cache_v = []
        
    def add(self, k, v):
        """Add key-value pair to cache"""
        self.cache_k.append(k)
        self.cache_v.append(v)
    
    def get_all(self):
        """Get all cached KV pairs"""
        if not self.cache_k:
            return None, None
        return np.array(self.cache_k), np.array(self.cache_v)
    
    def clear(self):
        """Clear cache"""
        self.cache_k.clear()
        self.cache_v.clear()

def simulate_generation_without_cache(seq_lengths):
    """Simulate computation cost without cache"""
    results = []
    for length in seq_lengths:
        # Without cache: compute attention for all previous tokens at each step
        # Cost = 1 + 2 + 3 + ... + n = n(n+1)/2
        total_ops = sum(range(1, length + 1))
        results.append(total_ops)
    return results

def simulate_generation_with_cache(seq_lengths):
    """Simulate computation cost with cache"""
    results = []
    for length in seq_lengths:
        # With cache: only compute new token's attention
        # Cost = 1 + 1 + 1 + ... (n times) = n
        total_ops = length
        results.append(total_ops)
    return results

def visualize_cache_mechanism():
    """Illustrate how KV cache works"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Without cache
    ax1 = axes[0]
    tokens = ['The', 'cat', 'sat', 'on', 'the']
    n_tokens = len(tokens)
    
    # Show recomputation for each step
    for step in range(n_tokens):
        y_pos = n_tokens - step - 1
        for i in range(step + 1):
            color = 'lightcoral' if i < step else 'darkred'
            ax1.add_patch(plt.Rectangle((i, y_pos), 0.9, 0.9, 
                                       facecolor=color, edgecolor='black'))
            ax1.text(i + 0.45, y_pos + 0.45, tokens[i], 
                    ha='center', va='center', fontsize=10)
    
    ax1.set_xlim(0, n_tokens)
    ax1.set_ylim(0, n_tokens)
    ax1.set_aspect('equal')
    ax1.set_title('Without Cache: Recompute Everything Each Step\n(Light Red = Recomputed, Dark Red = New)', 
                 fontsize=12, fontweight='bold')
    ax1.set_xlabel('Token Position')
    ax1.set_ylabel('Generation Step')
    ax1.invert_yaxis()
    
    # With cache
    ax2 = axes[1]
    for step in range(n_tokens):
        y_pos = n_tokens - step - 1
        for i in range(step + 1):
            if i < step:
                color = 'lightgreen'  # Cached
            else:
                color = 'darkgreen'  # New computation
            
            ax2.add_patch(plt.Rectangle((i, y_pos), 0.9, 0.9, 
                                       facecolor=color, edgecolor='black'))
            ax2.text(i + 0.45, y_pos + 0.45, tokens[i], 
                    ha='center', va='center', fontsize=10)
    
    ax2.set_xlim(0, n_tokens)
    ax2.set_ylim(0, n_tokens)
    ax2.set_aspect('equal')
    ax2.set_title('With Cache: Reuse Previous Computations\n(Light Green = Cached, Dark Green = New)', 
                 fontsize=12, fontweight='bold')
    ax2.set_xlabel('Token Position')
    ax2.set_ylabel('Generation Step')
    ax2.invert_yaxis()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightcoral', label='Wasted recomputation'),
        Patch(facecolor='darkred', label='New computation (no cache)'),
        Patch(facecolor='lightgreen', label='Cached (reused)'),
        Patch(facecolor='darkgreen', label='New computation (with cache)')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.savefig('cache_mechanism.png', dpi=150, bbox_inches='tight')
    plt.show()

def visualize_computational_savings():
    """Show computational complexity comparison"""
    lengths = list(range(1, 101))
    
    no_cache_ops = simulate_generation_without_cache(lengths)
    with_cache_ops = simulate_generation_with_cache(lengths)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Operations comparison
    ax1 = axes[0, 0]
    ax1.plot(lengths, no_cache_ops, 'r-', label='Without Cache O(n²)', linewidth=2)
    ax1.plot(lengths, with_cache_ops, 'g-', label='With Cache O(n)', linewidth=2)
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Total Operations')
    ax1.set_title('Computational Complexity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Speedup factor
    ax2 = axes[0, 1]
    speedup = [no / yes if yes > 0 else 0 for no, yes in zip(no_cache_ops, with_cache_ops)]
    ax2.plot(lengths, speedup, 'b-', linewidth=2)
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Cache Speedup (grows with length)')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No speedup')
    
    # 3. Wasted computation
    ax3 = axes[1, 0]
    wasted = [no - yes for no, yes in zip(no_cache_ops, with_cache_ops)]
    ax3.fill_between(lengths, wasted, alpha=0.3, color='red')
    ax3.plot(lengths, wasted, 'r-', linewidth=2)
    ax3.set_xlabel('Sequence Length')
    ax3.set_ylabel('Wasted Operations')
    ax3.set_title('Operations Saved by Cache')
    ax3.grid(True, alpha=0.3)
    
    # 4. Example specific lengths
    ax4 = axes[1, 1]
    example_lengths = [10, 50, 100, 200, 500]
    example_no_cache = simulate_generation_without_cache(example_lengths)
    example_with_cache = simulate_generation_with_cache(example_lengths)
    
    x = np.arange(len(example_lengths))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, example_no_cache, width, label='Without Cache', color='red', alpha=0.7)
    bars2 = ax4.bar(x + width/2, example_with_cache, width, label='With Cache', color='green', alpha=0.7)
    
    ax4.set_xlabel('Sequence Length')
    ax4.set_ylabel('Total Operations')
    ax4.set_title('Operations at Different Lengths')
    ax4.set_xticks(x)
    ax4.set_xticklabels(example_lengths)
    ax4.legend()
    
    # Add speedup annotations
    for i, (no, yes) in enumerate(zip(example_no_cache, example_with_cache)):
        speedup = no / yes
        ax4.text(i, max(no, yes) * 1.05, f'{speedup:.1f}x', 
                ha='center', fontsize=9, fontweight='bold')
    
    plt.suptitle('KV Cache: Computational Efficiency Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('kv_cache_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return example_lengths, example_no_cache, example_with_cache

def memory_analysis():
    """Analyze memory requirements for KV cache"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Memory vs sequence length
    seq_lengths = [512, 1024, 2048, 4096, 8192]
    
    configs = [
        ('Small\n2 layers', 2, 64, 4),
        ('Medium\n6 layers', 6, 128, 8),
        ('Large\n12 layers', 12, 256, 8),
        ('XL\n24 layers', 24, 512, 16)
    ]
    
    # Calculate memory for different configs
    for label, n_layers, d_model, n_heads in configs:
        d_k = d_model // n_heads
        memory_mb = []
        
        for seq_len in seq_lengths:
            # Memory = 2 (K and V) × n_layers × seq_len × n_heads × d_k × 4 bytes (float32)
            memory_bytes = 2 * n_layers * seq_len * n_heads * d_k * 4
            memory_mb.append(memory_bytes / (1024 ** 2))
        
        ax1.plot(seq_lengths, memory_mb, marker='o', label=label, linewidth=2)
    
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Memory (MB)')
    ax1.set_title('KV Cache Memory vs Sequence Length')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    
    # Memory breakdown for 2048 context
    config_labels = [c[0].replace('\n', ' ') for c in configs]
    memory_2048 = []
    
    for _, n_layers, d_model, n_heads in configs:
        d_k = d_model // n_heads
        memory_bytes = 2 * n_layers * 2048 * n_heads * d_k * 4
        memory_2048.append(memory_bytes / (1024 ** 2))
    
    bars = ax2.bar(config_labels, memory_2048, color='steelblue', alpha=0.7)
    ax2.set_ylabel('Memory (MB)')
    ax2.set_title('KV Cache Memory at 2048 Context\n(Different Model Sizes)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, mem in zip(bars, memory_2048):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{mem:.1f} MB',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('cache_memory_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def run_experiments():
    print("KV Cache Optimization Demo")
    print("=" * 50)
    
    print("\n1. Visualizing cache mechanism...")
    visualize_cache_mechanism()
    print("   Saved: cache_mechanism.png")
    
    print("\n2. Analyzing computational savings...")
    lengths, no_cache, with_cache = visualize_computational_savings()
    print("   Saved: kv_cache_analysis.png")
    
    print("\n3. Example computations:")
    for length, no, yes in zip(lengths, no_cache, with_cache):
        speedup = no / yes
        print(f"   Length {length:3d}: {no:6d} ops → {yes:3d} ops ({speedup:.1f}x speedup)")
    
    print("\n4. Memory analysis...")
    memory_analysis()
    print("   Saved: cache_memory_analysis.png")
    
    print("\n" + "=" * 50)
    print("KEY INSIGHTS:")
    print("1. Without cache: O(n²) operations (quadratic growth)")
    print("2. With cache: O(n) operations (linear growth)")
    print("3. Speedup increases with sequence length")
    print("4. Example: 100 tokens → 50x speedup, 500 tokens → 250x speedup")
    print("5. Memory cost: ~0.5-2MB per 1K tokens for small models")
    print("6. Critical for production: enables real-time generation")
    print("\nTrade-off: Memory for speed (always worth it for inference!)")

if __name__ == "__main__":
    run_experiments()
