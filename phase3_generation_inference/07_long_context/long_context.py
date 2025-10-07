# long_context.py
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

class AttentionPatterns:
    """Different attention patterns for long context"""
    
    def __init__(self, seq_len: int, n_heads: int = 8):
        self.seq_len = seq_len
        self.n_heads = n_heads
    
    def full_attention(self) -> np.ndarray:
        """Standard O(n²) full attention"""
        # Causal mask: can only attend to current and past
        mask = np.triu(np.ones((self.seq_len, self.seq_len)), k=1)
        attention = np.ones((self.seq_len, self.seq_len))
        attention[mask == 1] = 0
        return attention
    
    def sliding_window_attention(self, window_size: int = 256) -> np.ndarray:
        """Local attention - only look at nearby tokens"""
        attention = np.zeros((self.seq_len, self.seq_len))
        
        for i in range(self.seq_len):
            # Attend to tokens within window
            start = max(0, i - window_size)
            end = i + 1
            attention[i, start:end] = 1
            
        return attention
    
    def strided_attention(self, stride: int = 64) -> np.ndarray:
        """Attend to every k-th token (dilated/strided)"""
        attention = np.zeros((self.seq_len, self.seq_len))
        
        for i in range(self.seq_len):
            # Local window
            local_start = max(0, i - 128)
            attention[i, local_start:i+1] = 1
            
            # Strided global tokens
            for j in range(0, i, stride):
                attention[i, j] = 1
                
        return attention
    
    def block_sparse_attention(self, block_size: int = 64) -> np.ndarray:
        """Block-wise sparse attention (BigBird style)"""
        attention = np.zeros((self.seq_len, self.seq_len))
        n_blocks = self.seq_len // block_size
        
        for i in range(self.seq_len):
            block_idx = i // block_size
            
            # Local block
            block_start = block_idx * block_size
            block_end = min((block_idx + 1) * block_size, self.seq_len)
            attention[i, block_start:min(block_end, i+1)] = 1
            
            # Random global connections (simplified)
            if i > 0:
                global_indices = np.random.choice(i, min(8, i), replace=False)
                attention[i, global_indices] = 1
                
        return attention

class LongContextMetrics:
    """Measure complexity and memory for different patterns"""
    
    def __init__(self):
        pass
    
    def compute_complexity(self, attention_pattern: np.ndarray) -> dict:
        """Calculate computational and memory complexity"""
        seq_len = attention_pattern.shape[0]
        
        # Count non-zero elements (actual computations)
        actual_ops = np.sum(attention_pattern > 0)
        
        # Full attention would be
        full_ops = seq_len * seq_len
        
        # Memory (attention matrix storage)
        memory_mb = (attention_pattern.size * 4) / (1024 ** 2)  # float32
        
        return {
            'actual_ops': int(actual_ops),
            'full_ops': int(full_ops),
            'sparsity': 1 - (actual_ops / full_ops),
            'memory_mb': memory_mb,
            'speedup': full_ops / actual_ops if actual_ops > 0 else 0
        }

def visualize_attention_patterns():
    """Compare different long-context attention patterns"""
    seq_len = 512
    patterns_maker = AttentionPatterns(seq_len)
    
    patterns = {
        'Full Attention\n(Standard)': patterns_maker.full_attention(),
        'Sliding Window\n(window=128)': patterns_maker.sliding_window_attention(128),
        'Strided\n(stride=32)': patterns_maker.strided_attention(32),
        'Block Sparse\n(block=64)': patterns_maker.block_sparse_attention(64)
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    metrics = LongContextMetrics()
    
    for idx, (name, pattern) in enumerate(patterns.items()):
        ax = axes[idx]
        
        # Visualize pattern
        im = ax.imshow(pattern, cmap='Blues', aspect='auto', interpolation='nearest')
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        
        # Calculate metrics
        stats = metrics.compute_complexity(pattern)
        
        # Add text box with metrics
        textstr = f"Ops: {stats['actual_ops']:,}\n"
        textstr += f"Sparsity: {stats['sparsity']:.1%}\n"
        textstr += f"Speedup: {stats['speedup']:.1f}x"
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Long Context Attention Patterns\n(Blue = Attend, White = Ignore)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('attention_patterns.png', dpi=150, bbox_inches='tight')
    plt.show()

def complexity_scaling_analysis():
    """Show how complexity scales with sequence length"""
    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384]
    
    results = {
        'Full': [],
        'Sliding (256)': [],
        'Strided (64)': [],
        'Block Sparse': []
    }
    
    metrics_calc = LongContextMetrics()
    
    for seq_len in seq_lengths:
        patterns = AttentionPatterns(seq_len)
        
        # Full attention
        full = patterns.full_attention()
        results['Full'].append(metrics_calc.compute_complexity(full)['actual_ops'])
        
        # Sliding window
        sliding = patterns.sliding_window_attention(256)
        results['Sliding (256)'].append(metrics_calc.compute_complexity(sliding)['actual_ops'])
        
        # Strided
        strided = patterns.strided_attention(64)
        results['Strided (64)'].append(metrics_calc.compute_complexity(strided)['actual_ops'])
        
        # Block sparse
        block = patterns.block_sparse_attention(64)
        results['Block Sparse'].append(metrics_calc.compute_complexity(block)['actual_ops'])
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Linear scale
    for name, ops in results.items():
        ax1.plot(seq_lengths, ops, marker='o', label=name, linewidth=2)
    
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Operations')
    ax1.set_title('Computational Complexity (Linear Scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Log scale
    for name, ops in results.items():
        ax2.plot(seq_lengths, ops, marker='o', label=name, linewidth=2)
    
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Operations (log scale)')
    ax2.set_title('Computational Complexity (Log Scale)')
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.suptitle('Scaling: How Operations Grow with Context Length', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('complexity_scaling.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return seq_lengths, results

def memory_bandwidth_analysis():
    """Analyze memory requirements"""
    seq_lengths = [1024, 2048, 4096, 8192, 16384, 32768]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Memory usage
    full_memory = [(s * s * 4) / (1024 ** 2) for s in seq_lengths]  # MB
    sparse_memory = [(s * 256 * 4) / (1024 ** 2) for s in seq_lengths]  # Sliding window
    
    ax1.plot(seq_lengths, full_memory, 'r-o', label='Full Attention', linewidth=2)
    ax1.plot(seq_lengths, sparse_memory, 'g-s', label='Sliding Window', linewidth=2)
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Memory (MB)')
    ax1.set_title('Attention Matrix Memory Usage')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    
    # Memory savings
    savings = [(f - s) / f * 100 for f, s in zip(full_memory, sparse_memory)]
    
    bars = ax2.bar(range(len(seq_lengths)), savings, color='steelblue', alpha=0.7)
    ax2.set_xticks(range(len(seq_lengths)))
    ax2.set_xticklabels([f'{s//1024}K' for s in seq_lengths])
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Memory Saved (%)')
    ax2.set_title('Memory Savings with Sparse Attention')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    for bar, saving in zip(bars, savings):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{saving:.1f}%',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('memory_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def information_flow_comparison():
    """Show what information each pattern can access"""
    seq_len = 128
    patterns = AttentionPatterns(seq_len)
    
    # Pick a specific query position
    query_pos = 100
    
    patterns_dict = {
        'Full': patterns.full_attention()[query_pos],
        'Sliding\nWindow': patterns.sliding_window_attention(32)[query_pos],
        'Strided': patterns.strided_attention(16)[query_pos],
        'Block\nSparse': patterns.block_sparse_attention(16)[query_pos]
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.ravel()
    
    for idx, (name, pattern) in enumerate(patterns_dict.items()):
        ax = axes[idx]
        
        # Create visualization
        positions = np.arange(seq_len)
        colors = ['green' if p > 0 else 'lightgray' for p in pattern]
        
        ax.bar(positions, np.ones(seq_len), color=colors, alpha=0.7, width=1.0)
        ax.axvline(query_pos, color='red', linestyle='--', linewidth=2, label='Query Position')
        
        ax.set_xlim(0, seq_len)
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Can Attend')
        ax.set_title(f'{name}\n({np.sum(pattern > 0)} tokens visible)', fontweight='bold')
        ax.legend()
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['No', 'Yes'])
    
    plt.suptitle(f'Information Access at Query Position {query_pos}\n(Green = Visible, Gray = Hidden)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('information_flow.png', dpi=150, bbox_inches='tight')
    plt.show()

def run_experiments():
    print("Long Context Attention Patterns")
    print("=" * 50)
    
    print("\n1. Visualizing different attention patterns...")
    visualize_attention_patterns()
    print("   Saved: attention_patterns.png")
    
    print("\n2. Analyzing complexity scaling...")
    seq_lengths, results = complexity_scaling_analysis()
    print("   Saved: complexity_scaling.png")
    
    print("\n3. Complexity comparison at 16K context:")
    idx = seq_lengths.index(16384)
    for name, ops in results.items():
        print(f"   {name:20s}: {ops[idx]:,} operations")
    
    print("\n4. Memory bandwidth analysis...")
    memory_bandwidth_analysis()
    print("   Saved: memory_analysis.png")
    
    print("\n5. Information flow comparison...")
    information_flow_comparison()
    print("   Saved: information_flow.png")
    
    print("\n" + "=" * 50)
    print("KEY INSIGHTS:")
    print("1. Full attention: O(n²) - impractical beyond 8K tokens")
    print("2. Sliding window: O(n×w) - linear scaling, loses distant context")
    print("3. Strided: O(n×(w+n/s)) - balance between local and global")
    print("4. Block sparse: Combines local, global, and random for best trade-off")
    print("5. Flash Attention: Doesn't change pattern, but optimizes memory access")
    print("\nProduction models use combinations:")
    print("- LongFormer: Sliding window + global tokens")
    print("- BigBird: Block + sliding + random")
    print("- Llama 2: RoPE + grouped query for long context")

if __name__ == "__main__":
    run_experiments()
