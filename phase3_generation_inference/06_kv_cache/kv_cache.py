# kv_cache.py
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Optional

class KVCache:
    """Key-Value cache for efficient autoregressive generation"""
    
    def __init__(self, max_batch_size: int, max_seq_len: int, n_layers: int, n_heads: int, d_k: int):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_k = d_k
        
        # Initialize cache storage
        self.keys = np.zeros((n_layers, max_batch_size, max_seq_len, n_heads, d_k))
        self.values = np.zeros((n_layers, max_batch_size, max_seq_len, n_heads, d_k))
        self.current_length = 0
        
    def update(self, layer_idx: int, new_keys: np.ndarray, new_values: np.ndarray):
        """Add new keys and values to cache"""
        batch_size, n_new_tokens, n_heads, d_k = new_keys.shape
        
        # Store new KV pairs
        start_idx = self.current_length
        end_idx = start_idx + n_new_tokens
        
        self.keys[layer_idx, :batch_size, start_idx:end_idx] = new_keys
        self.values[layer_idx, :batch_size, start_idx:end_idx] = new_values
        
        return start_idx, end_idx
    
    def get(self, layer_idx: int, batch_size: int, seq_len: int):
        """Retrieve cached keys and values"""
        return (
            self.keys[layer_idx, :batch_size, :seq_len],
            self.values[layer_idx, :batch_size, :seq_len]
        )
    
    def advance(self):
        """Move to next token position"""
        self.current_length += 1
        
    def reset(self):
        """Clear cache"""
        self.current_length = 0
        self.keys.fill(0)
        self.values.fill(0)
    
    def memory_usage(self):
        """Calculate memory usage in MB"""
        total_elements = self.keys.size + self.values.size
        bytes_per_element = 4  # float32
        return (total_elements * bytes_per_element) / (1024 ** 2)

class NaiveGenerator:
    """Generation without KV cache - recomputes everything"""
    
    def __init__(self, model):
        self.model = model
        self.computation_count = 0
        
    def generate(self, prompt_ids: List[int], max_length: int = 20) -> List[int]:
        """Generate text by recomputing from start each time"""
        generated = list(prompt_ids)
        self.computation_count = 0
        
        for _ in range(max_length - len(prompt_ids)):
            # Full forward pass through entire sequence
            input_ids = np.array(generated).reshape(1, -1)
            logits, _ = self.model.forward(input_ids)
            
            # Count operations (proportional to sequence length)
            self.computation_count += len(generated)
            
            # Sample next token
            next_token_logits = logits[0, -1, :]
            next_token = self.model.sample_top_p(next_token_logits, temperature=0.8)
            
            generated.append(next_token)
            
            if next_token == 0:  # End token
                break
                
        return generated

class CachedGenerator:
    """Generation with KV cache - only computes new tokens"""
    
    def __init__(self, model):
        self.model = model
        self.computation_count = 0
        self.cache = None
        
    def generate(self, prompt_ids: List[int], max_length: int = 20) -> List[int]:
        """Generate text using KV cache"""
        generated = list(prompt_ids)
        self.computation_count = 0
        
        # Initialize cache
        self.cache = KVCache(
            max_batch_size=1,
            max_seq_len=max_length,
            n_layers=self.model.n_layers,
            n_heads=self.model.n_heads,
            d_k=self.model.d_k
        )
        
        # Process prompt (full pass needed first time)
        input_ids = np.array(prompt_ids).reshape(1, -1)
        logits, attention_maps = self.model.forward(input_ids)
        
        # Cache KV from prompt processing
        self._cache_attention_states(attention_maps, len(prompt_ids))
        self.computation_count += len(prompt_ids)
        
        # Sample first generated token
        next_token_logits = logits[0, -1, :]
        next_token = self.model.sample_top_p(next_token_logits, temperature=0.8)
        generated.append(next_token)
        
        # Generate remaining tokens (only new token computed)
        for _ in range(max_length - len(prompt_ids) - 1):
            # Only process new token
            input_ids = np.array([next_token]).reshape(1, 1)
            logits, attention_maps = self.model.forward_with_cache(input_ids, self.cache)
            
            # Only computed 1 token
            self.computation_count += 1
            
            next_token_logits = logits[0, -1, :]
            next_token = self.model.sample_top_p(next_token_logits, temperature=0.8)
            
            generated.append(next_token)
            
            if next_token == 0:
                break
                
        return generated
    
    def _cache_attention_states(self, attention_maps, seq_len):
        """Store attention KV states in cache"""
        # This is a simplified version - in practice you'd cache actual K,V tensors
        self.cache.current_length = seq_len

def benchmark_generation(model, prompt_ids, lengths=[10, 20, 50, 100]):
    """Compare naive vs cached generation"""
    results = {
        'lengths': lengths,
        'naive_time': [],
        'cached_time': [],
        'naive_ops': [],
        'cached_ops': [],
        'speedup': []
    }
    
    naive_gen = NaiveGenerator(model)
    cached_gen = CachedGenerator(model)
    
    for length in lengths:
        print(f"\nBenchmarking length={length}")
        
        # Naive generation
        start = time.time()
        _ = naive_gen.generate(prompt_ids, max_length=length)
        naive_time = time.time() - start
        naive_ops = naive_gen.computation_count
        
        # Cached generation
        start = time.time()
        _ = cached_gen.generate(prompt_ids, max_length=length)
        cached_time = time.time() - start
        cached_ops = cached_gen.computation_count
        
        speedup = naive_time / cached_time if cached_time > 0 else 0
        
        results['naive_time'].append(naive_time)
        results['cached_time'].append(cached_time)
        results['naive_ops'].append(naive_ops)
        results['cached_ops'].append(cached_ops)
        results['speedup'].append(speedup)
        
        print(f"  Naive: {naive_time:.3f}s ({naive_ops} ops)")
        print(f"  Cached: {cached_time:.3f}s ({cached_ops} ops)")
        print(f"  Speedup: {speedup:.2f}x")
    
    return results

def visualize_cache_efficiency(results):
    """Visualize cache benefits"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    lengths = results['lengths']
    
    # 1. Time comparison
    ax1 = axes[0, 0]
    ax1.plot(lengths, results['naive_time'], 'r-o', label='Without Cache', linewidth=2)
    ax1.plot(lengths, results['cached_time'], 'g-s', label='With Cache', linewidth=2)
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Generation Time: Naive vs Cached')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Operations count
    ax2 = axes[0, 1]
    ax2.plot(lengths, results['naive_ops'], 'r-o', label='Without Cache', linewidth=2)
    ax2.plot(lengths, results['cached_ops'], 'g-s', label='With Cache', linewidth=2)
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Computation Steps')
    ax2.set_title('Total Operations: O(n²) vs O(n)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Speedup
    ax3 = axes[1, 0]
    ax3.bar(range(len(lengths)), results['speedup'], color='steelblue', alpha=0.7)
    ax3.set_xticks(range(len(lengths)))
    ax3.set_xticklabels(lengths)
    ax3.set_xlabel('Sequence Length')
    ax3.set_ylabel('Speedup Factor')
    ax3.set_title('Cache Speedup (Higher is Better)')
    ax3.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Memory usage estimate
    ax4 = axes[1, 1]
    # Estimate memory for different configs
    configs = [
        ('Small\n(4 layers)', 4, 64, 4),
        ('Medium\n(8 layers)', 8, 128, 8),
        ('Large\n(12 layers)', 12, 256, 8),
        ('XL\n(24 layers)', 24, 512, 16)
    ]
    
    memory_usage = []
    for _, n_layers, d_model, n_heads in configs:
        cache = KVCache(
            max_batch_size=1,
            max_seq_len=2048,
            n_layers=n_layers,
            n_heads=n_heads,
            d_k=d_model // n_heads
        )
        memory_usage.append(cache.memory_usage())
    
    labels = [c[0] for c in configs]
    ax4.bar(labels, memory_usage, color='coral', alpha=0.7)
    ax4.set_ylabel('Memory (MB)')
    ax4.set_title('KV Cache Memory Usage\n(2048 token context)')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('KV Cache Efficiency Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('kv_cache_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

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
    ax1.set_title('Without Cache: Recompute Everything Each Step\n(Red = Recomputed)', 
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
                label = 'cached'
            else:
                color = 'darkgreen'  # New computation
                label = 'new'
            
            ax2.add_patch(plt.Rectangle((i, y_pos), 0.9, 0.9, 
                                       facecolor=color, edgecolor='black'))
            ax2.text(i + 0.45, y_pos + 0.45, tokens[i] if i <= step else '', 
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
        Patch(facecolor='darkred', label='Necessary computation (no cache)'),
        Patch(facecolor='lightgreen', label='Cached (reused)'),
        Patch(facecolor='darkgreen', label='New computation (cached)')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.savefig('cache_mechanism.png', dpi=150, bbox_inches='tight')
    plt.show()

def run_experiments():
    print("Exploring KV Cache Optimization...")
    print("=" * 50)
    
    # Import Mini-GPT
    import sys
    sys.path.append('../mini_gpt')
    from mini_gpt import MiniGPT, create_toy_dataset
    
    # Create model and dataset
    vocab, token_to_id, id_to_token, tokenized = create_toy_dataset()
    model = MiniGPT(
        vocab_size=len(vocab),
        d_model=64,
        n_heads=4,
        n_layers=2,
        seq_len=128
    )
    
    # Test prompt
    prompt = "the cat"
    prompt_ids = [token_to_id.get(t, 0) for t in prompt.split()]
    
    print("\n1. Visualizing cache mechanism...")
    visualize_cache_mechanism()
    print("   Saved: cache_mechanism.png")
    
    print("\n2. Benchmarking generation speeds...")
    results = benchmark_generation(model, prompt_ids, lengths=[10, 20, 30, 40])
    
    print("\n3. Creating efficiency visualizations...")
    visualize_cache_efficiency(results)
    print("   Saved: kv_cache_analysis.png")
    
    # Memory analysis
    print("\n4. Memory usage analysis:")
    for length in [512, 1024, 2048, 4096]:
        cache = KVCache(
            max_batch_size=1,
            max_seq_len=length,
            n_layers=model.n_layers,
            n_heads=model.n_heads,
            d_k=model.d_k
        )
        print(f"   Context {length}: {cache.memory_usage():.2f} MB")
    
    print("\n" + "=" * 50)
    print("KEY INSIGHTS:")
    print("1. KV cache trades memory for speed")
    print("2. Speedup grows with sequence length (O(n²) → O(n))")
    print("3. Critical for real-time chat applications")
    print("4. Memory cost scales with: layers × heads × d_k × seq_len")
    print("\nFor production: manage cache carefully for long conversations!")

if __name__ == "__main__":
    run_experiments()
