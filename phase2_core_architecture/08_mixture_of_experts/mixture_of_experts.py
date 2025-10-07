# mixture_of_experts.py
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class ExpertNetwork:
    """Single expert - a small feed-forward network"""
    
    def __init__(self, d_model: int, d_ff: int, expert_id: int):
        self.d_model = d_model
        self.d_ff = d_ff
        self.expert_id = expert_id
        
        # Initialize weights
        np.random.seed(42 + expert_id)
        self.W1 = np.random.randn(d_model, d_ff) * 0.02
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.02
        self.b2 = np.zeros(d_model)
        
        # Track usage
        self.call_count = 0
        self.total_tokens = 0
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through expert"""
        self.call_count += 1
        self.total_tokens += x.shape[0]
        
        # Two-layer FFN with GELU
        hidden = np.dot(x, self.W1) + self.b1
        hidden = self.gelu(hidden)
        output = np.dot(hidden, self.W2) + self.b2
        return output
    
    def gelu(self, x):
        """GELU activation"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    def reset_stats(self):
        """Reset usage statistics"""
        self.call_count = 0
        self.total_tokens = 0

class Router:
    """Router network - decides which experts to use"""
    
    def __init__(self, d_model: int, n_experts: int):
        self.d_model = d_model
        self.n_experts = n_experts
        
        # Router weights
        np.random.seed(42)
        self.W_router = np.random.randn(d_model, n_experts) * 0.02
        self.b_router = np.zeros(n_experts)
    
    def route(self, x: np.ndarray, top_k: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Determine which experts to use for each token
        
        Returns:
            expert_indices: (batch_size, top_k) - which experts to use
            expert_weights: (batch_size, top_k) - how much to weight each expert
        """
        # Compute routing logits
        logits = np.dot(x, self.W_router) + self.b_router  # (batch, n_experts)
        
        # Get top-k experts
        top_k_indices = np.argsort(logits, axis=-1)[:, -top_k:]  # (batch, top_k)
        
        # Softmax over top-k
        top_k_logits = np.take_along_axis(logits, top_k_indices, axis=-1)
        exp_logits = np.exp(top_k_logits - np.max(top_k_logits, axis=-1, keepdims=True))
        expert_weights = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        return top_k_indices, expert_weights

class MixtureOfExperts:
    """MoE layer - sparse activation of experts"""
    
    def __init__(self, d_model: int, d_ff: int, n_experts: int, top_k: int = 2):
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = n_experts
        self.top_k = top_k
        
        # Create experts
        self.experts = [ExpertNetwork(d_model, d_ff, i) for i in range(n_experts)]
        
        # Create router
        self.router = Router(d_model, n_experts)
        
        # Track routing decisions
        self.routing_history = []
    
    def forward(self, x: np.ndarray, track_routing: bool = True) -> np.ndarray:
        """
        Forward pass through MoE layer
        
        x: (batch_size, d_model)
        """
        batch_size = x.shape[0]
        
        # Route tokens to experts
        expert_indices, expert_weights = self.router.route(x, self.top_k)
        
        if track_routing:
            self.routing_history.append((expert_indices.copy(), expert_weights.copy()))
        
        # Compute expert outputs
        output = np.zeros_like(x)
        
        for i in range(batch_size):
            token_output = np.zeros(self.d_model)
            
            for k in range(self.top_k):
                expert_idx = expert_indices[i, k]
                expert_weight = expert_weights[i, k]
                
                # Pass token through expert
                expert_out = self.experts[expert_idx].forward(x[i:i+1])
                token_output += expert_weight * expert_out[0]
            
            output[i] = token_output
        
        return output
    
    def get_expert_utilization(self) -> np.ndarray:
        """Get how many tokens each expert processed"""
        return np.array([expert.total_tokens for expert in self.experts])
    
    def compute_flops_savings(self) -> dict:
        """Calculate FLOP savings vs dense model"""
        # Dense: all tokens through all experts
        dense_flops = self.d_model * self.d_ff * 2  # per token
        
        # Sparse: only top_k experts active
        sparse_flops = (self.d_model * self.d_ff * 2) * (self.top_k / self.n_experts)
        
        return {
            'dense_flops_per_token': dense_flops,
            'sparse_flops_per_token': sparse_flops,
            'flop_reduction': 1 - (sparse_flops / dense_flops),
            'theoretical_speedup': dense_flops / sparse_flops
        }
    
    def reset_stats(self):
        """Reset all statistics"""
        for expert in self.experts:
            expert.reset_stats()
        self.routing_history = []

class DenseFFN:
    """Standard dense feed-forward network for comparison"""
    
    def __init__(self, d_model: int, d_ff: int):
        self.d_model = d_model
        self.d_ff = d_ff
        
        np.random.seed(42)
        self.W1 = np.random.randn(d_model, d_ff) * 0.02
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.02
        self.b2 = np.zeros(d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass"""
        hidden = np.dot(x, self.W1) + self.b1
        hidden = 0.5 * hidden * (1 + np.tanh(np.sqrt(2 / np.pi) * (hidden + 0.044715 * hidden**3)))
        output = np.dot(hidden, self.W2) + self.b2
        return output

def visualize_expert_routing():
    """Show how tokens get routed to experts"""
    d_model = 64
    n_experts = 8
    n_tokens = 100
    
    # Create MoE
    moe = MixtureOfExperts(d_model, d_ff=256, n_experts=n_experts, top_k=2)
    
    # Generate random tokens
    np.random.seed(42)
    tokens = np.random.randn(n_tokens, d_model)
    
    # Process through MoE
    _ = moe.forward(tokens)
    
    # Get routing decisions
    expert_indices, expert_weights = moe.routing_history[0]
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Routing heatmap
    ax1 = axes[0, 0]
    routing_matrix = np.zeros((n_tokens, n_experts))
    for i in range(n_tokens):
        for k in range(2):
            expert_idx = expert_indices[i, k]
            weight = expert_weights[i, k]
            routing_matrix[i, expert_idx] = weight
    
    im1 = ax1.imshow(routing_matrix.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax1.set_xlabel('Token Index')
    ax1.set_ylabel('Expert Index')
    ax1.set_title('Token-to-Expert Routing Pattern')
    plt.colorbar(im1, ax=ax1, label='Routing Weight')
    
    # 2. Expert utilization
    ax2 = axes[0, 1]
    utilization = moe.get_expert_utilization()
    bars = ax2.bar(range(n_experts), utilization, color='steelblue', alpha=0.7)
    ax2.set_xlabel('Expert ID')
    ax2.set_ylabel('Tokens Processed')
    ax2.set_title('Expert Utilization')
    ax2.axhline(y=n_tokens/n_experts, color='red', linestyle='--', 
                label=f'Balanced ({n_tokens/n_experts:.0f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    for bar, count in zip(bars, utilization):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=9)
    
    # 3. Weight distribution
    ax3 = axes[1, 0]
    weights_flat = expert_weights.flatten()
    ax3.hist(weights_flat, bins=30, color='green', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Routing Weight')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Routing Weights')
    ax3.axvline(x=0.5, color='red', linestyle='--', label='Equal weight')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Load balance visualization
    ax4 = axes[1, 1]
    # Calculate load imbalance (coefficient of variation)
    mean_load = np.mean(utilization)
    std_load = np.std(utilization)
    cv = std_load / mean_load if mean_load > 0 else 0
    
    ax4.pie(utilization, labels=[f'E{i}' for i in range(n_experts)],
            autopct='%1.1f%%', startangle=90)
    ax4.set_title(f'Expert Load Distribution\nCV: {cv:.3f} (lower is better)')
    
    plt.suptitle('Mixture of Experts: Routing Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('expert_routing.png', dpi=150, bbox_inches='tight')
    plt.show()

def compare_moe_vs_dense():
    """Compare MoE with dense FFN"""
    d_model = 128
    d_ff = 512
    n_experts_list = [4, 8, 16, 32]
    
    results = {
        'n_experts': [],
        'flop_reduction': [],
        'speedup': [],
        'params_moe': [],
        'params_dense': []
    }
    
    # Dense baseline parameters
    dense_params = d_model * d_ff * 2  # W1 and W2
    
    for n_experts in n_experts_list:
        moe = MixtureOfExperts(d_model, d_ff, n_experts, top_k=2)
        
        # Calculate parameters
        params_per_expert = d_model * d_ff * 2
        moe_params = params_per_expert * n_experts + (d_model * n_experts)  # experts + router
        
        # Get FLOP savings
        flops = moe.compute_flops_savings()
        
        results['n_experts'].append(n_experts)
        results['flop_reduction'].append(flops['flop_reduction'] * 100)
        results['speedup'].append(flops['theoretical_speedup'])
        results['params_moe'].append(moe_params / 1e6)  # Millions
        results['params_dense'].append(dense_params / 1e6)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 1. FLOP reduction
    ax1 = axes[0]
    ax1.bar(range(len(n_experts_list)), results['flop_reduction'], 
            color='coral', alpha=0.7)
    ax1.set_xticks(range(len(n_experts_list)))
    ax1.set_xticklabels([f'{n} experts' for n in n_experts_list])
    ax1.set_ylabel('FLOP Reduction (%)')
    ax1.set_title('Computational Savings\n(with top-k=2)')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for i, val in enumerate(results['flop_reduction']):
        ax1.text(i, val, f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # 2. Speedup
    ax2 = axes[1]
    ax2.plot(n_experts_list, results['speedup'], 'o-', linewidth=2, 
             markersize=8, color='green')
    ax2.set_xlabel('Number of Experts')
    ax2.set_ylabel('Theoretical Speedup')
    ax2.set_title('Speedup vs Number of Experts')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    
    # 3. Parameters comparison
    ax3 = axes[2]
    x = np.arange(len(n_experts_list))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, results['params_dense'], width, 
                    label='Dense', color='red', alpha=0.7)
    bars2 = ax3.bar(x + width/2, results['params_moe'], width, 
                    label='MoE', color='blue', alpha=0.7)
    
    ax3.set_xlabel('Configuration')
    ax3.set_ylabel('Parameters (Millions)')
    ax3.set_title('Parameter Count: Dense vs MoE')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{n}E' for n in n_experts_list])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('MoE Efficiency Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('moe_efficiency.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return results

def load_balancing_experiment():
    """Show importance of load balancing"""
    d_model = 64
    n_experts = 8
    n_tokens = 200
    
    # Create MoE
    moe = MixtureOfExperts(d_model, d_ff=256, n_experts=n_experts, top_k=2)
    
    # Simulate biased routing (some tokens prefer certain experts)
    np.random.seed(42)
    
    # Create token clusters
    cluster_centers = np.random.randn(4, d_model)
    tokens = []
    for i in range(n_tokens):
        cluster = i % 4
        token = cluster_centers[cluster] + np.random.randn(d_model) * 0.1
        tokens.append(token)
    
    tokens = np.array(tokens)
    
    # Process
    _ = moe.forward(tokens)
    
    # Analyze load
    utilization = moe.get_expert_utilization()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Expert utilization
    ax1.bar(range(n_experts), utilization, color='steelblue', alpha=0.7)
    ax1.axhline(y=n_tokens/n_experts, color='red', linestyle='--', 
                linewidth=2, label='Perfect Balance')
    ax1.set_xlabel('Expert ID')
    ax1.set_ylabel('Tokens Processed')
    ax1.set_title('Load Imbalance Problem')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Efficiency loss
    max_load = np.max(utilization)
    avg_load = np.mean(utilization)
    efficiency = (avg_load / max_load) * 100
    
    experts_x = range(n_experts)
    ax2.bar(experts_x, [max_load] * n_experts, alpha=0.3, color='red', 
            label='Wasted Capacity')
    ax2.bar(experts_x, utilization, color='green', alpha=0.7, 
            label='Used Capacity')
    
    ax2.set_xlabel('Expert ID')
    ax2.set_ylabel('Capacity')
    ax2.set_title(f'Efficiency: {efficiency:.1f}%\n(Bottlenecked by busiest expert)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('load_balancing.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"   Max load: {int(max_load)} tokens")
    print(f"   Avg load: {avg_load:.1f} tokens")
    print(f"   Efficiency: {efficiency:.1f}%")
    print(f"   Wasted capacity: {(1 - efficiency/100)*100:.1f}%")

def run_experiments():
    print("Mixture of Experts (MoE) Analysis")
    print("=" * 50)
    
    print("\n1. Visualizing expert routing...")
    visualize_expert_routing()
    print("   Saved: expert_routing.png")
    
    print("\n2. Comparing MoE vs Dense...")
    results = compare_moe_vs_dense()
    print("   Saved: moe_efficiency.png")
    
    print("\n3. Efficiency summary:")
    for i, n in enumerate(results['n_experts']):
        print(f"   {n} experts: {results['flop_reduction'][i]:.1f}% FLOP reduction, "
              f"{results['speedup'][i]:.1f}x speedup")
    
    print("\n4. Load balancing analysis...")
    load_balancing_experiment()
    print("   Saved: load_balancing.png")
    
    print("\n" + "=" * 50)
    print("KEY INSIGHTS:")
    print("1. Sparse activation: Only top-k experts active per token")
    print("2. FLOP savings scale with number of experts")
    print("3. 8 experts with top-2 → 75% FLOP reduction")
    print("4. More parameters but same computation cost")
    print("5. Load balancing is critical for efficiency")
    print("6. Used in: GPT-4, Mixtral, Switch Transformer")
    print("\nTrade-off: Model capacity vs serving complexity")

if __name__ == "__main__":
    run_experiments()
