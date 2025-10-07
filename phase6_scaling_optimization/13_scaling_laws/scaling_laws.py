# scaling_laws.py
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

class ScalingLaws:
    """Understand and predict model performance scaling"""
    
    def __init__(self):
        # Chinchilla scaling law parameters (approximate)
        self.alpha = 0.34  # Loss ~ N^(-alpha)
        self.beta = 0.28   # Loss ~ D^(-beta)
        self.gamma = 0.37  # Loss ~ C^(-gamma)
        
    def loss_vs_params(self, N: np.ndarray, D: float = 1e10, C: float = None) -> np.ndarray:
        """
        Predict loss based on number of parameters
        N: number of parameters
        D: dataset size (tokens)
        """
        # Simplified Chinchilla formula
        L = 1.69 + 406.4 * (N ** -0.34) + 410.7 * (D ** -0.28)
        return L
    
    def loss_vs_data(self, D: np.ndarray, N: float = 1e9) -> np.ndarray:
        """Predict loss based on dataset size"""
        L = 1.69 + 406.4 * (N ** -0.34) + 410.7 * (D ** -0.28)
        return L
    
    def loss_vs_compute(self, C: np.ndarray) -> np.ndarray:
        """
        Predict loss based on compute budget
        C: FLOPs (compute)
        """
        # Compute = 6ND (approximation)
        # Chinchilla optimal: N ∝ C^0.5, D ∝ C^0.5
        L = 1.69 + 110 * (C ** -0.37)
        return L
    
    def chinchilla_optimal(self, compute_budget: float) -> Tuple[float, float]:
        """
        Given compute budget, find optimal N and D
        Chinchilla: N and D should scale equally with compute
        """
        # C = 6ND (training FLOPs)
        # Optimal: N ∝ C^0.5, D ∝ C^0.5
        
        # Chinchilla coefficients (approximate)
        N_optimal = (compute_budget / 6) ** 0.5 / 20
        D_optimal = (compute_budget / 6) ** 0.5 * 20
        
        return N_optimal, D_optimal
    
    def compute_flops(self, N: float, D: float) -> float:
        """Calculate training FLOPs"""
        # Training: 6ND (forward + backward pass)
        return 6 * N * D

def visualize_scaling_laws():
    """Visualize the three main scaling relationships"""
    scaling = ScalingLaws()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Loss vs Model Size
    ax1 = axes[0, 0]
    params = np.logspace(6, 11, 50)  # 1M to 100B parameters
    
    # Different dataset sizes
    for D in [1e9, 1e10, 1e11]:
        losses = scaling.loss_vs_params(params, D=D)
        label = f'D = {D/1e9:.0f}B tokens'
        ax1.plot(params, losses, linewidth=2, label=label)
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Model Parameters')
    ax1.set_ylabel('Loss')
    ax1.set_title('Scaling Law: Loss vs Model Size\n(Power law: L ∝ N^-α)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add reference points
    models = {
        'GPT-2': 1.5e9,
        'GPT-3': 175e9,
        'LLaMA 7B': 7e9,
        'LLaMA 70B': 70e9
    }
    for name, size in models.items():
        loss = scaling.loss_vs_params(np.array([size]), D=1e11)[0]
        ax1.plot(size, loss, 'ro', markersize=8)
        ax1.text(size, loss * 1.05, name, fontsize=8, ha='center')
    
    # 2. Loss vs Dataset Size
    ax2 = axes[0, 1]
    data_sizes = np.logspace(9, 12, 50)  # 1B to 1T tokens
    
    # Different model sizes
    for N in [1e8, 1e9, 1e10]:
        losses = scaling.loss_vs_data(data_sizes, N=N)
        label = f'N = {N/1e9:.1f}B params'
        ax2.plot(data_sizes, losses, linewidth=2, label=label)
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Dataset Size (tokens)')
    ax2.set_ylabel('Loss')
    ax2.set_title('Scaling Law: Loss vs Data Size\n(Power law: L ∝ D^-β)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Loss vs Compute
    ax3 = axes[1, 0]
    compute = np.logspace(18, 24, 50)  # FLOPs
    losses = scaling.loss_vs_compute(compute)
    
    ax3.plot(compute, losses, linewidth=2, color='green')
    ax3.set_xscale('log')
    ax3.set_xlabel('Compute Budget (FLOPs)')
    ax3.set_ylabel('Loss')
    ax3.set_title('Scaling Law: Loss vs Compute\n(Power law: L ∝ C^-γ)')
    ax3.grid(True, alpha=0.3)
    
    # Add compute budgets for reference
    compute_examples = {
        'GPT-2': 6 * 1.5e9 * 40e9,
        'GPT-3': 6 * 175e9 * 300e9,
        'LLaMA 65B': 6 * 65e9 * 1.4e12
    }
    for name, c in compute_examples.items():
        loss = scaling.loss_vs_compute(np.array([c]))[0]
        ax3.plot(c, loss, 'ro', markersize=8)
        ax3.text(c, loss * 1.05, name, fontsize=8, ha='center')
    
    # 4. Chinchilla optimal allocation
    ax4 = axes[1, 1]
    compute_budgets = np.logspace(18, 24, 20)
    
    N_optimal = []
    D_optimal = []
    
    for C in compute_budgets:
        N, D = scaling.chinchilla_optimal(C)
        N_optimal.append(N)
        D_optimal.append(D)
    
    ax4_twin = ax4.twinx()
    
    line1 = ax4.plot(compute_budgets, N_optimal, 'b-', linewidth=2, label='Model Size')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_xlabel('Compute Budget (FLOPs)')
    ax4.set_ylabel('Optimal Model Size (parameters)', color='b')
    ax4.tick_params(axis='y', labelcolor='b')
    
    line2 = ax4_twin.plot(compute_budgets, D_optimal, 'r-', linewidth=2, label='Dataset Size')
    ax4_twin.set_yscale('log')
    ax4_twin.set_ylabel('Optimal Dataset Size (tokens)', color='r')
    ax4_twin.tick_params(axis='y', labelcolor='r')
    
    ax4.set_title('Chinchilla Optimal Allocation\n(N and D scale equally with compute)')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Scaling Laws for Language Models', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('scaling_laws.png', dpi=150, bbox_inches='tight')
    plt.show()

def compare_training_strategies():
    """Compare different training strategies"""
    scaling = ScalingLaws()
    
    # Fixed compute budget
    compute_budget = 1e21  # FLOPs
    
    strategies = {
        'Chinchilla Optimal': {},
        'Overtrain Small': {},
        'Undertrain Large': {}
    }
    
    # Chinchilla optimal
    N_opt, D_opt = scaling.chinchilla_optimal(compute_budget)
    strategies['Chinchilla Optimal'] = {
        'N': N_opt,
        'D': D_opt,
        'loss': scaling.loss_vs_params(np.array([N_opt]), D=D_opt)[0],
        'C': compute_budget
    }
    
    # Overtrain small model (more data, smaller model)
    N_small = N_opt / 4
    D_large = compute_budget / (6 * N_small)
    strategies['Overtrain Small'] = {
        'N': N_small,
        'D': D_large,
        'loss': scaling.loss_vs_params(np.array([N_small]), D=D_large)[0],
        'C': compute_budget
    }
    
    # Undertrain large model (less data, bigger model)
    N_large = N_opt * 4
    D_small = compute_budget / (6 * N_large)
    strategies['Undertrain Large'] = {
        'N': N_large,
        'D': D_small,
        'loss': scaling.loss_vs_params(np.array([N_large]), D=D_small)[0],
        'C': compute_budget
    }
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Strategy comparison
    names = list(strategies.keys())
    losses = [strategies[s]['loss'] for s in names]
    colors = ['green', 'blue', 'red']
    
    bars = ax1.bar(names, losses, color=colors, alpha=0.7)
    ax1.set_ylabel('Final Loss')
    ax1.set_title('Training Strategy Comparison\n(Same compute budget)')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, loss in zip(bars, losses):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Model size vs data trade-off
    x = np.arange(len(names))
    width = 0.35
    
    model_sizes = [strategies[s]['N'] / 1e9 for s in names]
    data_sizes = [strategies[s]['D'] / 1e9 for s in names]
    
    bars1 = ax2.bar(x - width/2, model_sizes, width, label='Model Size (B params)', color='blue', alpha=0.7)
    bars2 = ax2.bar(x + width/2, data_sizes, width, label='Data Size (B tokens)', color='orange', alpha=0.7)
    
    ax2.set_ylabel('Size (Billions)')
    ax2.set_title('Model Size vs Training Data\n(Resource allocation)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, ha='right')
    ax2.legend()
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('training_strategies.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return strategies

def simulate_model_family():
    """Simulate training a family of models at different scales"""
    scaling = ScalingLaws()
    
    # Model family (similar to LLaMA)
    models = {
        '1B': {'N': 1e9, 'D': 200e9},
        '7B': {'N': 7e9, 'D': 1e12},
        '13B': {'N': 13e9, 'D': 1e12},
        '33B': {'N': 33e9, 'D': 1.4e12},
        '65B': {'N': 65e9, 'D': 1.4e12}
    }
    
    results = {}
    for name, config in models.items():
        N, D = config['N'], config['D']
        loss = scaling.loss_vs_params(np.array([N]), D=D)[0]
        compute = scaling.compute_flops(N, D)
        
        results[name] = {
            'loss': loss,
            'compute': compute,
            'params': N,
            'data': D
        }
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    names = list(results.keys())
    
    # Loss
    ax1 = axes[0, 0]
    losses = [results[n]['loss'] for n in names]
    bars = ax1.bar(names, losses, color='steelblue', alpha=0.7)
    ax1.set_ylabel('Loss')
    ax1.set_title('Model Performance')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, loss in zip(bars, losses):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    # Compute
    ax2 = axes[0, 1]
    compute = [results[n]['compute'] / 1e21 for n in names]  # In ZettaFLOPs
    bars = ax2.bar(names, compute, color='orange', alpha=0.7)
    ax2.set_ylabel('Training Compute (ZettaFLOPs)')
    ax2.set_title('Compute Requirements')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, c in zip(bars, compute):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{c:.1f}',
                ha='center', va='bottom', fontsize=9)
    
    # Params vs Data
    ax3 = axes[1, 0]
    params = [results[n]['params'] / 1e9 for n in names]
    data = [results[n]['data'] / 1e9 for n in names]
    
    ax3.scatter(params, data, s=200, alpha=0.6, c=range(len(names)), cmap='viridis')
    
    for i, name in enumerate(names):
        ax3.annotate(name, (params[i], data[i]), fontsize=10, ha='center')
    
    ax3.set_xlabel('Model Size (B parameters)')
    ax3.set_ylabel('Training Data (B tokens)')
    ax3.set_title('Model Size vs Training Data')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Efficiency (loss per compute)
    ax4 = axes[1, 1]
    efficiency = [1/results[n]['loss'] / (results[n]['compute']/1e21) for n in names]
    bars = ax4.bar(names, efficiency, color='green', alpha=0.7)
    ax4.set_ylabel('Efficiency (1/Loss per ZettaFLOP)')
    ax4.set_title('Training Efficiency')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Model Family Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('model_family.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return results

def extrapolation_analysis():
    """Show how scaling laws allow extrapolation"""
    scaling = ScalingLaws()
    
    # "Observed" data (pretend we trained these)
    observed_sizes = np.array([1e8, 5e8, 1e9, 5e9, 1e10])  # 100M to 10B
    observed_losses = scaling.loss_vs_params(observed_sizes, D=1e11)
    
    # Add some noise to simulate real measurements
    np.random.seed(42)
    observed_losses += np.random.randn(len(observed_losses)) * 0.05
    
    # Extrapolate to larger sizes
    all_sizes = np.logspace(8, 11.5, 50)
    predicted_losses = scaling.loss_vs_params(all_sizes, D=1e11)
    
    # Future models
    future_sizes = np.array([50e9, 100e9, 175e9])  # 50B, 100B, 175B
    future_losses = scaling.loss_vs_params(future_sizes, D=1e11)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Extrapolation plot
    ax1.scatter(observed_sizes, observed_losses, s=100, color='blue', 
               label='Observed (trained)', zorder=3)
    ax1.plot(all_sizes, predicted_losses, 'g--', linewidth=2, 
            label='Scaling law prediction', alpha=0.7)
    ax1.scatter(future_sizes, future_losses, s=150, color='red', marker='*',
               label='Extrapolated (future)', zorder=3)
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Model Parameters')
    ax1.set_ylabel('Loss')
    ax1.set_title('Scaling Law Extrapolation\n(Predict future model performance)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add annotations for future models
    future_names = ['50B', '100B', 'GPT-3\n(175B)']
    for size, loss, name in zip(future_sizes, future_losses, future_names):
        ax1.annotate(name, (size, loss), xytext=(10, 10), 
                    textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Prediction accuracy over time
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Simulate prediction errors
    sizes_for_error = observed_sizes
    actual = observed_losses
    predicted = scaling.loss_vs_params(sizes_for_error, D=1e11)
    errors = ((predicted - actual) / actual) * 100
    
    ax2.bar(range(len(errors)), errors, color=['green' if e > 0 else 'red' for e in errors], 
           alpha=0.7)
    ax2.set_xticks(range(len(errors)))
    ax2.set_xticklabels([f'{s/1e9:.1f}B' for s in sizes_for_error])
    ax2.set_xlabel('Model Size')
    ax2.set_ylabel('Prediction Error (%)')
    ax2.set_title('Scaling Law Accuracy\n(Error in loss prediction)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for i, err in enumerate(errors):
        ax2.text(i, err, f'{err:.1f}%',
                ha='center', va='bottom' if err > 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('scaling_extrapolation.png', dpi=150, bbox_inches='tight')
    plt.show()

def run_experiments():
    print("Scaling Laws Analysis")
    print("=" * 50)
    
    print("\n1. Visualizing scaling laws...")
    visualize_scaling_laws()
    print("   Saved: scaling_laws.png")
    
    print("\n2. Comparing training strategies...")
    strategies = compare_training_strategies()
    print("   Saved: training_strategies.png")
    print("\n   Strategy results (same compute budget):")
    for name, result in strategies.items():
        print(f"     {name:20s}: Loss={result['loss']:.3f}, "
              f"N={result['N']/1e9:.1f}B params, D={result['D']/1e9:.0f}B tokens")
    
    print("\n3. Simulating model family...")
    results = simulate_model_family()
    print("   Saved: model_family.png")
    
    print("\n4. Analyzing extrapolation...")
    extrapolation_analysis()
    print("   Saved: scaling_extrapolation.png")
    
    print("\n" + "=" * 50)
    print("KEY INSIGHTS:")
    print("1. Power laws: Loss scales predictably with N, D, and C")
    print("2. Chinchilla: N and D should scale equally with compute")
    print("3. Overtraining small models is inefficient")
    print("4. Can extrapolate: predict 175B model from 10B data")
    print("5. GPT-3 (175B) was undertrained by Chinchilla standards")
    print("6. Scaling laws enable compute-optimal training")
    print("\nPractical: Know your compute budget → find optimal N and D")

if __name__ == "__main__":
    run_experiments()
