# normalization_activations.py
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

class NormalizationLayers:
    """Different normalization techniques"""
    
    def __init__(self):
        pass
    
    def layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5):
        """Layer Normalization - normalize across features"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma * x_norm + beta, mean, var
    
    def rms_norm(self, x: np.ndarray, gamma: np.ndarray, eps: float = 1e-5):
        """RMS Normalization - simpler, no mean subtraction"""
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        x_norm = x / rms
        return gamma * x_norm
    
    def batch_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, 
                   running_mean: np.ndarray, running_var: np.ndarray,
                   training: bool = True, momentum: float = 0.1, eps: float = 1e-5):
        """Batch Normalization - normalize across batch"""
        if training:
            mean = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True)
            
            # Update running statistics
            running_mean = (1 - momentum) * running_mean + momentum * mean
            running_var = (1 - momentum) * running_var + momentum * var
        else:
            mean = running_mean
            var = running_var
        
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma * x_norm + beta, running_mean, running_var

class ActivationFunctions:
    """Different activation functions"""
    
    def __init__(self):
        pass
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU: max(0, x)"""
        return np.maximum(0, x)
    
    def gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU: Gaussian Error Linear Unit"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    def swish(self, x: np.ndarray) -> np.ndarray:
        """Swish/SiLU: x * sigmoid(x)"""
        return x / (1 + np.exp(-x))
    
    def swiglu(self, x: np.ndarray, gate: np.ndarray) -> np.ndarray:
        """SwiGLU: Swish with gating"""
        return self.swish(gate) * x
    
    def geglu(self, x: np.ndarray, gate: np.ndarray) -> np.ndarray:
        """GeGLU: GELU with gating"""
        return self.gelu(gate) * x

def visualize_normalizations():
    """Compare different normalization methods"""
    np.random.seed(42)
    
    # Create sample data with different scales
    batch_size = 32
    d_model = 64
    x = np.random.randn(batch_size, d_model) * 5 + 10
    
    # Add some outliers
    x[0, :10] = 50
    x[1, 10:20] = -30
    
    norm = NormalizationLayers()
    
    # Layer norm parameters
    gamma_ln = np.ones(d_model)
    beta_ln = np.zeros(d_model)
    
    # Batch norm parameters
    gamma_bn = np.ones(d_model)
    beta_bn = np.zeros(d_model)
    running_mean = np.zeros((1, d_model))
    running_var = np.ones((1, d_model))
    
    # Apply normalizations
    x_layernorm, ln_mean, ln_var = norm.layer_norm(x, gamma_ln, beta_ln)
    x_rmsnorm = norm.rms_norm(x, gamma_ln)
    x_batchnorm, _, _ = norm.batch_norm(x, gamma_bn, beta_bn, running_mean, running_var, training=True)
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Original data
    ax1 = axes[0, 0]
    im1 = ax1.imshow(x, aspect='auto', cmap='coolwarm', vmin=-3, vmax=3)
    ax1.set_title('Original Data\n(Note extreme values)')
    ax1.set_xlabel('Feature Dimension')
    ax1.set_ylabel('Batch Sample')
    plt.colorbar(im1, ax=ax1)
    
    # Layer Norm
    ax2 = axes[0, 1]
    im2 = ax2.imshow(x_layernorm, aspect='auto', cmap='coolwarm', vmin=-3, vmax=3)
    ax2.set_title('Layer Normalization\n(Normalize across features per sample)')
    ax2.set_xlabel('Feature Dimension')
    ax2.set_ylabel('Batch Sample')
    plt.colorbar(im2, ax=ax2)
    
    # RMS Norm
    ax3 = axes[1, 0]
    im3 = ax3.imshow(x_rmsnorm, aspect='auto', cmap='coolwarm', vmin=-3, vmax=3)
    ax3.set_title('RMS Normalization\n(Simpler, no mean subtraction)')
    ax3.set_xlabel('Feature Dimension')
    ax3.set_ylabel('Batch Sample')
    plt.colorbar(im3, ax=ax3)
    
    # Batch Norm
    ax4 = axes[1, 1]
    im4 = ax4.imshow(x_batchnorm, aspect='auto', cmap='coolwarm', vmin=-3, vmax=3)
    ax4.set_title('Batch Normalization\n(Normalize across batch per feature)')
    ax4.set_xlabel('Feature Dimension')
    ax4.set_ylabel('Batch Sample')
    plt.colorbar(im4, ax=ax4)
    
    plt.suptitle('Normalization Methods Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('normalization_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def visualize_activations():
    """Compare different activation functions"""
    x = np.linspace(-5, 5, 1000)
    
    act = ActivationFunctions()
    
    activations = {
        'ReLU': act.relu(x),
        'GELU': act.gelu(x),
        'Swish/SiLU': act.swish(x)
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot all activations
    ax1 = axes[0, 0]
    for name, y in activations.items():
        ax1.plot(x, y, label=name, linewidth=2)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Input')
    ax1.set_ylabel('Output')
    ax1.set_title('Activation Functions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gradients
    ax2 = axes[0, 1]
    dx = x[1] - x[0]
    for name, y in activations.items():
        grad = np.gradient(y, dx)
        ax2.plot(x, grad, label=f'{name} gradient', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Input')
    ax2.set_ylabel('Gradient')
    ax2.set_title('Activation Gradients')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Histogram of activations on random data
    ax3 = axes[1, 0]
    np.random.seed(42)
    random_input = np.random.randn(10000)
    
    for name, func_name in [('ReLU', 'relu'), ('GELU', 'gelu'), ('Swish', 'swish')]:
        func = getattr(act, func_name)
        output = func(random_input)
        ax3.hist(output, bins=50, alpha=0.5, label=name, density=True)
    
    ax3.set_xlabel('Activation Value')
    ax3.set_ylabel('Density')
    ax3.set_title('Distribution of Activations\n(Input: N(0,1))')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Dead neurons (for ReLU)
    ax4 = axes[1, 1]
    input_ranges = np.linspace(-5, 5, 50)
    dead_ratios = []
    
    for mean in input_ranges:
        test_input = np.random.randn(1000) + mean
        relu_output = act.relu(test_input)
        dead_ratio = np.sum(relu_output == 0) / len(relu_output) * 100
        dead_ratios.append(dead_ratio)
    
    ax4.plot(input_ranges, dead_ratios, 'r-', linewidth=2)
    ax4.fill_between(input_ranges, dead_ratios, alpha=0.3, color='red')
    ax4.set_xlabel('Input Mean')
    ax4.set_ylabel('Dead Neurons (%)')
    ax4.set_title('ReLU Dead Neuron Problem\n(Neurons outputting zero)')
    ax4.grid(True, alpha=0.3)
    ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    plt.suptitle('Activation Functions Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('activation_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def gated_activations_demo():
    """Demonstrate gated activations (SwiGLU, GeGLU)"""
    np.random.seed(42)
    d_model = 512
    d_ff = 2048
    batch_size = 32
    
    # Input
    x = np.random.randn(batch_size, d_model)
    
    # Standard FFN with GELU
    W1_gelu = np.random.randn(d_model, d_ff) * 0.02
    W2_gelu = np.random.randn(d_ff, d_model) * 0.02
    
    act = ActivationFunctions()
    hidden_gelu = act.gelu(np.dot(x, W1_gelu))
    output_gelu = np.dot(hidden_gelu, W2_gelu)
    
    # GeGLU FFN (gated)
    W_gate = np.random.randn(d_model, d_ff) * 0.02
    W_value = np.random.randn(d_model, d_ff) * 0.02
    W2_geglu = np.random.randn(d_ff, d_model) * 0.02
    
    gate = np.dot(x, W_gate)
    value = np.dot(x, W_value)
    hidden_geglu = act.geglu(value, gate)
    output_geglu = np.dot(hidden_geglu, W2_geglu)
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Standard GELU path
    axes[0, 0].hist(hidden_gelu.flatten(), bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_title('GELU Hidden\nActivations')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Count')
    
    axes[0, 1].hist(output_gelu.flatten(), bins=50, alpha=0.7, color='blue')
    axes[0, 1].set_title('GELU Output')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Count')
    
    # Stats
    axes[0, 2].axis('off')
    stats_text = f"GELU FFN Stats:\n\n"
    stats_text += f"Hidden mean: {np.mean(hidden_gelu):.3f}\n"
    stats_text += f"Hidden std: {np.std(hidden_gelu):.3f}\n"
    stats_text += f"Output mean: {np.mean(output_gelu):.3f}\n"
    stats_text += f"Output std: {np.std(output_gelu):.3f}\n"
    stats_text += f"Parameters: {(W1_gelu.size + W2_gelu.size):,}"
    axes[0, 2].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # GeGLU path
    axes[1, 0].hist(hidden_geglu.flatten(), bins=50, alpha=0.7, color='green')
    axes[1, 0].set_title('GeGLU Hidden\nActivations (gated)')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Count')
    
    axes[1, 1].hist(output_geglu.flatten(), bins=50, alpha=0.7, color='green')
    axes[1, 1].set_title('GeGLU Output')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Count')
    
    # Stats
    axes[1, 2].axis('off')
    stats_text = f"GeGLU FFN Stats:\n\n"
    stats_text += f"Hidden mean: {np.mean(hidden_geglu):.3f}\n"
    stats_text += f"Hidden std: {np.std(hidden_geglu):.3f}\n"
    stats_text += f"Output mean: {np.mean(output_geglu):.3f}\n"
    stats_text += f"Output std: {np.std(output_geglu):.3f}\n"
    stats_text += f"Parameters: {(W_gate.size + W_value.size + W2_geglu.size):,}\n"
    stats_text += f"(~2x params for gating)"
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.suptitle('Gated vs Non-Gated Activations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('gated_activations.png', dpi=150, bbox_inches='tight')
    plt.show()

def training_stability_simulation():
    """Simulate training with different norm/activation combos"""
    np.random.seed(42)
    
    # Simulate layer outputs during training
    n_steps = 100
    d_model = 128
    
    configs = [
        ('No Norm + ReLU', False, 'relu', 'red'),
        ('LayerNorm + ReLU', True, 'relu', 'orange'),
        ('LayerNorm + GELU', True, 'gelu', 'green'),
        ('RMSNorm + GELU', 'rms', 'gelu', 'blue')
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    norm = NormalizationLayers()
    act = ActivationFunctions()
    
    for name, norm_type, act_type, color in configs:
        activations_mean = []
        gradients_magnitude = []
        
        x = np.random.randn(32, d_model) * 0.1
        
        for step in range(n_steps):
            # Forward pass simulation
            if norm_type == True:
                gamma = np.ones(d_model)
                beta = np.zeros(d_model)
                x, _, _ = norm.layer_norm(x, gamma, beta)
            elif norm_type == 'rms':
                gamma = np.ones(d_model)
                x = norm.rms_norm(x, gamma)
            
            # Activation
            if act_type == 'relu':
                x = act.relu(x)
            elif act_type == 'gelu':
                x = act.gelu(x)
            
            # Add some noise to simulate training dynamics
            x = x + np.random.randn(32, d_model) * 0.05
            
            # Track statistics
            activations_mean.append(np.abs(np.mean(x)))
            gradients_magnitude.append(np.mean(np.abs(x)))  # Simplified
        
        # Plot activation magnitudes
        ax1.plot(activations_mean, label=name, color=color, linewidth=2)
        
        # Plot gradient magnitudes
        ax2.plot(gradients_magnitude, label=name, color=color, linewidth=2)
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Mean Activation Magnitude')
    ax1.set_title('Activation Stability During Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Gradient Magnitude (simulated)')
    ax2.set_title('Gradient Flow Stability')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.suptitle('Training Stability: Impact of Normalization & Activation', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('training_stability.png', dpi=150, bbox_inches='tight')
    plt.show()

def run_experiments():
    print("Normalization & Activation Functions Analysis")
    print("=" * 50)
    
    print("\n1. Comparing normalization methods...")
    visualize_normalizations()
    print("   Saved: normalization_comparison.png")
    
    print("\n2. Analyzing activation functions...")
    visualize_activations()
    print("   Saved: activation_comparison.png")
    
    print("\n3. Demonstrating gated activations...")
    gated_activations_demo()
    print("   Saved: gated_activations.png")
    
    print("\n4. Simulating training stability...")
    training_stability_simulation()
    print("   Saved: training_stability.png")
    
    print("\n" + "=" * 50)
    print("KEY INSIGHTS:")
    print("1. LayerNorm: Normalizes across features (standard in transformers)")
    print("2. RMSNorm: Simpler than LayerNorm, ~10-20% faster, similar quality")
    print("3. BatchNorm: Normalizes across batch (problematic for small batches)")
    print("4. GELU: Smoother than ReLU, better gradients, no dead neurons")
    print("5. SwiGLU/GeGLU: Gated activations, 2x params but better capacity")
    print("6. Modern transformers: RMSNorm + SwiGLU (Llama 2)")
    print("\nTrade-offs: Simplicity vs performance vs compute cost")

if __name__ == "__main__":
    run_experiments()
