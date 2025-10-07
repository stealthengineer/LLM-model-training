# quantization.py
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

class Quantization:
    """Different quantization techniques for model compression"""
    
    def __init__(self):
        pass
    
    def quantize_symmetric(self, weights: np.ndarray, n_bits: int = 8) -> Tuple[np.ndarray, float]:
        """
        Symmetric quantization: map [-max, max] to [-2^(n-1), 2^(n-1)-1]
        """
        max_val = np.max(np.abs(weights))
        n_levels = 2 ** n_bits
        
        # Scale factor
        scale = max_val / (n_levels / 2 - 1)
        
        # Quantize
        weights_quantized = np.round(weights / scale)
        weights_quantized = np.clip(weights_quantized, -(n_levels // 2), (n_levels // 2) - 1)
        
        return weights_quantized.astype(np.int8 if n_bits == 8 else np.int16), scale
    
    def dequantize(self, weights_quantized: np.ndarray, scale: float) -> np.ndarray:
        """Dequantize back to float"""
        return weights_quantized.astype(np.float32) * scale
    
    def quantize_asymmetric(self, weights: np.ndarray, n_bits: int = 8) -> Tuple[np.ndarray, float, float]:
        """
        Asymmetric quantization: map [min, max] to [0, 2^n-1]
        Better for activations with positive range
        """
        min_val = np.min(weights)
        max_val = np.max(weights)
        n_levels = 2 ** n_bits
        
        # Scale and zero point
        scale = (max_val - min_val) / (n_levels - 1)
        zero_point = -min_val / scale
        
        # Quantize
        weights_quantized = np.round(weights / scale + zero_point)
        weights_quantized = np.clip(weights_quantized, 0, n_levels - 1)
        
        return weights_quantized.astype(np.uint8 if n_bits == 8 else np.uint16), scale, zero_point
    
    def per_channel_quantization(self, weights: np.ndarray, n_bits: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantize each output channel separately for better accuracy
        weights shape: (out_channels, in_channels)
        """
        out_channels = weights.shape[0]
        weights_quantized = np.zeros_like(weights, dtype=np.int8 if n_bits == 8 else np.int16)
        scales = np.zeros(out_channels)
        
        for i in range(out_channels):
            weights_quantized[i], scales[i] = self.quantize_symmetric(weights[i], n_bits)
        
        return weights_quantized, scales

def visualize_quantization_error():
    """Show quantization error for different bit widths"""
    np.random.seed(42)
    
    # Generate realistic weight distribution (roughly normal)
    weights = np.random.randn(10000) * 0.5
    
    quant = Quantization()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    bit_widths = [8, 4, 3, 2]
    
    for idx, (ax, n_bits) in enumerate(zip(axes.flat, bit_widths)):
        # Quantize and dequantize
        w_quant, scale = quant.quantize_symmetric(weights, n_bits)
        w_dequant = quant.dequantize(w_quant, scale)
        
        # Calculate error
        error = w_dequant - weights
        mse = np.mean(error ** 2)
        
        # Plot distributions
        ax.hist(weights, bins=50, alpha=0.5, label='Original', density=True, color='blue')
        ax.hist(w_dequant, bins=50, alpha=0.5, label='Quantized', density=True, color='red')
        
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Density')
        ax.set_title(f'{n_bits}-bit Quantization\nMSE: {mse:.6f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Quantization Error vs Bit Width', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('quantization_error.png', dpi=150, bbox_inches='tight')
    plt.show()

def compare_quantization_methods():
    """Compare per-tensor vs per-channel quantization"""
    np.random.seed(42)
    
    # Create weight matrix with different channel magnitudes
    out_channels = 8
    in_channels = 512
    weights = np.random.randn(out_channels, in_channels) * 0.1
    
    # Make some channels have larger weights
    weights[0] *= 5
    weights[7] *= 0.2
    
    quant = Quantization()
    
    # Per-tensor quantization
    w_flat = weights.flatten()
    w_quant_tensor, scale_tensor = quant.quantize_symmetric(w_flat, n_bits=8)
    w_dequant_tensor = quant.dequantize(w_quant_tensor, scale_tensor).reshape(weights.shape)
    
    # Per-channel quantization
    w_quant_channel, scales_channel = quant.per_channel_quantization(weights, n_bits=8)
    w_dequant_channel = np.zeros_like(weights)
    for i in range(out_channels):
        w_dequant_channel[i] = quant.dequantize(w_quant_channel[i], scales_channel[i])
    
    # Calculate errors
    error_tensor = np.abs(w_dequant_tensor - weights)
    error_channel = np.abs(w_dequant_channel - weights)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Original weights
    im1 = axes[0, 0].imshow(weights, aspect='auto', cmap='coolwarm', vmin=-0.5, vmax=0.5)
    axes[0, 0].set_title('Original Weights')
    axes[0, 0].set_xlabel('Input Channel')
    axes[0, 0].set_ylabel('Output Channel')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Per-tensor error
    im2 = axes[0, 1].imshow(error_tensor, aspect='auto', cmap='Reds', vmin=0, vmax=0.1)
    axes[0, 1].set_title(f'Per-Tensor Error\nMean: {np.mean(error_tensor):.6f}')
    axes[0, 1].set_xlabel('Input Channel')
    axes[0, 1].set_ylabel('Output Channel')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Per-channel error
    im3 = axes[1, 0].imshow(error_channel, aspect='auto', cmap='Reds', vmin=0, vmax=0.1)
    axes[1, 0].set_title(f'Per-Channel Error\nMean: {np.mean(error_channel):.6f}')
    axes[1, 0].set_xlabel('Input Channel')
    axes[1, 0].set_ylabel('Output Channel')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Error comparison by channel
    channel_error_tensor = np.mean(error_tensor, axis=1)
    channel_error_channel = np.mean(error_channel, axis=1)
    
    x = np.arange(out_channels)
    width = 0.35
    
    bars1 = axes[1, 1].bar(x - width/2, channel_error_tensor, width, 
                          label='Per-Tensor', color='red', alpha=0.7)
    bars2 = axes[1, 1].bar(x + width/2, channel_error_channel, width,
                          label='Per-Channel', color='green', alpha=0.7)
    
    axes[1, 1].set_xlabel('Output Channel')
    axes[1, 1].set_ylabel('Mean Absolute Error')
    axes[1, 1].set_title('Error per Channel')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Per-Tensor vs Per-Channel Quantization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('quantization_methods.png', dpi=150, bbox_inches='tight')
    plt.show()

def model_size_analysis():
    """Analyze model size reduction from quantization"""
    # Model sizes (representative)
    models = {
        'GPT-2 (1.5B)': 1.5e9,
        'LLaMA 7B': 7e9,
        'LLaMA 13B': 13e9,
        'LLaMA 33B': 33e9,
        'LLaMA 70B': 70e9
    }
    
    # Bytes per parameter for different precisions
    precisions = {
        'FP32': 4,
        'FP16': 2,
        'INT8': 1,
        'INT4': 0.5
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Size for LLaMA 70B across precisions
    model_name = 'LLaMA 70B'
    model_params = models[model_name]
    
    sizes = []
    labels = []
    for prec, bytes_per_param in precisions.items():
        size_gb = (model_params * bytes_per_param) / (1024 ** 3)
        sizes.append(size_gb)
        labels.append(f'{prec}\n{size_gb:.1f} GB')
    
    bars = ax1.bar(range(len(precisions)), sizes, 
                   color=['red', 'orange', 'green', 'blue'], alpha=0.7)
    ax1.set_xticks(range(len(precisions)))
    ax1.set_xticklabels(list(precisions.keys()))
    ax1.set_ylabel('Model Size (GB)')
    ax1.set_title(f'{model_name} Size Across Precisions')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add size labels
    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{size:.1f} GB',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Compression ratios
    ax2_data = []
    for name, params in models.items():
        fp32_size = (params * 4) / (1024 ** 3)
        int8_size = (params * 1) / (1024 ** 3)
        int4_size = (params * 0.5) / (1024 ** 3)
        
        ax2_data.append({
            'name': name,
            'fp32': fp32_size,
            'int8': int8_size,
            'int4': int4_size
        })
    
    x = np.arange(len(models))
    width = 0.25
    
    fp32_sizes = [d['fp32'] for d in ax2_data]
    int8_sizes = [d['int8'] for d in ax2_data]
    int4_sizes = [d['int4'] for d in ax2_data]
    
    bars1 = ax2.bar(x - width, fp32_sizes, width, label='FP32', color='red', alpha=0.7)
    bars2 = ax2.bar(x, int8_sizes, width, label='INT8', color='green', alpha=0.7)
    bars3 = ax2.bar(x + width, int4_sizes, width, label='INT4', color='blue', alpha=0.7)
    
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Size (GB)')
    ax2.set_title('Model Sizes Across Quantization')
    ax2.set_xticks(x)
    ax2.set_xticklabels([d['name'] for d in ax2_data], rotation=15, ha='right')
    ax2.legend()
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('model_sizes.png', dpi=150, bbox_inches='tight')
    plt.show()

def accuracy_vs_bits():
    """Simulate accuracy degradation with quantization"""
    bit_widths = [16, 8, 6, 4, 3, 2]
    
    # Simulated accuracy (based on real observations)
    # FP16 baseline = 100%
    relative_accuracy = {
        16: 100.0,
        8: 99.5,
        6: 98.5,
        4: 95.0,
        3: 85.0,
        2: 60.0
    }
    
    # Different model sizes behave differently
    model_sizes = ['Small (1B)', 'Medium (7B)', 'Large (70B)']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy degradation
    accuracies = [relative_accuracy[b] for b in bit_widths]
    
    ax1.plot(bit_widths, accuracies, 'o-', linewidth=3, markersize=10, color='steelblue')
    ax1.axhline(y=95, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Good (>95%)')
    ax1.axhline(y=90, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Acceptable (>90%)')
    ax1.axhline(y=80, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Poor (<80%)')
    
    ax1.set_xlabel('Bit Width')
    ax1.set_ylabel('Relative Accuracy (%)')
    ax1.set_title('Accuracy vs Quantization Bit Width')
    ax1.set_xticks(bit_widths)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(50, 105)
    
    # Add annotations
    for bits, acc in zip(bit_widths, accuracies):
        ax1.annotate(f'{acc:.1f}%', (bits, acc), 
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=9)
    
    # Compression vs accuracy trade-off
    compression = [32/b for b in bit_widths]
    
    ax2.scatter(compression, accuracies, s=200, alpha=0.6, c=bit_widths, cmap='viridis')
    
    for bits, comp, acc in zip(bit_widths, compression, accuracies):
        ax2.annotate(f'{bits}-bit', (comp, acc), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax2.set_xlabel('Compression Ratio (vs FP32)')
    ax2.set_ylabel('Relative Accuracy (%)')
    ax2.set_title('Compression vs Accuracy Trade-off')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=95, color='green', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('accuracy_vs_bits.png', dpi=150, bbox_inches='tight')
    plt.show()

def run_experiments():
    print("Quantization Analysis")
    print("=" * 50)
    
    print("\n1. Visualizing quantization error...")
    visualize_quantization_error()
    print("   Saved: quantization_error.png")
    
    print("\n2. Comparing quantization methods...")
    compare_quantization_methods()
    print("   Saved: quantization_methods.png")
    
    print("\n3. Analyzing model sizes...")
    model_size_analysis()
    print("   Saved: model_sizes.png")
    
    print("\n4. Accuracy vs bit width...")
    accuracy_vs_bits()
    print("   Saved: accuracy_vs_bits.png")
    
    print("\n" + "=" * 50)
    print("KEY INSIGHTS:")
    print("1. INT8: 4x smaller, <1% accuracy loss (production standard)")
    print("2. INT4: 8x smaller, ~5% accuracy loss (acceptable for many uses)")
    print("3. Per-channel quantization >> per-tensor (lower error)")
    print("4. LLaMA 70B: 280GB (FP32) → 70GB (INT8) → 35GB (INT4)")
    print("5. Larger models handle quantization better")
    print("6. INT4 makes 70B models run on consumer GPUs")
    print("\nProduction: GPTQ/AWQ for INT4, bitsandbytes for INT8")

if __name__ == "__main__":
    run_experiments()
