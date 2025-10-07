# finetuning_strategies.py
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

class FineTuningStrategies:
    """Compare different fine-tuning approaches"""
    
    def __init__(self, n_layers: int = 12, d_model: int = 768):
        self.n_layers = n_layers
        self.d_model = d_model
        self.total_params = self._calculate_total_params()
    
    def _calculate_total_params(self):
        """Estimate total parameters"""
        # Simplified: embeddings + layers + output
        vocab_size = 50000
        params_per_layer = self.d_model * self.d_model * 4  # Q,K,V,O in attention
        params_per_layer += self.d_model * (self.d_model * 4) * 2  # FFN
        
        total = vocab_size * self.d_model  # Embeddings
        total += params_per_layer * self.n_layers  # Layers
        total += vocab_size * self.d_model  # Output head
        
        return total
    
    def full_finetuning(self):
        """All parameters trainable"""
        return {
            'trainable_params': self.total_params,
            'frozen_params': 0,
            'memory_multiplier': 1.0,
            'training_speed': 1.0
        }
    
    def last_layer_tuning(self):
        """Only tune last layer"""
        trainable = self.d_model * 50000  # Just output projection
        return {
            'trainable_params': trainable,
            'frozen_params': self.total_params - trainable,
            'memory_multiplier': 0.1,
            'training_speed': 10.0
        }
    
    def lora_tuning(self, rank: int = 8):
        """LoRA: Low-Rank Adaptation"""
        # Add low-rank matrices to attention weights
        # Instead of updating W (d×d), add A·B where A is d×r and B is r×d
        
        lora_params_per_layer = (self.d_model * rank * 2) * 4  # Q,K,V,O
        trainable = lora_params_per_layer * self.n_layers
        
        return {
            'trainable_params': trainable,
            'frozen_params': self.total_params - trainable,
            'memory_multiplier': 0.3,
            'training_speed': 3.0,
            'rank': rank
        }
    
    def adapter_tuning(self):
        """Add small adapter layers between frozen layers"""
        adapter_size = self.d_model // 4
        params_per_adapter = self.d_model * adapter_size * 2  # Down + up projection
        trainable = params_per_adapter * self.n_layers * 2  # 2 adapters per layer
        
        return {
            'trainable_params': trainable,
            'frozen_params': self.total_params - trainable,
            'memory_multiplier': 0.4,
            'training_speed': 2.5
        }
    
    def prompt_tuning(self, n_prompt_tokens: int = 20):
        """Learn soft prompts (virtual tokens)"""
        trainable = n_prompt_tokens * self.d_model
        
        return {
            'trainable_params': trainable,
            'frozen_params': self.total_params - trainable,
            'memory_multiplier': 0.05,
            'training_speed': 20.0
        }

def visualize_parameter_efficiency():
    """Compare trainable parameters across methods"""
    strategy = FineTuningStrategies(n_layers=12, d_model=768)
    
    methods = {
        'Full Fine-tuning': strategy.full_finetuning(),
        'LoRA (r=8)': strategy.lora_tuning(rank=8),
        'LoRA (r=32)': strategy.lora_tuning(rank=32),
        'Adapters': strategy.adapter_tuning(),
        'Last Layer Only': strategy.last_layer_tuning(),
        'Prompt Tuning': strategy.prompt_tuning()
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Trainable parameters
    ax1 = axes[0, 0]
    names = list(methods.keys())
    trainable = [methods[m]['trainable_params'] / 1e6 for m in names]
    
    bars = ax1.barh(names, trainable, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Trainable Parameters (Millions)')
    ax1.set_title('Parameter Efficiency')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add percentage labels
    total_params = strategy.total_params / 1e6
    for bar, val in zip(bars, trainable):
        percentage = (val / total_params) * 100
        ax1.text(val, bar.get_y() + bar.get_height()/2,
                f' {val:.1f}M ({percentage:.1f}%)',
                va='center', fontsize=9)
    
    # 2. Memory efficiency
    ax2 = axes[0, 1]
    memory = [methods[m]['memory_multiplier'] for m in names]
    
    bars = ax2.barh(names, memory, color='green', alpha=0.7)
    ax2.set_xlabel('Relative Memory Usage')
    ax2.set_title('Memory Efficiency\n(Lower is better)')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    for bar, val in zip(bars, memory):
        ax2.text(val, bar.get_y() + bar.get_height()/2,
                f' {val:.1f}x',
                va='center', fontsize=9)
    
    # 3. Training speed
    ax3 = axes[1, 0]
    speed = [methods[m]['training_speed'] for m in names]
    
    bars = ax3.barh(names, speed, color='orange', alpha=0.7)
    ax3.set_xlabel('Relative Training Speed')
    ax3.set_title('Training Speed\n(Higher is better)')
    ax3.grid(True, alpha=0.3, axis='x')
    
    for bar, val in zip(bars, speed):
        ax3.text(val, bar.get_y() + bar.get_height()/2,
                f' {val:.1f}x',
                va='center', fontsize=9)
    
    # 4. Efficiency score
    ax4 = axes[1, 1]
    efficiency_scores = []
    for name in names:
        m = methods[name]
        params_ratio = m['trainable_params'] / strategy.total_params
        score = (m['training_speed'] / m['memory_multiplier']) * (1 / (params_ratio + 0.01))
        efficiency_scores.append(score)
    
    bars = ax4.barh(names, efficiency_scores, color='purple', alpha=0.7)
    ax4.set_xlabel('Efficiency Score (higher = better)')
    ax4.set_title('Overall Efficiency\n(Speed / Memory / Params)')
    ax4.grid(True, alpha=0.3, axis='x')
    
    for bar, val in zip(bars, efficiency_scores):
        ax4.text(val, bar.get_y() + bar.get_height()/2,
                f' {val:.0f}',
                va='center', fontsize=9)
    
    plt.suptitle('Fine-Tuning Strategy Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('finetuning_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def visualize_lora_concept():
    """Illustrate how LoRA works"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    d = 8  # Model dimension (simplified)
    r = 2  # LoRA rank
    
    # Full fine-tuning
    ax1 = axes[0]
    delta_W = np.random.randn(d, d) * 0.05
    
    ax1.imshow(delta_W, cmap='coolwarm', aspect='auto', vmin=-0.2, vmax=0.2)
    ax1.set_title(f'Full Fine-tuning\nUpdate entire W ({d}×{d} = {d*d} params)')
    ax1.set_xlabel('Output Dim')
    ax1.set_ylabel('Input Dim')
    
    # LoRA decomposition - FIXED
    ax2 = axes[1]
    A = np.random.randn(d, r) * 0.1
    B = np.random.randn(r, d) * 0.1
    
    # Create visualization with proper dimensions
    # Show A (d×r) and B^T (d×r) side by side
    combined = np.hstack([A, B.T])  # Now (d, 2r)
    
    im = ax2.imshow(combined, cmap='coolwarm', aspect='auto', vmin=-0.2, vmax=0.2)
    ax2.axvline(x=r-0.5, color='black', linewidth=3)
    ax2.set_title(f'LoRA Matrices\nA ({d}×{r}) and B ({r}×{d}) = {d*r*2} params')
    ax2.set_xlabel('Dimensions')
    ax2.set_ylabel('Input Dim')
    ax2.text(r/2, -0.8, 'A', ha='center', fontsize=12, fontweight='bold')
    ax2.text(r + r/2, -0.8, 'B^T', ha='center', fontsize=12, fontweight='bold')
    
    # Reconstructed update
    ax3 = axes[2]
    delta_W_lora = np.dot(A, B)
    
    ax3.imshow(delta_W_lora, cmap='coolwarm', aspect='auto', vmin=-0.2, vmax=0.2)
    ax3.set_title(f'LoRA Update\nA·B approximates ΔW\n({100*(d*r*2)/(d*d):.1f}% params)')
    ax3.set_xlabel('Output Dim')
    ax3.set_ylabel('Input Dim')
    
    plt.suptitle('LoRA: Low-Rank Adaptation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('lora_concept.png', dpi=150, bbox_inches='tight')
    plt.show()

def simulate_finetuning_scenarios():
    """Compare methods across different scenarios"""
    scenarios = {
        'Large Dataset\n(1M samples)': {
            'full': 95,
            'lora': 93,
            'adapter': 91,
            'prompt': 75,
            'last_layer': 70
        },
        'Medium Dataset\n(100K samples)': {
            'full': 85,
            'lora': 88,
            'adapter': 86,
            'prompt': 80,
            'last_layer': 75
        },
        'Small Dataset\n(10K samples)': {
            'full': 65,
            'lora': 82,
            'adapter': 80,
            'prompt': 78,
            'last_layer': 72
        },
        'Tiny Dataset\n(1K samples)': {
            'full': 45,
            'lora': 70,
            'adapter': 68,
            'prompt': 75,
            'last_layer': 65
        }
    }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(scenarios))
    width = 0.15
    
    methods = ['full', 'lora', 'adapter', 'prompt', 'last_layer']
    labels = ['Full FT', 'LoRA', 'Adapters', 'Prompt Tuning', 'Last Layer']
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    
    for i, (method, label, color) in enumerate(zip(methods, labels, colors)):
        values = [scenarios[s][method] for s in scenarios.keys()]
        offset = (i - 2) * width
        bars = ax.bar(x + offset, values, width, label=label, color=color, alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Performance Score')
    ax.set_title('Fine-Tuning Strategy Performance vs Dataset Size\n(Simulated Results)')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios.keys())
    ax.legend(loc='upper left')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add annotation
    ax.text(0.5, 0.02, 'Key Insight: PEFT methods (LoRA, Adapters) excel with small datasets',
           transform=ax.transAxes, ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('finetuning_scenarios.png', dpi=150, bbox_inches='tight')
    plt.show()

def catastrophic_forgetting_demo():
    """Show how full fine-tuning can forget pretrained knowledge"""
    tasks = ['General\nKnowledge', 'Reasoning', 'Original\nTask', 'New\nTask', 'Code\nGeneration']
    
    pretrained = [85, 80, 50, 0, 75]
    full_ft = [60, 55, 70, 95, 50]
    lora_ft = [82, 78, 75, 90, 72]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(tasks))
    width = 0.25
    
    # Absolute performance
    bars1 = ax1.bar(x - width, pretrained, width, label='Pretrained', color='gray', alpha=0.7)
    bars2 = ax1.bar(x, full_ft, width, label='After Full FT', color='red', alpha=0.7)
    bars3 = ax1.bar(x + width, lora_ft, width, label='After LoRA', color='green', alpha=0.7)
    
    ax1.set_ylabel('Performance Score')
    ax1.set_title('Catastrophic Forgetting\n(Absolute Performance)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks)
    ax1.legend()
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Change from pretrained
    full_change = [f - p for f, p in zip(full_ft, pretrained)]
    lora_change = [l - p for l, p in zip(lora_ft, pretrained)]
    
    bars1 = ax2.bar(x - width/2, full_change, width, label='Full FT Change', 
                    color=['red' if c < 0 else 'green' for c in full_change], alpha=0.7)
    bars2 = ax2.bar(x + width/2, lora_change, width, label='LoRA Change',
                    color=['red' if c < 0 else 'green' for c in lora_change], alpha=0.7)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_ylabel('Performance Change')
    ax2.set_title('Performance Change from Pretrained\n(Negative = Forgetting)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(tasks)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):+d}',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):+d}',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('catastrophic_forgetting.png', dpi=150, bbox_inches='tight')
    plt.show()

def run_experiments():
    print("Fine-Tuning Strategies Analysis")
    print("=" * 50)
    
    print("\n1. Comparing parameter efficiency...")
    visualize_parameter_efficiency()
    print("   Saved: finetuning_comparison.png")
    
    print("\n2. Explaining LoRA concept...")
    visualize_lora_concept()
    print("   Saved: lora_concept.png")
    
    print("\n3. Simulating different scenarios...")
    simulate_finetuning_scenarios()
    print("   Saved: finetuning_scenarios.png")
    
    print("\n4. Demonstrating catastrophic forgetting...")
    catastrophic_forgetting_demo()
    print("   Saved: catastrophic_forgetting.png")
    
    print("\n" + "=" * 50)
    print("KEY INSIGHTS:")
    print("1. Full fine-tuning: Most flexible but risks catastrophic forgetting")
    print("2. LoRA: Best balance - 0.1% params, 90%+ performance")
    print("3. Prompt tuning: Extremely efficient for simple adaptations")
    print("4. PEFT methods excel with small datasets (<10K samples)")
    print("5. Full FT only needed for large datasets or major task shifts")
    print("6. Production: LoRA is the standard (used everywhere)")
    print("\nModern approach: LoRA with r=8-32 for most tasks")

if __name__ == "__main__":
    run_experiments()
