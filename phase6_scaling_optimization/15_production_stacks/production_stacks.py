# production_stacks.py
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

class ProductionStack:
    """Analyze different production serving stacks"""
    
    def __init__(self, name: str):
        self.name = name
        
    def get_specs(self) -> Dict:
        """Get specifications for this stack"""
        pass

class HuggingFaceStack(ProductionStack):
    """Standard HuggingFace Transformers"""
    
    def get_specs(self):
        return {
            'throughput': 1.0,  # Baseline
            'latency': 1.0,
            'memory_efficiency': 1.0,
            'ease_of_use': 10,
            'features': ['Easy setup', 'All models', 'Research friendly'],
            'limitations': ['Not optimized', 'No batching', 'Slow']
        }

class vLLMStack(ProductionStack):
    """vLLM - optimized inference"""
    
    def get_specs(self):
        return {
            'throughput': 24.0,  # ~24x faster with batching
            'latency': 0.5,
            'memory_efficiency': 2.0,  # PagedAttention
            'ease_of_use': 7,
            'features': ['Continuous batching', 'PagedAttention', 'High throughput'],
            'limitations': ['Less flexible', 'CUDA only', 'Setup complexity']
        }

class TensorRTStack(ProductionStack):
    """TensorRT-LLM - NVIDIA optimized"""
    
    def get_specs(self):
        return {
            'throughput': 30.0,
            'latency': 0.3,
            'memory_efficiency': 2.5,
            'ease_of_use': 4,
            'features': ['Fastest', 'Kernel fusion', 'FP8 support'],
            'limitations': ['NVIDIA only', 'Complex setup', 'Limited models']
        }

class LlamaCppStack(ProductionStack):
    """llama.cpp - CPU/edge optimized"""
    
    def get_specs(self):
        return {
            'throughput': 0.3,  # Slower but runs anywhere
            'latency': 3.0,
            'memory_efficiency': 1.5,
            'ease_of_use': 9,
            'features': ['CPU support', 'Quantization', 'Edge deployment'],
            'limitations': ['Slower', 'Limited to LLaMA-style', 'No batching']
        }

def compare_stacks():
    """Compare different production stacks"""
    stacks = {
        'HuggingFace': HuggingFaceStack('HuggingFace'),
        'vLLM': vLLMStack('vLLM'),
        'TensorRT-LLM': TensorRTStack('TensorRT-LLM'),
        'llama.cpp': LlamaCppStack('llama.cpp')
    }
    
    specs = {name: stack.get_specs() for name, stack in stacks.items()}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    names = list(specs.keys())
    
    # 1. Throughput
    ax1 = axes[0, 0]
    throughput = [specs[n]['throughput'] for n in names]
    bars = ax1.bar(names, throughput, color=['blue', 'green', 'red', 'orange'], alpha=0.7)
    ax1.set_ylabel('Relative Throughput')
    ax1.set_title('Throughput (Higher = Better)')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, throughput):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}x',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Latency
    ax2 = axes[0, 1]
    latency = [specs[n]['latency'] for n in names]
    bars = ax2.bar(names, latency, color=['blue', 'green', 'red', 'orange'], alpha=0.7)
    ax2.set_ylabel('Relative Latency')
    ax2.set_title('Latency (Lower = Better)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, latency):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}x',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Memory efficiency
    ax3 = axes[1, 0]
    memory = [specs[n]['memory_efficiency'] for n in names]
    bars = ax3.bar(names, memory, color=['blue', 'green', 'red', 'orange'], alpha=0.7)
    ax3.set_ylabel('Memory Efficiency')
    ax3.set_title('Memory Efficiency (Higher = Better)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, memory):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}x',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Ease of use
    ax4 = axes[1, 1]
    ease = [specs[n]['ease_of_use'] for n in names]
    bars = ax4.bar(names, ease, color=['blue', 'green', 'red', 'orange'], alpha=0.7)
    ax4.set_ylabel('Ease of Use Score')
    ax4.set_title('Ease of Use (Higher = Better)')
    ax4.set_ylim(0, 11)
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, ease):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}/10',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('Production Stack Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('stack_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return specs

def continuous_batching_visualization():
    """Show how continuous batching works"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Static batching (traditional)
    requests_static = [
        {'id': 1, 'tokens': 100, 'start': 0},
        {'id': 2, 'tokens': 50, 'start': 0},
        {'id': 3, 'tokens': 150, 'start': 0},
        {'id': 4, 'tokens': 200, 'start': 200},  # Has to wait!
    ]
    
    colors = ['blue', 'green', 'red', 'orange']
    
    # Static batching timeline
    for i, req in enumerate(requests_static[:3]):
        ax1.barh(i, req['tokens'], left=req['start'], 
                color=colors[i], alpha=0.7, height=0.8)
        ax1.text(req['start'] + req['tokens']/2, i, f"Req {req['id']}", 
                ha='center', va='center', fontweight='bold')
    
    # Request 4 has to wait for batch to finish
    ax1.barh(3, 50, left=0, color='lightgray', alpha=0.3, height=0.8)
    ax1.text(25, 3, 'Waiting...', ha='center', va='center', style='italic')
    ax1.barh(3, requests_static[3]['tokens'], left=requests_static[3]['start'],
            color=colors[3], alpha=0.7, height=0.8)
    ax1.text(requests_static[3]['start'] + requests_static[3]['tokens']/2, 3, 
            f"Req {requests_static[3]['id']}", ha='center', va='center', fontweight='bold')
    
    ax1.set_ylim(-0.5, 3.5)
    ax1.set_xlim(0, 450)
    ax1.set_xlabel('Time (tokens generated)')
    ax1.set_ylabel('Request')
    ax1.set_title('Static Batching\n(New requests wait for batch to complete)')
    ax1.set_yticks(range(4))
    ax1.set_yticklabels([f'Request {i+1}' for i in range(4)])
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Continuous batching (vLLM)
    requests_continuous = [
        {'id': 1, 'tokens': 100, 'start': 0},
        {'id': 2, 'tokens': 50, 'start': 0},
        {'id': 3, 'tokens': 150, 'start': 0},
        {'id': 4, 'tokens': 200, 'start': 50},  # Joins mid-batch!
    ]
    
    for i, req in enumerate(requests_continuous):
        ax2.barh(i, req['tokens'], left=req['start'],
                color=colors[i], alpha=0.7, height=0.8)
        ax2.text(req['start'] + req['tokens']/2, i, f"Req {req['id']}",
                ha='center', va='center', fontweight='bold')
    
    # Show joining points
    ax2.axvline(x=50, color='green', linestyle='--', linewidth=2, alpha=0.5)
    ax2.text(50, 4, 'New request joins', ha='center', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    ax2.set_ylim(-0.5, 4.5)
    ax2.set_xlim(0, 300)
    ax2.set_xlabel('Time (tokens generated)')
    ax2.set_ylabel('Request')
    ax2.set_title('Continuous Batching (vLLM)\n(New requests join immediately)')
    ax2.set_yticks(range(4))
    ax2.set_yticklabels([f'Request {i+1}' for i in range(4)])
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('continuous_batching.png', dpi=150, bbox_inches='tight')
    plt.show()

def cost_analysis():
    """Compare serving costs"""
    # Cost per 1M tokens
    scenarios = {
        'Research\n(single GPU)': {
            'HuggingFace': 100,
            'vLLM': 20,
            'TensorRT': 15
        },
        'Production\n(multi-GPU)': {
            'HuggingFace': 500,
            'vLLM': 50,
            'TensorRT': 30
        },
        'High Scale\n(data center)': {
            'HuggingFace': 2000,
            'vLLM': 100,
            'TensorRT': 60
        }
    }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(scenarios))
    width = 0.25
    
    stacks = ['HuggingFace', 'vLLM', 'TensorRT']
    colors = ['blue', 'green', 'red']
    
    for i, (stack, color) in enumerate(zip(stacks, colors)):
        values = [scenarios[scenario][stack] for scenario in scenarios.keys()]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=stack, color=color, alpha=0.7)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${val}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Deployment Scenario')
    ax.set_ylabel('Cost per 1M Tokens ($)')
    ax.set_title('Serving Cost Comparison\n(Approximate compute costs)')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios.keys())
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('cost_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def deployment_decision_tree():
    """Create decision tree for choosing stack"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    # Decision tree structure
    tree_text = """
    CHOOSING A PRODUCTION STACK
    
    START: What's your priority?
    │
    ├─ EASE OF USE / PROTOTYPING
    │  └─→ HuggingFace Transformers
    │     • Pros: Easy, flexible, all models
    │     • Cons: Slow, not production-ready
    │     • Use case: Research, prototyping
    │
    ├─ HIGH THROUGHPUT / COST EFFICIENCY
    │  │
    │  ├─ GPU Available?
    │  │  ├─ YES → vLLM
    │  │  │  • Pros: 20x faster, continuous batching
    │  │  │  • Cons: CUDA only, some setup
    │  │  │  • Use case: Production serving, APIs
    │  │  │
    │  │  └─ NO → llama.cpp
    │  │     • Pros: CPU support, quantization
    │  │     • Cons: Slower, limited models
    │  │     • Use case: Edge devices, local deploy
    │  │
    │  └─ Budget for optimization?
    │     └─ HIGH → TensorRT-LLM
    │        • Pros: Fastest (30x), best latency
    │        • Cons: Complex, NVIDIA only
    │        • Use case: High-scale production
    │
    └─ SPECIAL REQUIREMENTS
       ├─ Edge deployment → llama.cpp + INT4
       ├─ Lowest latency → TensorRT-LLM + FP8
       ├─ Highest batch → vLLM + PagedAttention
       └─ Multi-platform → ONNX Runtime
    
    PRODUCTION CHECKLIST:
    ✓ Monitoring (latency, throughput, errors)
    ✓ Load balancing across replicas
    ✓ Rate limiting and quotas
    ✓ Model versioning and rollback
    ✓ Cache warming and preloading
    ✓ Graceful degradation
    """
    
    ax.text(0.05, 0.95, tree_text, transform=ax.transAxes,
           fontfamily='monospace', fontsize=9,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('deployment_decision_tree.png', dpi=150, bbox_inches='tight')
    plt.show()

def run_experiments():
    print("Production Stacks Analysis")
    print("=" * 50)
    
    print("\n1. Comparing production stacks...")
    specs = compare_stacks()
    print("   Saved: stack_comparison.png")
    
    print("\n2. Demonstrating continuous batching...")
    continuous_batching_visualization()
    print("   Saved: continuous_batching.png")
    
    print("\n3. Analyzing costs...")
    cost_analysis()
    print("   Saved: cost_analysis.png")
    
    print("\n4. Creating deployment decision tree...")
    deployment_decision_tree()
    print("   Saved: deployment_decision_tree.png")
    
    print("\n" + "=" * 50)
    print("STACK RECOMMENDATIONS:")
    print("• Research/Prototyping: HuggingFace (easy, flexible)")
    print("• Production API: vLLM (20x faster, great batching)")
    print("• Maximum Performance: TensorRT-LLM (30x faster)")
    print("• Edge/CPU: llama.cpp (runs anywhere)")
    print("• Multi-cloud: ONNX Runtime (portable)")
    print("\nProduction reality: Most companies use vLLM")

if __name__ == "__main__":
    run_experiments()
