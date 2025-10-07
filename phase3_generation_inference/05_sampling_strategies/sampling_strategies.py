# sampling_strategies.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as mpatches

class SamplingStrategies:
    """Different sampling methods for text generation"""
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        # Simulate a vocabulary
        self.vocab = [f"token_{i}" for i in range(vocab_size)]
        
    def softmax(self, logits, temperature=1.0):
        """Apply softmax with temperature"""
        if temperature == 0:
            # Greedy decoding
            probs = np.zeros_like(logits)
            probs[np.argmax(logits)] = 1.0
            return probs
        
        # Apply temperature
        logits = logits / temperature
        
        # Stable softmax
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)
    
    def greedy_sampling(self, logits):
        """Always pick the highest probability token"""
        return np.argmax(logits)
    
    def temperature_sampling(self, logits, temperature=1.0):
        """Sample with temperature control"""
        probs = self.softmax(logits, temperature)
        return np.random.choice(len(probs), p=probs)
    
    def top_k_sampling(self, logits, k=10, temperature=1.0):
        """Sample from top-k tokens only"""
        # Get top k indices
        top_k_indices = np.argsort(logits)[-k:]
        
        # Create filtered logits
        filtered_logits = np.full_like(logits, -np.inf)
        filtered_logits[top_k_indices] = logits[top_k_indices]
        
        # Apply softmax and sample
        probs = self.softmax(filtered_logits, temperature)
        return np.random.choice(len(probs), p=probs)
    
    def top_p_sampling(self, logits, p=0.9, temperature=1.0):
        """Nucleus sampling - sample from smallest set with cumulative prob > p"""
        probs = self.softmax(logits, temperature)
        
        # Sort probabilities
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        
        # Find cutoff
        cumsum_probs = np.cumsum(sorted_probs)
        cutoff_index = np.searchsorted(cumsum_probs, p) + 1
        
        # Create filtered logits
        filtered_logits = np.full_like(logits, -np.inf)
        filtered_logits[sorted_indices[:cutoff_index]] = logits[sorted_indices[:cutoff_index]]
        
        # Apply softmax and sample
        probs = self.softmax(filtered_logits, temperature)
        return np.random.choice(len(probs), p=probs)
    
    def visualize_sampling_effects(self):
        """Interactive visualization of sampling parameters"""
        # Create example logits (simulating model output)
        np.random.seed(42)
        base_logits = np.random.randn(50) * 2
        base_logits[5] = 4  # Make one token highly probable
        base_logits[10] = 3
        base_logits[15] = 2.5
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Temperature effect
        temps = [0.1, 0.5, 1.0, 2.0]
        for i, temp in enumerate(temps):
            ax = axes[i // 2, i % 2]
            probs = self.softmax(base_logits, temperature=temp)
            
            bars = ax.bar(range(len(probs)), probs, color='steelblue', alpha=0.7)
            # Highlight top tokens
            top_indices = np.argsort(probs)[-5:]
            for idx in top_indices:
                bars[idx].set_color('orange')
            
            ax.set_title(f'Temperature = {temp}', fontsize=12)
            ax.set_xlabel('Token ID')
            ax.set_ylabel('Probability')
            ax.set_ylim(0, max(0.5, probs.max() * 1.1))
            
            # Add entropy value
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            ax.text(0.7, 0.9, f'Entropy: {entropy:.2f}', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Temperature Effect on Token Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('temperature_effects.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def compare_sampling_methods(self, num_samples=100):
        """Compare different sampling strategies"""
        np.random.seed(42)
        logits = np.random.randn(20) * 2
        logits[0] = 3  # High probability token
        logits[1] = 2.5
        logits[2] = 2
        
        methods = {
            'Greedy': [],
            'Temp=0.5': [],
            'Temp=1.0': [],
            'Top-k=5': [],
            'Top-p=0.9': []
        }
        
        for _ in range(num_samples):
            methods['Greedy'].append(self.greedy_sampling(logits))
            methods['Temp=0.5'].append(self.temperature_sampling(logits, 0.5))
            methods['Temp=1.0'].append(self.temperature_sampling(logits, 1.0))
            methods['Top-k=5'].append(self.top_k_sampling(logits, k=5))
            methods['Top-p=0.9'].append(self.top_p_sampling(logits, p=0.9))
        
        # Visualize distributions
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        
        for ax, (name, samples) in zip(axes, methods.items()):
            unique, counts = np.unique(samples, return_counts=True)
            ax.bar(unique, counts, color='teal', alpha=0.7)
            ax.set_title(name, fontsize=11)
            ax.set_xlabel('Token ID')
            ax.set_ylabel('Count')
            ax.set_xlim(-1, 20)
            
            # Calculate diversity
            diversity = len(unique)
            ax.text(0.5, 0.9, f'Unique: {diversity}', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        plt.suptitle(f'Sampling Methods Comparison ({num_samples} samples)', fontsize=14)
        plt.tight_layout()
        plt.savefig('sampling_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def demonstrate_repetition_problem(self):
        """Show how greedy/low-temp leads to repetition"""
        sequence_length = 50
        
        # Simulate generation with different temperatures
        fig, axes = plt.subplots(3, 1, figsize=(14, 8))
        
        for i, (temp, ax) in enumerate(zip([0.0, 0.5, 1.0], axes)):
            np.random.seed(42)
            sequence = []
            
            # Simulate generation
            for _ in range(sequence_length):
                # Create logits that favor recent tokens (simulating repetition tendency)
                logits = np.random.randn(10) * 0.5
                if len(sequence) > 0:
                    # Boost probability of recent tokens
                    for recent_token in sequence[-3:]:
                        logits[recent_token] += 1.5
                
                if temp == 0:
                    token = self.greedy_sampling(logits)
                else:
                    token = self.temperature_sampling(logits, temp)
                
                sequence.append(token)
            
            # Visualize sequence
            ax.imshow([sequence], aspect='auto', cmap='tab10', interpolation='nearest')
            ax.set_title(f'Temperature = {temp}', fontsize=12)
            ax.set_ylabel('Sequence')
            ax.set_xlabel('Position')
            ax.set_yticks([])
            
            # Count repetitions
            unique_tokens = len(set(sequence))
            ax.text(1.02, 0.5, f'Unique tokens: {unique_tokens}/10', 
                   transform=ax.transAxes, va='center')
        
        plt.suptitle('Repetition Problem with Different Temperatures', fontsize=14)
        plt.tight_layout()
        plt.savefig('repetition_problem.png', dpi=150, bbox_inches='tight')
        plt.show()

def run_experiments():
    print("Exploring Sampling Strategies...")
    print("=" * 50)
    
    sampler = SamplingStrategies(vocab_size=1000)
    
    print("\n1. Visualizing temperature effects...")
    sampler.visualize_sampling_effects()
    print("   Saved: temperature_effects.png")
    
    print("\n2. Comparing sampling methods...")
    sampler.compare_sampling_methods()
    print("   Saved: sampling_comparison.png")
    
    print("\n3. Demonstrating repetition problem...")
    sampler.demonstrate_repetition_problem()
    print("   Saved: repetition_problem.png")
    
    print("\n" + "=" * 50)
    print("KEY INSIGHTS:")
    print("- Temperature=0 (Greedy): Deterministic but repetitive")
    print("- Low temp (0.1-0.5): Focused but some variety")
    print("- High temp (1.5-2.0): Creative but sometimes nonsensical")
    print("- Top-k: Limits to k most likely tokens")
    print("- Top-p: Dynamic cutoff based on probability mass")
    print("\nBest practice: Combine top-p=0.9 with temp=0.7-0.9")

if __name__ == "__main__":
    run_experiments()
