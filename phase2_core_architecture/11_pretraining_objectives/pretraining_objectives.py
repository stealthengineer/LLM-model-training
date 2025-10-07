# pretraining_objectives.py
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class PretrainingObjectives:
    """Different pretraining objectives for language models"""
    
    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size
    
    def causal_lm_loss(self, logits: np.ndarray, targets: np.ndarray, mask: np.ndarray = None):
        """
        Causal Language Modeling (GPT-style)
        Predict next token given all previous tokens
        """
        # Shift targets (predict next token)
        # logits: (batch, seq_len, vocab)
        # targets: (batch, seq_len)
        
        batch_size, seq_len, _ = logits.shape
        
        # Flatten for easier computation
        logits_flat = logits[:, :-1, :].reshape(-1, self.vocab_size)
        targets_flat = targets[:, 1:].reshape(-1)
        
        # Softmax
        exp_logits = np.exp(logits_flat - np.max(logits_flat, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Cross-entropy loss
        loss = -np.log(probs[np.arange(len(targets_flat)), targets_flat] + 1e-10)
        
        if mask is not None:
            mask_flat = mask[:, 1:].reshape(-1)
            loss = loss * mask_flat
            return np.sum(loss) / np.sum(mask_flat)
        
        return np.mean(loss)
    
    def masked_lm_loss(self, logits: np.ndarray, targets: np.ndarray, mask_positions: np.ndarray):
        """
        Masked Language Modeling (BERT-style)
        Predict masked tokens given bidirectional context
        """
        # Only compute loss on masked positions
        batch_size, seq_len, _ = logits.shape
        
        # Extract logits at masked positions
        masked_logits = logits[mask_positions]  # (num_masked, vocab)
        masked_targets = targets[mask_positions]
        
        # Softmax
        exp_logits = np.exp(masked_logits - np.max(masked_logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Cross-entropy
        loss = -np.log(probs[np.arange(len(masked_targets)), masked_targets] + 1e-10)
        
        return np.mean(loss)
    
    def prefix_lm_loss(self, logits: np.ndarray, targets: np.ndarray, 
                       prefix_len: int, mask: np.ndarray = None):
        """
        Prefix Language Modeling (T5-style)
        Bidirectional on prefix, causal on completion
        """
        batch_size, seq_len, _ = logits.shape
        
        # Only predict tokens after prefix
        logits_flat = logits[:, prefix_len:-1, :].reshape(-1, self.vocab_size)
        targets_flat = targets[:, prefix_len+1:].reshape(-1)
        
        # Softmax
        exp_logits = np.exp(logits_flat - np.max(logits_flat, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Cross-entropy
        loss = -np.log(probs[np.arange(len(targets_flat)), targets_flat] + 1e-10)
        
        if mask is not None:
            mask_flat = mask[:, prefix_len+1:].reshape(-1)
            loss = loss * mask_flat
            return np.sum(loss) / np.sum(mask_flat)
        
        return np.mean(loss)

def visualize_attention_patterns():
    """Show what each objective can see"""
    seq_len = 10
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Causal LM (GPT)
    ax1 = axes[0]
    causal_mask = np.tril(np.ones((seq_len, seq_len)))
    im1 = ax1.imshow(causal_mask, cmap='Greens', aspect='auto', vmin=0, vmax=1)
    ax1.set_title('Causal LM (GPT)\nAutoregressive: Only see past', fontweight='bold')
    ax1.set_xlabel('Token Position (can attend to)')
    ax1.set_ylabel('Token Position (predicting)')
    
    # Add annotations
    for i in range(seq_len):
        for j in range(seq_len):
            if causal_mask[i, j] == 1:
                ax1.text(j, i, '✓', ha='center', va='center', fontsize=8)
    
    # Masked LM (BERT)
    ax2 = axes[1]
    # Create masked pattern (random 15% masked)
    np.random.seed(42)
    masked_pattern = np.ones((seq_len, seq_len))
    masked_positions = np.random.choice(seq_len, size=int(seq_len * 0.15), replace=False)
    
    # Show bidirectional attention except at masked positions
    for pos in masked_positions:
        masked_pattern[pos, pos] = 0.5  # Predicting this token
    
    im2 = ax2.imshow(masked_pattern, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax2.set_title('Masked LM (BERT)\nBidirectional: See all context', fontweight='bold')
    ax2.set_xlabel('Token Position (can attend to)')
    ax2.set_ylabel('Token Position')
    
    # Mark masked positions
    for pos in masked_positions:
        ax2.text(pos, pos, '[M]', ha='center', va='center', 
                fontsize=10, fontweight='bold', color='red')
    
    # Prefix LM (T5)
    ax3 = axes[2]
    prefix_len = 4
    prefix_pattern = np.ones((seq_len, seq_len))
    
    # Bidirectional on prefix
    prefix_pattern[:prefix_len, :prefix_len] = 1
    
    # Causal on completion
    for i in range(prefix_len, seq_len):
        for j in range(seq_len):
            if j < prefix_len or j <= i:
                prefix_pattern[i, j] = 1
            else:
                prefix_pattern[i, j] = 0
    
    im3 = ax3.imshow(prefix_pattern, cmap='Oranges', aspect='auto', vmin=0, vmax=1)
    ax3.set_title('Prefix LM (T5)\nBidirectional prefix + Causal completion', fontweight='bold')
    ax3.set_xlabel('Token Position (can attend to)')
    ax3.set_ylabel('Token Position (predicting)')
    
    # Mark prefix boundary
    ax3.axvline(x=prefix_len-0.5, color='red', linestyle='--', linewidth=2)
    ax3.axhline(y=prefix_len-0.5, color='red', linestyle='--', linewidth=2)
    ax3.text(prefix_len/2, -0.5, 'Prefix', ha='center', fontsize=10, fontweight='bold')
    ax3.text(prefix_len + (seq_len-prefix_len)/2, -0.5, 'Completion', 
            ha='center', fontsize=10, fontweight='bold')
    
    plt.suptitle('Pretraining Objectives: What Can the Model See?', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('attention_patterns.png', dpi=150, bbox_inches='tight')
    plt.show()

def simulate_learning_curves():
    """Simulate learning curves for different objectives"""
    np.random.seed(42)
    
    epochs = 100
    
    # Simulate learning (simplified)
    causal_loss = 5 * np.exp(-0.03 * np.arange(epochs)) + np.random.randn(epochs) * 0.1 + 1
    masked_loss = 4 * np.exp(-0.04 * np.arange(epochs)) + np.random.randn(epochs) * 0.1 + 0.8
    prefix_loss = 4.5 * np.exp(-0.035 * np.arange(epochs)) + np.random.randn(epochs) * 0.1 + 0.9
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.plot(causal_loss, label='Causal LM (GPT)', linewidth=2, color='green')
    ax1.plot(masked_loss, label='Masked LM (BERT)', linewidth=2, color='blue')
    ax1.plot(prefix_loss, label='Prefix LM (T5)', linewidth=2, color='orange')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Task performance (simulated)
    tasks = ['Next Token\nPrediction', 'Fill in\nBlanks', 'Q&A', 'Summarization', 'Classification']
    
    # Performance scores (0-100)
    causal_scores = [95, 60, 75, 85, 70]
    masked_scores = [40, 95, 85, 65, 90]
    prefix_scores = [80, 85, 80, 80, 80]
    
    x = np.arange(len(tasks))
    width = 0.25
    
    bars1 = ax2.bar(x - width, causal_scores, width, label='Causal LM', color='green', alpha=0.7)
    bars2 = ax2.bar(x, masked_scores, width, label='Masked LM', color='blue', alpha=0.7)
    bars3 = ax2.bar(x + width, prefix_scores, width, label='Prefix LM', color='orange', alpha=0.7)
    
    ax2.set_ylabel('Performance Score')
    ax2.set_title('Downstream Task Performance\n(Simulated)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(tasks, fontsize=9)
    ax2.legend()
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
    plt.show()

def objective_comparison_table():
    """Create comparison table of objectives"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    data = [
        ['Objective', 'Causal LM\n(GPT)', 'Masked LM\n(BERT)', 'Prefix LM\n(T5)'],
        ['', '', '', ''],
        ['Attention\nPattern', 'Causal\n(left-to-right)', 'Bidirectional\n(all tokens)', 'Hybrid\n(bi + causal)'],
        ['', '', '', ''],
        ['What is\nMasked', 'Future tokens', '~15% random\ntokens', 'Completion\ntokens'],
        ['', '', '', ''],
        ['Training\nEfficiency', 'Medium\n(one prediction\nper token)', 'High\n(many predictions\nper sequence)', 'Medium-High'],
        ['', '', '', ''],
        ['Best For', '• Generation\n• Chat\n• Code completion', '• Classification\n• NER\n• Understanding', '• Seq2seq\n• Translation\n• Summarization'],
        ['', '', '', ''],
        ['Example\nModels', '• GPT-3/4\n• LLaMA\n• Mistral', '• BERT\n• RoBERTa\n• ALBERT', '• T5\n• BART\n• UL2'],
        ['', '', '', ''],
        ['Key\nAdvantage', 'Natural for\ngeneration', 'Better at\nunderstanding', 'Flexible for\nmany tasks'],
        ['', '', '', ''],
        ['Key\nLimitation', 'Only sees past\ncontext', 'Not natural for\ngeneration', 'More complex\nto implement']
    ]
    
    # Color coding
    colors = []
    for i, row in enumerate(data):
        if i == 0:  # Header
            colors.append(['lightgray', 'lightgreen', 'lightblue', 'lightyellow'])
        elif i % 2 == 1:  # Empty row
            colors.append(['white', 'white', 'white', 'white'])
        else:
            colors.append(['lightgray', 'white', 'white', 'white'])
    
    table = ax.table(cellText=data, cellLoc='left', loc='center',
                    cellColours=colors, colWidths=[0.15, 0.28, 0.28, 0.28])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_text_props(weight='bold', size=11)
        table[(0, i)].set_facecolor('darkgray')
    
    # Style first column
    for i in range(len(data)):
        if i % 2 == 0 and i > 0:
            table[(i, 0)].set_text_props(weight='bold')
    
    plt.title('Pretraining Objectives Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.savefig('objectives_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def training_efficiency_analysis():
    """Compare training efficiency"""
    batch_size = 32
    seq_len = 512
    
    # Causal LM: predict 1 token per position (except first)
    causal_predictions = batch_size * (seq_len - 1)
    
    # Masked LM: predict ~15% of tokens
    masked_predictions = batch_size * int(seq_len * 0.15)
    
    # Prefix LM: predict completion tokens (assume 50% is completion)
    prefix_predictions = batch_size * int(seq_len * 0.5)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Predictions per batch
    objectives = ['Causal LM', 'Masked LM', 'Prefix LM']
    predictions = [causal_predictions, masked_predictions, prefix_predictions]
    colors_list = ['green', 'blue', 'orange']
    
    bars = ax1.bar(objectives, predictions, color=colors_list, alpha=0.7)
    ax1.set_ylabel('Predictions per Batch')
    ax1.set_title('Training Efficiency\n(Predictions per Forward Pass)')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, predictions):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:,}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Effective throughput (tokens seen per prediction)
    # Causal: sees 1 to N tokens for N predictions
    # Masked: sees all N tokens for 0.15N predictions
    # Prefix: sees all N tokens for 0.5N predictions
    
    causal_efficiency = seq_len / 2  # Average context
    masked_efficiency = seq_len  # Full context
    prefix_efficiency = seq_len  # Full context for predictions
    
    efficiencies = [causal_efficiency, masked_efficiency, prefix_efficiency]
    
    bars = ax2.bar(objectives, efficiencies, color=colors_list, alpha=0.7)
    ax2.set_ylabel('Average Context Length per Prediction')
    ax2.set_title('Context Utilization\n(Higher = More efficient use of context)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, efficiencies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('training_efficiency.png', dpi=150, bbox_inches='tight')
    plt.show()

def run_experiments():
    print("Pretraining Objectives Analysis")
    print("=" * 50)
    
    print("\n1. Visualizing attention patterns...")
    visualize_attention_patterns()
    print("   Saved: attention_patterns.png")
    
    print("\n2. Simulating learning curves...")
    simulate_learning_curves()
    print("   Saved: learning_curves.png")
    
    print("\n3. Creating comparison table...")
    objective_comparison_table()
    print("   Saved: objectives_comparison.png")
    
    print("\n4. Analyzing training efficiency...")
    training_efficiency_analysis()
    print("   Saved: training_efficiency.png")
    
    print("\n" + "=" * 50)
    print("KEY INSIGHTS:")
    print("1. Causal LM: Best for generation (GPT, LLaMA)")
    print("2. Masked LM: Best for understanding (BERT)")
    print("3. Prefix LM: Balanced approach (T5)")
    print("4. Masked LM is most sample-efficient (sees full context)")
    print("5. Causal LM is most natural for chat/generation")
    print("6. Modern trend: Causal LM dominates (GPT-4, Claude)")
    print("\nWhy causal won: Generation is the killer app")

if __name__ == "__main__":
    run_experiments()
