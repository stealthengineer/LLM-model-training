# visualize_bpe.py
from bpe_tokenizer import BytePairEncoder
import matplotlib.pyplot as plt

# Train BPE with different vocab sizes
vocab_sizes = [50, 100, 200, 500]
corpus = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning algorithms are powerful",
    "Natural language processing is fascinating"
] * 50

results = []
for size in vocab_sizes:
    bpe = BytePairEncoder(vocab_size=size)
    bpe.train(corpus, verbose=False)
    
    # Test on same text
    test = "machine learning algorithms"
    analysis = bpe.analyze_tokenization(test)
    
    results.append({
        'vocab_size': size,
        'num_tokens': analysis['num_tokens'],
        'compression': analysis['compression_ratio']
    })
    
    print(f"Vocab {size}: {analysis['num_tokens']} tokens, {analysis['compression_ratio']:.2f}x compression")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

vocab = [r['vocab_size'] for r in results]
tokens = [r['num_tokens'] for r in results]
compression = [r['compression'] for r in results]

ax1.plot(vocab, tokens, marker='o', linewidth=2, markersize=8)
ax1.set_xlabel('Vocabulary Size')
ax1.set_ylabel('Number of Tokens')
ax1.set_title('Tokenization Efficiency')
ax1.grid(True, alpha=0.3)

ax2.plot(vocab, compression, marker='s', color='green', linewidth=2, markersize=8)
ax2.set_xlabel('Vocabulary Size')
ax2.set_ylabel('Compression Ratio')
ax2.set_title('Compression vs Vocabulary Size')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bpe_analysis.png', dpi=150, bbox_inches='tight')
print(f"\nVisualization saved as 'bpe_analysis.png'")
plt.show()
