# test_bpe.py
from bpe_tokenizer import BytePairEncoder, demo_bpe

print("Testing BPE Tokenizer...")
print("=" * 50)

# Run the full demo - this will show more interesting results
demo_bpe()

print("\n" + "=" * 50)
print("Custom test with larger vocabulary...")

# Create a more substantial test
bpe = BytePairEncoder(vocab_size=200)

# Bigger corpus for better merge patterns
corpus = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning algorithms are powerful",
    "Natural language processing is fascinating",
    "Tokenization breaks text into tokens",
    "Byte pair encoding compresses common patterns"
] * 20  # More repetition for better patterns

print("\nTraining on larger corpus...")
bpe.train(corpus, verbose=True)

# Test various texts
print("\n" + "=" * 50)
print("Testing various tokenizations:")

test_cases = [
    "machine learning",
    "the quick fox",
    "tokenization algorithm",
    "new unseen words"
]

for text in test_cases:
    analysis = bpe.analyze_tokenization(text)
    print(f"\nText: '{text}'")
    print(f"Number of tokens: {analysis['num_tokens']}")
    print(f"Token representation: {analysis['tokens'][:10]}")  # Show first 10
    print(f"Compression ratio: {analysis['compression_ratio']:.2f}x")
    
# Show the merges that were learned
print("\n" + "=" * 50)
bpe.visualize_merges(num_merges=20)
