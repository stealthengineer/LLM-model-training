import re
from collections import defaultdict, Counter
import json
from typing import Dict, List, Tuple, Set
import matplotlib.pyplot as plt

class BytePairEncoder:
    """
    A byte-pair encoding tokenizer implementation from scratch.
    """
    
    def __init__(self, vocab_size: int = 1000):
        """
        Initialize BPE tokenizer.
        
        Args:
            vocab_size: Target vocabulary size (includes base characters)
        """
        self.vocab_size = vocab_size
        self.word_freqs = defaultdict(int)
        self.vocab = {}
        self.merges = []
        
    def _get_word_frequencies(self, corpus: List[str]) -> Dict[str, int]:
        """
        Count word frequencies in corpus.
        """
        word_freqs = defaultdict(int)
        for text in corpus:
            # Simple tokenization - you can make this more sophisticated
            words = text.lower().split()
            for word in words:
                # Add end-of-word token
                word = ' '.join(list(word)) + ' </w>'
                word_freqs[word] += 1
        return dict(word_freqs)
    
    def _get_pair_frequencies(self, word_freqs: Dict[str, int]) -> Counter:
        """
        Count frequencies of adjacent token pairs across all words.
        """
        pairs = Counter()
        for word, freq in word_freqs.items():
            tokens = word.split()
            for i in range(len(tokens) - 1):
                pairs[(tokens[i], tokens[i + 1])] += freq
        return pairs
    
    def _merge_pair(self, pair: Tuple[str, str], word_freqs: Dict[str, int]) -> Dict[str, int]:
        """
        Merge most frequent pair in all words.
        """
        new_word_freqs = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word, freq in word_freqs.items():
            # Replace the bigram with merged version
            new_word = word.replace(bigram, replacement)
            new_word_freqs[new_word] = freq
            
        return new_word_freqs
    
    def train(self, corpus: List[str], verbose: bool = True) -> None:
        """
        Train BPE on corpus.
        
        Args:
            corpus: List of text strings to train on
            verbose: Print training progress
        """
        # Get initial word frequencies
        self.word_freqs = self._get_word_frequencies(corpus)
        
        # Get unique characters (base vocabulary)
        chars = set()
        for word in self.word_freqs.keys():
            chars.update(word.split())
        
        # Initialize vocabulary with base characters
        self.vocab = {char: idx for idx, char in enumerate(sorted(chars))}
        
        if verbose:
            print(f"Initial vocabulary size: {len(self.vocab)}")
            print(f"Base characters: {sorted(chars)[:20]}...")
        
        # Iteratively merge most frequent pairs
        num_merges = self.vocab_size - len(self.vocab)
        
        for i in range(num_merges):
            # Get pair frequencies
            pairs = self._get_pair_frequencies(self.word_freqs)
            
            if not pairs:
                break
                
            # Find most frequent pair
            most_frequent = pairs.most_common(1)[0]
            pair, frequency = most_frequent
            
            if verbose and i % 50 == 0:
                print(f"Merge {i}: {pair} (frequency: {frequency})")
            
            # Merge the pair
            self.merges.append(pair)
            self.word_freqs = self._merge_pair(pair, self.word_freqs)
            
            # Add merged token to vocabulary
            merged = ''.join(pair)
            if merged not in self.vocab:
                self.vocab[merged] = len(self.vocab)
        
        if verbose:
            print(f"\nFinal vocabulary size: {len(self.vocab)}")
            print(f"Number of merges: {len(self.merges)}")
    
    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize text using learned BPE.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of token IDs
        """
        words = text.lower().split()
        tokens = []
        
        for word in words:
            # Convert word to character tokens with end-of-word marker
            word_tokens = list(word) + ['</w>']
            
            # Apply merges in order
            for pair in self.merges:
                i = 0
                while i < len(word_tokens) - 1:
                    if (word_tokens[i], word_tokens[i + 1]) == pair:
                        # Merge the pair
                        word_tokens = word_tokens[:i] + [''.join(pair)] + word_tokens[i + 2:]
                    else:
                        i += 1
            
            # Convert to IDs
            for token in word_tokens:
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                else:
                    # Handle out-of-vocabulary tokens
                    # In practice, you might want to use <UNK> token
                    for char in token:
                        if char in self.vocab:
                            tokens.append(self.vocab[char])
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        # Create reverse vocabulary
        id_to_token = {idx: token for token, idx in self.vocab.items()}
        
        # Convert IDs to tokens
        tokens = []
        for idx in token_ids:
            if idx in id_to_token:
                tokens.append(id_to_token[idx])
        
        # Join tokens and clean up
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        return text.strip()
    
    def save_vocab(self, filepath: str) -> None:
        """Save vocabulary and merges to file."""
        data = {
            'vocab': self.vocab,
            'merges': [list(pair) for pair in self.merges]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Vocabulary saved to {filepath}")
    
    def load_vocab(self, filepath: str) -> None:
        """Load vocabulary and merges from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.vocab = data['vocab']
        self.merges = [tuple(pair) for pair in data['merges']]
        print(f"Vocabulary loaded from {filepath}")
    
    def visualize_merges(self, num_merges: int = 20) -> None:
        """
        Visualize the first N merges.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Most common merges
        merge_strings = [f"{pair[0]}+{pair[1]}" for pair in self.merges[:num_merges]]
        positions = range(len(merge_strings))
        
        ax1.barh(positions, range(len(merge_strings), 0, -1))
        ax1.set_yticks(positions)
        ax1.set_yticklabels(merge_strings, fontsize=10)
        ax1.set_xlabel('Merge Order (earlier = more frequent)')
        ax1.set_title(f'First {num_merges} BPE Merges')
        ax1.invert_yaxis()
        
        # Plot 2: Vocabulary growth
        vocab_sizes = [len(self.vocab) - len(self.merges) + i for i in range(len(self.merges) + 1)]
        ax2.plot(vocab_sizes, linewidth=2)
        ax2.set_xlabel('Number of Merges')
        ax2.set_ylabel('Vocabulary Size')
        ax2.set_title('Vocabulary Growth During Training')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_tokenization(self, text: str) -> Dict:
        """
        Analyze tokenization of given text.
        """
        tokens = self.tokenize(text)
        
        # Create reverse vocab for analysis
        id_to_token = {idx: token for token, idx in self.vocab.items()}
        token_strings = [id_to_token.get(t, '<UNK>') for t in tokens]
        
        analysis = {
            'original_text': text,
            'num_tokens': len(tokens),
            'num_characters': len(text),
            'compression_ratio': len(text) / len(tokens),
            'tokens': token_strings,
            'token_ids': tokens
        }
        
        return analysis


# Example usage and experiments
def demo_bpe():
    """
    Demonstrate BPE training and usage.
    """
    # Sample corpus (you can use any text data)
    corpus = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is fascinating and powerful",
        "Natural language processing with transformers",
        "Deep learning models require lots of data",
        "Tokenization is the first step in NLP",
        "Byte pair encoding helps with subword tokenization",
        "GPT models use BPE for tokenization",
        "The transformer architecture revolutionized NLP",
        "Attention is all you need for many tasks",
        "Large language models are trained on massive datasets"
    ] * 10  # Repeat to get more frequency patterns
    
    # Train BPE
    print("Training BPE tokenizer...")
    print("=" * 50)
    
    bpe = BytePairEncoder(vocab_size=150)
    bpe.train(corpus, verbose=True)
    
    print("\n" + "=" * 50)
    print("Testing tokenization...")
    print("=" * 50)
    
    # Test tokenization
    test_texts = [
        "machine learning",
        "tokenization",
        "transformer",
        "unknown cryptocurrency"  # Contains OOV words
    ]
    
    for text in test_texts:
        analysis = bpe.analyze_tokenization(text)
        print(f"\nText: '{text}'")
        print(f"Tokens: {analysis['tokens']}")
        print(f"Compression ratio: {analysis['compression_ratio']:.2f}")
        
        # Verify round-trip
        decoded = bpe.decode(analysis['token_ids'])
        print(f"Decoded: '{decoded}'")
    
    # Visualize merges
    print("\n" + "=" * 50)
    print("Generating visualizations...")
    bpe.visualize_merges(num_merges=30)
    
    # Save vocabulary
    bpe.save_vocab("bpe_vocab.json")
    
    return bpe


def compare_tokenization_methods():
    """
    Compare different tokenization approaches.
    """
    text = "The transformer architecture revolutionized natural language processing"
    
    # Character-level tokenization
    char_tokens = list(text.lower())
    
    # Word-level tokenization  
    word_tokens = text.lower().split()
    
    # Train BPE with different vocab sizes
    corpus = [text] * 100
    
    bpe_small = BytePairEncoder(vocab_size=50)
    bpe_small.train(corpus, verbose=False)
    
    bpe_large = BytePairEncoder(vocab_size=200)
    bpe_large.train(corpus, verbose=False)
    
    # Compare
    print("Tokenization Comparison")
    print("=" * 50)
    print(f"Original text: {text}")
    print(f"Character-level: {len(char_tokens)} tokens")
    print(f"Word-level: {len(word_tokens)} tokens")
    print(f"BPE (small vocab): {len(bpe_small.tokenize(text))} tokens")
    print(f"BPE (large vocab): {len(bpe_large.tokenize(text))} tokens")
    
    # Visualize compression ratios
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['Character', 'Word', 'BPE-50', 'BPE-200']
    token_counts = [
        len(char_tokens),
        len(word_tokens),
        len(bpe_small.tokenize(text)),
        len(bpe_large.tokenize(text))
    ]
    
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4', '#d62728']
    bars = ax.bar(methods, token_counts, color=colors)
    
    ax.set_ylabel('Number of Tokens')
    ax.set_title('Token Count by Tokenization Method')
    ax.set_ylim(0, max(token_counts) * 1.2)
    
    # Add value labels on bars
    for bar, count in zip(bars, token_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run the demo
    bpe_tokenizer = demo_bpe()
    
    print("\n" + "=" * 50)
    print("Comparing tokenization methods...")
    print("=" * 50)
    compare_tokenization_methods()
    
    # Advanced experiment: vocabulary size vs compression
    print("\n" + "=" * 50)
    print("Experiment: Vocabulary Size vs Compression Ratio")
    print("=" * 50)
    
    corpus = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning algorithms process data efficiently"
    ] * 50
    
    vocab_sizes = [50, 100, 200, 500, 1000]
    compression_ratios = []
    
    test_text = "machine learning transformers attention mechanisms"
    
    for vocab_size in vocab_sizes:
        bpe = BytePairEncoder(vocab_size=vocab_size)
        bpe.train(corpus, verbose=False)
        analysis = bpe.analyze_tokenization(test_text)
        compression_ratios.append(analysis['compression_ratio'])
        print(f"Vocab size {vocab_size}: compression ratio = {analysis['compression_ratio']:.2f}")
    
    # Plot vocabulary size vs compression
    plt.figure(figsize=(10, 6))
    plt.plot(vocab_sizes, compression_ratios, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Vocabulary Size')
    plt.ylabel('Compression Ratio (chars/token)')
    plt.title('BPE Vocabulary Size vs Compression Ratio')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()