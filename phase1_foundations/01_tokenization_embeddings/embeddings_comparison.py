# embeddings_comparison.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from bpe_tokenizer import BytePairEncoder

class EmbeddingComparison:
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        
    def create_one_hot_embeddings(self):
        """One-hot encoding: each token is a sparse vector"""
        return np.eye(self.vocab_size)
    
    def create_random_embeddings(self, dim=128):
        """Random dense embeddings (simulating learned embeddings)"""
        np.random.seed(42)  # For reproducibility
        return np.random.randn(self.vocab_size, dim) * 0.1
    
    def create_learned_embeddings_simulation(self, dim=128):
        """Simulate 'learned' embeddings with semantic structure"""
        np.random.seed(42)
        embeddings = np.random.randn(self.vocab_size, dim) * 0.1
        
        # Simulate some semantic clustering
        # Tokens 0-20: common words (cluster together)
        embeddings[0:20] += np.random.randn(1, dim) * 0.3
        
        # Tokens 20-40: technical terms (different cluster)  
        embeddings[20:40] += np.random.randn(1, dim) * 0.3 + 1.0
        
        # Tokens 40-60: punctuation/special (another cluster)
        embeddings[40:60] += np.random.randn(1, dim) * 0.3 - 1.0
        
        return embeddings
    
    def compare_similarity_patterns(self):
        """Compare how different embeddings represent token relationships"""
        # Create embeddings
        one_hot = self.create_one_hot_embeddings()
        random_embed = self.create_random_embeddings()
        learned_embed = self.create_learned_embeddings_simulation()
        
        # Calculate similarity matrices
        one_hot_sim = cosine_similarity(one_hot[:50], one_hot[:50])
        random_sim = cosine_similarity(random_embed[:50], random_embed[:50])
        learned_sim = cosine_similarity(learned_embed[:50], learned_embed[:50])
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # One-hot similarities
        im1 = axes[0].imshow(one_hot_sim, cmap='coolwarm', vmin=-1, vmax=1)
        axes[0].set_title('One-Hot Embeddings\n(No Similarity Between Different Tokens)')
        axes[0].set_xlabel('Token ID')
        axes[0].set_ylabel('Token ID')
        
        # Random similarities
        im2 = axes[1].imshow(random_sim, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1].set_title('Random Embeddings\n(Random Similarities)')
        axes[1].set_xlabel('Token ID')
        axes[1].set_ylabel('Token ID')
        
        # "Learned" similarities
        im3 = axes[2].imshow(learned_sim, cmap='coolwarm', vmin=-1, vmax=1)
        axes[2].set_title('Learned Embeddings\n(Semantic Clusters)')
        axes[2].set_xlabel('Token ID')
        axes[2].set_ylabel('Token ID')
        
        # Add colorbar
        plt.colorbar(im3, ax=axes.ravel().tolist(), label='Cosine Similarity')
        
        plt.suptitle('Embedding Type Comparison: Similarity Patterns', fontsize=14)
        plt.tight_layout()
        plt.savefig('embedding_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return one_hot_sim, random_sim, learned_sim
    
    def plot_compression_analysis(self):
        """Analyze space efficiency of different embeddings"""
        vocab_sizes = [50, 100, 200, 500, 1000]
        
        # Calculate memory usage
        one_hot_memory = [v * v * 4 / 1024 for v in vocab_sizes]  # float32, in KB
        dense_memory_128 = [v * 128 * 4 / 1024 for v in vocab_sizes]
        dense_memory_256 = [v * 256 * 4 / 1024 for v in vocab_sizes]
        
        plt.figure(figsize=(10, 6))
        plt.plot(vocab_sizes, one_hot_memory, marker='o', label='One-Hot', linewidth=2)
        plt.plot(vocab_sizes, dense_memory_128, marker='s', label='Dense (128d)', linewidth=2)
        plt.plot(vocab_sizes, dense_memory_256, marker='^', label='Dense (256d)', linewidth=2)
        
        plt.xlabel('Vocabulary Size')
        plt.ylabel('Memory Usage (KB)')
        plt.title('Memory Efficiency: One-Hot vs Dense Embeddings')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig('embedding_memory.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def analyze_with_bpe(self):
        """Compare embeddings with actual BPE tokens"""
        # Train a small BPE
        corpus = [
            "the quick brown fox",
            "machine learning algorithms", 
            "natural language processing"
        ] * 20
        
        bpe = BytePairEncoder(vocab_size=50)
        bpe.train(corpus, verbose=False)
        
        # Get some actual tokens
        test_texts = ["the", "machine", "learning", "fox", "algorithm"]
        token_ids = []
        token_names = []
        
        for text in test_texts:
            ids = bpe.tokenize(text)
            if ids:  # If tokenization successful
                token_ids.append(ids[0])
                token_names.append(text)
        
        print("Token Analysis:")
        print("-" * 40)
        for name, tid in zip(token_names, token_ids):
            print(f"'{name}' -> Token ID: {tid}")
        
        # Create embeddings for these specific tokens
        one_hot = self.create_one_hot_embeddings()
        learned = self.create_learned_embeddings_simulation(dim=50)
        
        if len(token_ids) >= 2:
            # Compare specific token pairs
            print("\nSimilarity Analysis:")
            print("-" * 40)
            
            for i in range(len(token_ids)-1):
                for j in range(i+1, len(token_ids)):
                    id1, id2 = token_ids[i], token_ids[j]
                    
                    if id1 < len(one_hot) and id2 < len(one_hot):
                        oh_sim = cosine_similarity([one_hot[id1]], [one_hot[id2]])[0][0]
                        learned_sim = cosine_similarity([learned[id1]], [learned[id2]])[0][0]
                        
                        print(f"'{token_names[i]}' vs '{token_names[j]}':")
                        print(f"  One-hot similarity: {oh_sim:.3f}")
                        print(f"  Learned similarity: {learned_sim:.3f}")

# Run the analysis
if __name__ == "__main__":
    print("Comparing Embedding Types...")
    print("=" * 50)
    
    comparator = EmbeddingComparison(vocab_size=100)
    
    # Compare similarity patterns
    print("\n1. Creating similarity visualizations...")
    comparator.compare_similarity_patterns()
    print("   Saved as 'embedding_comparison.png'")
    
    # Analyze memory usage
    print("\n2. Analyzing memory efficiency...")
    comparator.plot_compression_analysis()
    print("   Saved as 'embedding_memory.png'")
    
    # Analyze with real BPE tokens
    print("\n3. Analyzing with BPE tokens...")
    comparator.analyze_with_bpe()
    
    print("\n" + "=" * 50)
    print("Analysis complete! Check the generated images.")
