# mini_gpt.py
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle

class MiniGPT:
    """A minimal GPT implementation combining all components"""
    
    def __init__(self, vocab_size=100, d_model=128, n_heads=4, n_layers=2, seq_len=64):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.d_k = d_model // n_heads
        
        # Initialize all parameters
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        """Initialize all model parameters"""
        # Token embeddings
        self.token_embeddings = np.random.randn(self.vocab_size, self.d_model) * 0.02
        
        # Positional embeddings
        self.positional_embeddings = self._create_positional_embeddings()
        
        # Transformer layers
        self.layers = []
        for _ in range(self.n_layers):
            layer = {
                # Attention weights
                'W_q': np.random.randn(self.d_model, self.d_model) * np.sqrt(2.0 / self.d_model),
                'W_k': np.random.randn(self.d_model, self.d_model) * np.sqrt(2.0 / self.d_model),
                'W_v': np.random.randn(self.d_model, self.d_model) * np.sqrt(2.0 / self.d_model),
                'W_o': np.random.randn(self.d_model, self.d_model) * np.sqrt(2.0 / self.d_model),
                
                # FFN weights
                'W_ff1': np.random.randn(self.d_model, self.d_model * 4) * np.sqrt(2.0 / self.d_model),
                'b_ff1': np.zeros(self.d_model * 4),
                'W_ff2': np.random.randn(self.d_model * 4, self.d_model) * np.sqrt(2.0 / (self.d_model * 4)),
                'b_ff2': np.zeros(self.d_model),
                
                # Layer norm parameters
                'ln1_gamma': np.ones(self.d_model),
                'ln1_beta': np.zeros(self.d_model),
                'ln2_gamma': np.ones(self.d_model),
                'ln2_beta': np.zeros(self.d_model),
            }
            self.layers.append(layer)
        
        # Output projection
        self.W_out = np.random.randn(self.d_model, self.vocab_size) * 0.02
        self.b_out = np.zeros(self.vocab_size)
        
        # Final layer norm
        self.ln_f_gamma = np.ones(self.d_model)
        self.ln_f_beta = np.zeros(self.d_model)
        
    def _create_positional_embeddings(self):
        """Create sinusoidal positional embeddings"""
        position = np.arange(self.seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        
        pos_emb = np.zeros((self.seq_len, self.d_model))
        pos_emb[:, 0::2] = np.sin(position * div_term)
        pos_emb[:, 1::2] = np.cos(position * div_term)
        
        return pos_emb
    
    def layer_norm(self, x, gamma, beta, eps=1e-5):
        """Layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma * x_norm + beta
    
    def gelu(self, x):
        """GELU activation function"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    def attention(self, Q, K, V, mask=None):
        """Scaled dot-product attention"""
        scores = np.matmul(Q, K.swapaxes(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores + mask * -1e9
        
        # Softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        output = np.matmul(weights, V)
        return output, weights
    
    def transformer_block(self, x, layer_idx, mask=None):
        """Single transformer block"""
        layer = self.layers[layer_idx]
        
        # Multi-head attention
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = np.dot(x, layer['W_q']).reshape(batch_size, seq_len, self.n_heads, self.d_k)
        K = np.dot(x, layer['W_k']).reshape(batch_size, seq_len, self.n_heads, self.d_k)
        V = np.dot(x, layer['W_v']).reshape(batch_size, seq_len, self.n_heads, self.d_k)
        
        # Transpose for attention
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        # Apply attention
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        attn_output = np.dot(attn_output, layer['W_o'])
        
        # Add & norm
        x = self.layer_norm(x + attn_output, layer['ln1_gamma'], layer['ln1_beta'])
        
        # Feed-forward network
        ff_output = np.dot(x, layer['W_ff1']) + layer['b_ff1']
        ff_output = self.gelu(ff_output)
        ff_output = np.dot(ff_output, layer['W_ff2']) + layer['b_ff2']
        
        # Add & norm
        x = self.layer_norm(x + ff_output, layer['ln2_gamma'], layer['ln2_beta'])
        
        return x, attn_weights
    
    def forward(self, input_ids):
        """Forward pass through the model"""
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.token_embeddings[input_ids]
        
        # Add positional embeddings
        x = x + self.positional_embeddings[:seq_len]
        
        # Create causal mask
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        
        # Pass through transformer layers
        attention_maps = []
        for i in range(self.n_layers):
            x, attn_weights = self.transformer_block(x, i, mask)
            attention_maps.append(attn_weights)
        
        # Final layer norm
        x = self.layer_norm(x, self.ln_f_gamma, self.ln_f_beta)
        
        # Output projection
        logits = np.dot(x, self.W_out) + self.b_out
        
        return logits, attention_maps
    
    def generate(self, prompt_ids, max_length=50, temperature=0.8, top_p=0.9):
        """Generate text given a prompt"""
        generated = list(prompt_ids)
        
        for _ in range(max_length - len(prompt_ids)):
            # Prepare input
            input_ids = np.array(generated[-self.seq_len:]).reshape(1, -1)
            
            # Forward pass
            logits, _ = self.forward(input_ids)
            
            # Get next token logits
            next_token_logits = logits[0, -1, :]
            
            # Apply temperature and sampling
            next_token = self.sample_top_p(next_token_logits, temperature, top_p)
            
            generated.append(next_token)
            
            # Stop if we hit end token (assuming 0 is end)
            if next_token == 0:
                break
        
        return generated
    
    def sample_top_p(self, logits, temperature=1.0, top_p=0.9):
        """Top-p (nucleus) sampling"""
        # Apply temperature
        logits = logits / temperature
        
        # Convert to probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Sort and find cutoff
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumsum = np.cumsum(sorted_probs)
        cutoff_idx = np.searchsorted(cumsum, top_p) + 1
        
        # Filter and renormalize
        filtered_indices = sorted_indices[:cutoff_idx]
        filtered_probs = probs[filtered_indices]
        filtered_probs = filtered_probs / np.sum(filtered_probs)
        
        # Sample
        choice = np.random.choice(filtered_indices, p=filtered_probs)
        return choice
    
    def visualize_attention_patterns(self, text_ids):
        """Visualize attention patterns for given text"""
        # Forward pass
        logits, attention_maps = self.forward(text_ids.reshape(1, -1))
        
        # Plot attention for each layer
        fig, axes = plt.subplots(1, self.n_layers, figsize=(6 * self.n_layers, 5))
        if self.n_layers == 1:
            axes = [axes]
        
        for layer_idx, attn in enumerate(attention_maps):
            # Average over batch and heads
            avg_attn = attn[0].mean(axis=0)  # Shape: (seq_len, seq_len)
            
            im = axes[layer_idx].imshow(avg_attn, cmap='Blues', aspect='auto')
            axes[layer_idx].set_title(f'Layer {layer_idx + 1} Attention')
            axes[layer_idx].set_xlabel('Key Position')
            axes[layer_idx].set_ylabel('Query Position')
            plt.colorbar(im, ax=axes[layer_idx])
        
        plt.suptitle('Attention Patterns Across Layers', fontsize=14)
        plt.tight_layout()
        plt.savefig('mini_gpt_attention.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def count_parameters(self):
        """Count total number of parameters"""
        total = 0
        # Embeddings
        total += self.token_embeddings.size
        total += self.W_out.size
        total += self.b_out.size
        
        # Transformer layers
        for layer in self.layers:
            for param in layer.values():
                total += param.size
        
        # Final layer norm
        total += self.ln_f_gamma.size
        total += self.ln_f_beta.size
        
        return total

def create_toy_dataset():
    """Create a simple dataset for testing"""
    # Simple vocabulary
    vocab = ['<pad>', '<end>', 'the', 'cat', 'sat', 'on', 'mat', 'dog', 'run', 'jump', 
             'quick', 'lazy', 'brown', 'fox', 'over', 'a', 'is', 'and', 'in', 'to']
    
    # Create token mappings
    token_to_id = {token: i for i, token in enumerate(vocab)}
    id_to_token = {i: token for i, token in enumerate(vocab)}
    
    # Simple sentences for training
    sentences = [
        "the cat sat on the mat",
        "the dog sat on the mat", 
        "the quick brown fox",
        "the lazy dog sat",
        "the fox jump over the dog"
    ]
    
    # Tokenize sentences
    tokenized = []
    for sent in sentences:
        tokens = sent.split()
        ids = [token_to_id.get(t, 0) for t in tokens]
        tokenized.append(ids)
    
    return vocab, token_to_id, id_to_token, tokenized

def run_experiments():
    print("Building Mini-GPT...")
    print("=" * 50)
    
    # Create toy dataset
    vocab, token_to_id, id_to_token, tokenized = create_toy_dataset()
    
    # Initialize model
    model = MiniGPT(
        vocab_size=len(vocab),
        d_model=64,
        n_heads=4,
        n_layers=2,
        seq_len=32
    )
    
    print(f"\nModel Configuration:")
    print(f"  Vocabulary size: {model.vocab_size}")
    print(f"  Model dimension: {model.d_model}")
    print(f"  Number of heads: {model.n_heads}")
    print(f"  Number of layers: {model.n_layers}")
    print(f"  Total parameters: ~{model.count_parameters() // 1000}K")
    
    # Test forward pass
    print("\n1. Testing forward pass...")
    test_input = np.array(tokenized[0][:5])  # "the cat sat on the"
    logits, attention_maps = model.forward(test_input.reshape(1, -1))
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {logits.shape}")
    
    # Visualize attention
    print("\n2. Visualizing attention patterns...")
    model.visualize_attention_patterns(test_input)
    print("   Saved: mini_gpt_attention.png")
    
    # Test generation
    print("\n3. Testing text generation...")
    prompt = "the cat"
    prompt_ids = [token_to_id.get(t, 0) for t in prompt.split()]
    
    print(f"   Prompt: '{prompt}'")
    
    for temp in [0.5, 0.8, 1.2]:
        generated_ids = model.generate(prompt_ids, max_length=15, temperature=temp)
        generated_text = ' '.join([id_to_token[id] for id in generated_ids])
        print(f"   Temp={temp}: '{generated_text}'")
    
    print("\n" + "=" * 50)
    print("MINI-GPT COMPLETE!")
    print("You've built a complete transformer language model from scratch!")
    print("\nNext steps:")
    print("- Add training loop with cross-entropy loss")
    print("- Implement gradient computation")
    print("- Train on larger dataset")
    print("- Add KV caching for faster generation")

if __name__ == "__main__":
    run_experiments()
