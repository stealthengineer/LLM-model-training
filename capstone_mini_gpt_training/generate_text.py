# generate_text.py
import pickle
import numpy as np

class TextGenerator:
    """Generate text using trained Mini-GPT"""
    
    def __init__(self, model_path='mini_gpt_trained.pkl'):
        print("Loading trained model...")
        with open(model_path, 'rb') as f:
            self.weights = pickle.load(f)
        
        self.config = self.weights['config']
        self.vocab_size = self.config['vocab_size']
        self.d_model = self.config['d_model']
        self.n_heads = self.config['n_heads']
        self.n_layers = self.config['n_layers']
        self.max_seq_len = self.config['max_seq_len']
        self.d_k = self.d_model // self.n_heads
        
        # Vocabulary (same as training)
        self.vocab = {
            0: '<pad>', 1: '<eos>',
            2: 'the', 3: 'cat', 4: 'sat', 5: 'on', 6: 'mat',
            7: 'dog', 8: 'ran', 9: 'quick', 10: 'brown',
            11: 'fox', 12: 'jumped', 13: 'over', 14: 'lazy',
            15: 'a', 16: 'is', 17: 'happy', 18: 'big', 19: 'small'
        }
        self.word_to_id = {v: k for k, v in self.vocab.items()}
        
        print(f"Model loaded successfully!")
        print(f"Parameters: {sum(w.size for w in self.weights.values() if hasattr(w, 'size')):,}")
    
    def forward(self, input_ids):
        """Run forward pass"""
        batch_size, seq_len = input_ids.shape if len(input_ids.shape) > 1 else (1, len(input_ids))
        
        if len(input_ids.shape) == 1:
            input_ids = input_ids.reshape(1, -1)
        
        # Embeddings
        x = self.weights['token_emb'][input_ids] + self.weights['pos_emb'][:seq_len]
        
        # Causal mask
        mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
        
        # Layers
        for layer in self.weights['layers']:
            x_norm = self._layer_norm(x, layer['ln1_gamma'], layer['ln1_beta'])
            x_attn = self._attention(x_norm, layer, mask)
            x = x + x_attn
            
            x_norm = self._layer_norm(x, layer['ln2_gamma'], layer['ln2_beta'])
            x_ffn = self._ffn(x_norm, layer)
            x = x + x_ffn
        
        # Output
        x = self._layer_norm(x, self.weights['ln_f_gamma'], self.weights['ln_f_beta'])
        logits = np.dot(x, self.weights['W_out']) + self.weights['b_out']
        
        return logits
    
    def _attention(self, x, layer, mask):
        """Multi-head attention"""
        batch_size, seq_len, _ = x.shape
        
        Q = np.dot(x, layer['W_q']).reshape(batch_size, seq_len, self.n_heads, self.d_k)
        K = np.dot(x, layer['W_k']).reshape(batch_size, seq_len, self.n_heads, self.d_k)
        V = np.dot(x, layer['W_v']).reshape(batch_size, seq_len, self.n_heads, self.d_k)
        
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k) + mask
        weights = self._softmax(scores)
        context = np.matmul(weights, V)
        
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        output = np.dot(context, layer['W_o'])
        
        return output
    
    def _ffn(self, x, layer):
        """Feed-forward network"""
        hidden = np.dot(x, layer['W_ff1']) + layer['b_ff1']
        hidden = self._gelu(hidden)
        output = np.dot(hidden, layer['W_ff2']) + layer['b_ff2']
        return output
    
    def _layer_norm(self, x, gamma, beta, eps=1e-5):
        """Layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma * x_norm + beta
    
    def _softmax(self, x):
        """Softmax"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _gelu(self, x):
        """GELU activation"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    def generate(self, prompt, max_length=20, temperature=1.0):
        """Generate text given a prompt"""
        # Convert prompt to tokens
        tokens = [self.word_to_id.get(word, 0) for word in prompt.lower().split()]
        
        print(f"\nPrompt: {prompt}")
        print(f"Generating (max {max_length} tokens)...\n")
        
        generated_tokens = tokens.copy()
        
        for _ in range(max_length):
            # Get logits for current sequence
            input_ids = np.array(generated_tokens[-self.max_seq_len:]).reshape(1, -1)
            logits = self.forward(input_ids)
            
            # Get logits for next token (last position)
            next_token_logits = logits[0, -1, :] / temperature
            
            # Sample from distribution
            probs = self._softmax(next_token_logits.reshape(1, -1)).flatten()
            next_token = np.random.choice(len(probs), p=probs)
            
            # Stop if EOS
            if next_token == 1:
                break
            
            generated_tokens.append(next_token)
        
        # Convert back to text
        generated_text = ' '.join([self.vocab.get(t, '<unk>') for t in generated_tokens])
        
        return generated_text

def demo_generation():
    """Demo text generation"""
    print("=" * 70)
    print("MINI-GPT TEXT GENERATION DEMO")
    print("=" * 70)
    
    generator = TextGenerator('mini_gpt_trained.pkl')
    
    print("\n" + "=" * 70)
    print("VOCABULARY:")
    print("-" * 70)
    words = [generator.vocab[i] for i in range(len(generator.vocab)) if i > 1]
    print(', '.join(words))
    print("=" * 70)
    
    # Test prompts
    prompts = [
        "the cat",
        "a dog",
        "the quick brown",
        "a big",
    ]
    
    print("\nGENERATION EXAMPLES:")
    print("=" * 70)
    
    for prompt in prompts:
        generated = generator.generate(prompt, max_length=15, temperature=0.8)
        print(f"\n→ Input:  '{prompt}'")
        print(f"  Output: '{generated}'")
        print("-" * 70)
    
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    print("Enter a prompt (or 'quit' to exit)")
    print("Vocabulary: the, cat, sat, on, mat, dog, ran, quick, brown,")
    print("           fox, jumped, over, lazy, a, is, happy, big, small")
    print("=" * 70)
    
    while True:
        prompt = input("\nPrompt: ").strip()
        
        if prompt.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not prompt:
            continue
        
        try:
            generated = generator.generate(prompt, max_length=20, temperature=0.8)
            print(f"Generated: {generated}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    demo_generation()
