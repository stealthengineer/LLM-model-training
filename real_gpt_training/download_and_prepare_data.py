# download_and_prepare_data.py
import requests
import pickle
import numpy as np
from collections import Counter

def download_shakespeare():
    """Download Shakespeare complete works"""
    print("Downloading Shakespeare dataset...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    
    response = requests.get(url)
    text = response.text
    
    print(f"Downloaded {len(text):,} characters")
    print(f"First 200 characters:\n{text[:200]}")
    
    # Save raw text
    with open('shakespeare.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    
    return text

def build_tokenizer(text):
    """Build character-level tokenizer"""
    print("\nBuilding tokenizer...")
    
    # Get all unique characters
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    print(f"Vocabulary size: {vocab_size} characters")
    print(f"Characters: {''.join(chars[:50])}...")
    
    # Create mappings
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # Save tokenizer
    tokenizer = {
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'vocab_size': vocab_size,
        'chars': chars
    }
    
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    print("Tokenizer saved to tokenizer.pkl")
    
    return tokenizer

def create_training_data(text, tokenizer, seq_len=128):
    """Convert text to training sequences"""
    print(f"\nCreating training sequences (seq_len={seq_len})...")
    
    # Encode entire text
    data = [tokenizer['char_to_idx'][ch] for ch in text]
    
    # Create sequences
    sequences = []
    for i in range(0, len(data) - seq_len, seq_len // 2):  # 50% overlap
        sequences.append(data[i:i + seq_len])
    
    sequences = np.array(sequences, dtype=np.int32)
    
    print(f"Created {len(sequences):,} sequences")
    print(f"Shape: {sequences.shape}")
    
    # Train/val split (90/10)
    split_idx = int(0.9 * len(sequences))
    train_data = sequences[:split_idx]
    val_data = sequences[split_idx:]
    
    print(f"Train: {len(train_data):,} sequences")
    print(f"Val: {len(val_data):,} sequences")
    
    # Save
    np.save('train_data.npy', train_data)
    np.save('val_data.npy', val_data)
    
    print("Data saved to train_data.npy and val_data.npy")
    
    return train_data, val_data

def main():
    print("=" * 70)
    print("REAL GPT TRAINING - DATA PREPARATION")
    print("=" * 70)
    
    # Download
    text = download_shakespeare()
    
    # Build tokenizer
    tokenizer = build_tokenizer(text)
    
    # Create training data
    train_data, val_data = create_training_data(text, tokenizer, seq_len=128)
    
    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 70)
    print("\nFiles created:")
    print("  - shakespeare.txt (raw text)")
    print("  - tokenizer.pkl (character mappings)")
    print("  - train_data.npy (training sequences)")
    print("  - val_data.npy (validation sequences)")
    print("\nReady for training!")

if __name__ == "__main__":
    main()
