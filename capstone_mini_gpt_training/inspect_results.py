# inspect_results.py
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def check_saved_files():
    """Check what files exist from training"""
    print("Checking for saved files...")
    print("=" * 60)
    
    files_found = []
    
    # Check for model
    if os.path.exists('mini_gpt_trained.pkl'):
        size = os.path.getsize('mini_gpt_trained.pkl')
        print(f"✓ Model found: mini_gpt_trained.pkl ({size:,} bytes)")
        files_found.append('model')
    else:
        print("✗ Model NOT found: mini_gpt_trained.pkl")
    
    # Check for visualizations
    viz_files = [
        'training_complete.png',
        'trained_model_results.png'
    ]
    
    for viz in viz_files:
        if os.path.exists(viz):
            print(f"✓ Visualization found: {viz}")
            files_found.append(viz)
        else:
            print(f"✗ Visualization NOT found: {viz}")
    
    print("=" * 60)
    return files_found

def load_and_inspect_model():
    """Load and inspect the saved model"""
    if not os.path.exists('mini_gpt_trained.pkl'):
        print("\nNo saved model found. Training did not complete.")
        return None
    
    print("\nLoading model...")
    with open('mini_gpt_trained.pkl', 'rb') as f:
        weights = pickle.load(f)
    
    config = weights['config']
    
    print("\nMODEL CONFIGURATION:")
    print("-" * 60)
    for key, value in config.items():
        print(f"{key:20s}: {value}")
    
    print("\nPARAMETER SHAPES:")
    print("-" * 60)
    print(f"Token embeddings: {weights['token_emb'].shape}")
    print(f"Position embeddings: {weights['pos_emb'].shape}")
    print(f"Number of layers: {len(weights['layers'])}")
    print(f"Output weights: {weights['W_out'].shape}")
    
    # Count total parameters
    total_params = weights['token_emb'].size
    total_params += weights['pos_emb'].size
    for layer in weights['layers']:
        for param in layer.values():
            total_params += param.size
    total_params += weights['W_out'].size
    total_params += weights['b_out'].size
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Model size: {(total_params * 4) / (1024**2):.2f} MB (FP32)")
    
    return weights

def recreate_model_from_weights(weights):
    """Recreate the model architecture from saved weights"""
    print("\nModel successfully loaded!")
    print("You can use this model for:")
    print("  1. Generate text (though vocabulary is limited)")
    print("  2. Continue training with more data")
    print("  3. Fine-tune on new tasks")
    print("  4. Analyze learned representations")
    
    return True

if __name__ == "__main__":
    print("MINI-GPT MODEL INSPECTOR")
    print("=" * 60)
    
    # Check files
    files = check_saved_files()
    
    # Load model if it exists
    if 'model' in files:
        weights = load_and_inspect_model()
        if weights:
            recreate_model_from_weights(weights)
    else:
        print("\n" + "=" * 60)
        print("TRAINING DID NOT COMPLETE")
        print("=" * 60)
        print("\nOptions:")
        print("1. Re-run training (will take 2-3 hours)")
        print("2. Use a faster training script (30 minutes)")
        print("3. Use PyTorch for automatic differentiation (seconds)")
        print("\nWould you like me to provide a faster training option?")
