# train_gpt2_wikipedia.py
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import torch

def preprocess_function(examples, tokenizer, max_length=512):
    """Tokenize the texts"""
    texts = [text + tokenizer.eos_token for text in examples["text"]]
    
    result = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None
    )
    
    return result

def main():
    print("=" * 80)
    print("TRAINING GPT-2 ON WIKIPEDIA")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    print("\nLoading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    print("\nAdding LoRA adapters...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['c_attn'],
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Use the new Parquet-based Wikipedia dataset
    print("\nLoading Wikipedia dataset...")
    print("(This will download ~20GB - first time only)")
    
    # Use wikimedia/wikipedia instead
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train"
    )
    
    print(f"Dataset size: {len(dataset):,} articles")
    
    # Optional: use subset for testing (comment out for full dataset)
    dataset = dataset.select(range(100000))  # First 100k articles
    print(f"Using subset: {len(dataset):,} articles for faster training")
    
    print("\nTokenizing dataset...")
    
    def tokenize(examples):
        return preprocess_function(examples, tokenizer, max_length=512)
    
    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    split_dataset = tokenized_dataset.train_test_split(test_size=0.01, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"\nTrain samples: {len(train_dataset):,}")
    print(f"Eval samples: {len(eval_dataset):,}")
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    training_args = TrainingArguments(
        output_dir="./gpt2-lora-wikipedia",
        num_train_epochs=3,  # 3 epochs on 100k articles
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=3e-4,
        weight_decay=0.01,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        warmup_steps=500,
        fp16=True,
        gradient_accumulation_steps=4,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print("Training on 100k Wikipedia articles (subset)")
    print("This will take approximately 2-3 hours")
    print("You can stop early with Ctrl+C")
    print("=" * 80 + "\n")
    
    trainer.train()
    
    print("\nSaving final model...")
    model.save_pretrained("./gpt2-lora-wikipedia-final")
    tokenizer.save_pretrained("./gpt2-lora-wikipedia-final")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Model saved to: ./gpt2-lora-wikipedia-final")
    print("\nThis model can now discuss Wikipedia topics!")

if __name__ == "__main__":
    main()
