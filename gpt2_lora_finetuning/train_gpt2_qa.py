# train_gpt2_qa.py
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

def format_instruction(example):
    """Format examples as Q&A pairs"""
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output = example.get('output', '')
    
    if input_text:
        text = f"Question: {instruction}\nContext: {input_text}\nAnswer: {output}"
    else:
        text = f"Question: {instruction}\nAnswer: {output}"
    
    return {"text": text}

def main():
    print("=" * 80)
    print("TRAINING GPT-2 FOR Q&A")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    print("\nLoading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    print("\nAdding LoRA...")
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
    
    print("\nLoading instruction dataset...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    
    print(f"Dataset size: {len(dataset):,} instructions")
    
    dataset = dataset.select(range(20000))
    print(f"Using: {len(dataset):,} instructions")
    
    print("\nFormatting as Q&A pairs...")
    dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)
    
    # Fixed tokenization with proper padding
    def tokenize(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",  # FIX: Pad to max_length
        )
        # Set labels (use -100 for padding tokens to ignore in loss)
        result["labels"] = [
            [(token if token != tokenizer.pad_token_id else -100) for token in ids]
            for ids in result["input_ids"]
        ]
        return result
    
    print("Tokenizing...")
    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing"
    )
    
    split = tokenized.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    
    print(f"\nTrain: {len(train_dataset):,}")
    print(f"Eval: {len(eval_dataset):,}")
    
    # Simple data collator (no additional padding needed)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=None
    )
    
    training_args = TrainingArguments(
        output_dir="./gpt2-lora-qa",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=3e-4,
        weight_decay=0.01,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        warmup_steps=500,
        fp16=True,
        gradient_accumulation_steps=8,
        save_total_limit=2,
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
    print("TRAINING")
    print("=" * 80)
    print("Training on 20k instruction examples")
    print("This will take about 1-2 hours")
    print("=" * 80 + "\n")
    
    trainer.train()
    
    print("\nSaving model...")
    model.save_pretrained("./gpt2-lora-qa-final")
    tokenizer.save_pretrained("./gpt2-lora-qa-final")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print("Model saved to: ./gpt2-lora-qa-final")

if __name__ == "__main__":
    main()
