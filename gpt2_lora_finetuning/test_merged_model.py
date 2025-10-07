# test_merged_model.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel
import torch

def ask(model, tokenizer, question):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_length=len(inputs.input_ids[0]) + 100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.split("Answer:")[-1].strip().split('\n')[0]
    return answer

print("Loading merged model...\n")

tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-lora-merged")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model = PeftModel.from_pretrained(model, "./gpt2-lora-merged")
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

questions = [
    "What is Python?",
    "Explain the theory of relativity",
    "How does machine learning work?",
    "What is the capital of France?"
]

print("Testing merged model:\n")
for q in questions:
    print(f"Q: {q}")
    print(f"A: {ask(model, tokenizer, q)}\n")
