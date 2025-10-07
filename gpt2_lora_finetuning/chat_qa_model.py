# chat_qa_model.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel
import torch

def ask_question(model, tokenizer, question, max_length=150):
    model.eval()
    
    # Format as the model was trained
    prompt = f"Question: {question}\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=len(inputs.input_ids[0]) + max_length,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the answer
    answer = response.split("Answer:")[-1].strip()
    
    # Stop at first newline or question
    if '\n' in answer:
        answer = answer.split('\n')[0].strip()
    
    return answer

def main():
    print("Loading Q&A model...\n")
    
    tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-lora-qa-final")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = PeftModel.from_pretrained(model, "./gpt2-lora-qa-final")
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 80)
    print("GPT-2 Q&A ASSISTANT")
    print("=" * 80)
    print("Ask me anything! Type 'quit' to exit")
    print("=" * 80 + "\n")
    
    while True:
        question = input("You: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not question:
            continue
        
        answer = ask_question(model, tokenizer, question)
        print(f"\nAssistant: {answer}\n")

if __name__ == "__main__":
    main()
