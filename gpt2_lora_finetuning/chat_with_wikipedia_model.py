# chat_with_wikipedia_model.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel
import torch

def generate_response(model, tokenizer, prompt, max_length=150):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=len(inputs.input_ids[0]) + max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the response
    response = response[len(prompt):].strip()
    return response

def main():
    print("Loading Wikipedia model...\n")
    
    tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-lora-wikipedia-final")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = PeftModel.from_pretrained(model, "./gpt2-lora-wikipedia-final")
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 80)
    print("WIKIPEDIA Q&A CHATBOT")
    print("=" * 80)
    print("Ask questions or type 'quit' to exit")
    print("=" * 80 + "\n")
    
    while True:
        question = input("You: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not question:
            continue
        
        # Format as a completion prompt
        prompt = f"Q: {question}\nA:"
        
        print("\nModel: ", end="")
        response = generate_response(model, tokenizer, prompt, max_length=150)
        
        # Clean up the response
        # Stop at question marks or newlines that indicate end of answer
        if '\n' in response:
            response = response.split('\n')[0]
        
        print(response + "\n")

if __name__ == "__main__":
    main()
