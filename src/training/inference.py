def load_model(model_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    model_path = "path/to/your/trained/model"  # Update with your model path
    model, tokenizer = load_model(model_path)

    prompt = "Once upon a time"
    response = generate_response(model, tokenizer, prompt)
    print("Generated Response:", response)

if __name__ == "__main__":
    main()