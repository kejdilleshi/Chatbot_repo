import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

# Load the saved model and tokenizer
model_path = "./saved_new_modelT5"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
model.to(device)


# Function to generate a response
def generate_response(input_text):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=150, num_beams=5, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# Interactive loop for conversation
print("Chatbot is ready! Type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    response = generate_response(user_input)
    print(f"Chatbot: {response}")
