import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

# Load the trained model and tokenizer
model_dir = "../ensemble_results/ChatBot_1-1"  # Replace with the path to your trained model directory
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)

# Set device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# Special tokens
user_token = "<|user|>"
system_token = "<|system|>"
endofturn_token = "<|endofturn|>"

def chat_with_model(model, tokenizer, context="", max_length=256):
    """
    Generates a response from the model based on the given context.
    
    Args:
        model: The GPT-2 model.
        tokenizer: The tokenizer for the model.
        context: The conversation history or context (as a string).
        max_length: The maximum length of the generated sequence.
    
    Returns:
        A tuple of (updated_context, model_response).
    """
    # Tokenize the input context
    inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    # Generate the response
    output = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids(endofturn_token),
        top_p=0.9,
        temperature=0.7,
    )

    # Decode the generated tokens to text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
    
    # Extract the new system response
    system_response_start = generated_text.find(system_token) + len(system_token)
    system_response_end = generated_text.find(endofturn_token, system_response_start)
    system_response = generated_text[system_response_start:system_response_end].strip()

    # Update context with the system response
    updated_context = context + f"{system_token} {system_response} {endofturn_token}"
    return updated_context, system_response

# Start a chat session
print("Chat with your model! Type 'exit' to end the chat.")
context = ""

while True:
    # Get user input
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Exiting the chat. Goodbye!")
        break
    
    # Add user input to the context
    context += f"{user_token} {user_input.strip()} "
    
    # Get the model's response
    context, response = chat_with_model(model, tokenizer, context)
    
    # Print the system response
    print(f"Bot: {response}")
