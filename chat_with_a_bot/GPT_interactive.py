import torch
from transformers import GPT2LMHeadModel, AutoTokenizer

# Load the trained model and tokenizer
model_path="../ensemble_results/SamBot_1-1-3"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Make sure the model is in evaluation mode
model.eval()

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def chat_with_bot(input_text, previous_chat,count):
    # Encode the input text based on whether it's the first message or not
    if count == 0:
        input_text = f"<|user|> {input_text}<|system|>"
    else:
        input_text = f"{previous_chat}<|user|> {input_text}<|system|>"
    
    # print("INPUT: ",count, input_text)
    # Encode the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)


    output = model.generate(
        input_ids,
        max_new_tokens=30,  # Limit to 30 new tokens
        num_return_sequences=1,
        no_repeat_ngram_size=1,
        pad_token_id=tokenizer.convert_tokens_to_ids("<|pad|>"),
        # temperature=0.1,    # Control randomness
        # top_p=0.9,          # Limit to top 80% most likely words
        # length_penalty=0.5, # Encourage shorter responses
        # early_stopping=True,# Stop if end-of-sequence is generated
        do_sample=False      # Use sampling for more natural responses
    )




    # Decode the output, skipping special tokens
    response = tokenizer.decode(
        output[:, input_ids.shape[-1] :][0], skip_special_tokens=True
    )
    return response, input_text + response


# Chat loop
# print(tokenizer.special_tokens_map)

print("Start chatting with the bot! Type 'exit' to stop.")
previous_chat = ""
count = 0
while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Exiting chat...")
        break

    bot_response, previous_chat = chat_with_bot(user_input, previous_chat,count)
    print(f"Bot: {bot_response}")
    count += 1
