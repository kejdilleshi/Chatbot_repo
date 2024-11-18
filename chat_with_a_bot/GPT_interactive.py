import torch
from transformers import GPT2LMHeadModel, AutoTokenizer

# Load the trained model and tokenizer
model_path="../ensemble_results/ChatBot_75-25"
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
        input_text = f"{previous_chat}<|endofturn|><|user|> {input_text}<|system|>"
    
    print(count, input_text)
    # Encode the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)


    output = model.generate(
        input_ids,
        # max_length= 256,
        max_new_tokens=30,  
        num_return_sequences=1,
        # no_repeat_ngram_size=2,
        pad_token_id=tokenizer.convert_tokens_to_ids("<|pad|>"),
        # temperature=0.7,  # Controls randomness
        # top_p=0.9,        # Limits to more likely words
        do_sample=False,   # Enables controlled sampling for more variety
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
