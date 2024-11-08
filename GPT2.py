import torch
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    logging,
)
from pynvml import nvmlDeviceGetHandleByIndex, nvmlInit, nvmlDeviceGetMemoryInfo

from datasets import load_dataset
# my modules 
from pipeline import SaveResults, get_args, set_seed


args = get_args()
# Set the seed for reproducibility
set_seed(args.seed)

# Save results if necessary
SR = SaveResults(args)

# File directory for samuel's blog data
#  
file_dir = 'Data/samuel_data.txt'

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


# ==============

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {device}")

# Load the dataset
dataset = load_dataset("multi_woz_v22", trust_remote_code=True)

# Preprocess the dataset
tokenizer = AutoTokenizer.from_pretrained(args.model)
# tokenizer = AutoTokenizer.from_pretrained("Data/SamBOT-gpt2")


special_tokens_dict = {
    "additional_special_tokens": ["<|user|>", "<|system|>", "<|endofturn|>"],
    "pad_token": "<|pad|>",
}
tokenizer.add_special_tokens(special_tokens_dict)


def extract_singleturn(dialogue):
    dialogues=[]
    for i in range(0, len(dialogue["utterance"]) - 1, 2):  # Each user-bot exchange is 2 turns
        if dialogue["speaker"][i] == 0 and dialogue["speaker"][i + 1] == 1:
            user_utterance = dialogue["utterance"][i].strip()
            bot_response = dialogue["utterance"][i + 1].strip()
            single_turn = f"<|user|> {user_utterance} <|system|> {bot_response} <|endofturn|>"
            dialogues.append(single_turn)
    return dialogues

def extract_multiturn(dialogue,n_turn=4):
    context = ""  # Persistent context across turns
    for id, speaker, utterance in zip(dialogue['turn_id'],dialogue["speaker"], dialogue["utterance"]):
        if int(id) < n_turn :
            if speaker == 0:  # User input
                context += f"<|user|> {utterance.strip()} "
            elif speaker == 1:  # System response
                if context.strip():  # Ensure there's context to process
                    context += f"<|system|> {utterance.strip()} <|endofturn|>"  # Append system response to context Add endo of text token
                    
    return context

def extract_text_segment(tokens, max_length, n, tokenizer):
    start_idx = (n - 1) * max_length
    end_idx = start_idx + max_length

    segment_tokens = tokens[start_idx:end_idx]

    # Decode back to text
    segment_text = tokenizer.decode(segment_tokens, skip_special_tokens=True)

    # Pad if necessary
    if len(segment_tokens) < max_length:
        pad_length = max_length - len(segment_tokens)
        pad_token_id = tokenizer.convert_tokens_to_ids('<|pad|>')
        segment_tokens.extend([pad_token_id] * pad_length)
        segment_text = tokenizer.decode(segment_tokens, skip_special_tokens=True)
    return segment_text

def preprocess_data_balanced(examples):

    dialogues = []
    block=1
    # Read the text file once
    with open(file_dir, 'r', encoding='utf-8') as file:
        full_text = file.read()

    # Tokenize the text once
    tokens = tokenizer(full_text)['input_ids']
    for dialogue in examples['turns']:
        single_list=extract_singleturn(dialogue)
        dialogues.extend(single_list)
        if block%4==0:
            multi_list=extract_multiturn(dialogue,n_turn=4)
            dialogues.append(multi_list)
            # sam_text = extract_text_segment(tokens, 256, block, tokenizer)
            # dialogues.append(sam_text)
        # if block%3==0:
        #     multi_list=extract_multiturn(dialogue,n_turn=8)
        #     dialogues.append(multi_list)
            
        block+=1
   
    tokenized_inputs = tokenizer(
        dialogues, padding="max_length", truncation=True, max_length=256
    )

    # Add labels as a copy of input_ids
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()

    return tokenized_inputs


# Map the preprocess function to the dataset
tokenized_datasets = dataset.map(
    preprocess_data_balanced,
    batched=True,
    remove_columns=["dialogue_id", "services", "turns"],
)


# Filter out empty examples
def filter_empty(examples):
    return len(examples["input_ids"]) > 0 and len(examples["labels"]) > 0


tokenized_datasets = tokenized_datasets.filter(filter_empty)

# Visualize the input and output for the first few dialogues
# SR.print_conversation(tokenized_datasets, tokenizer)
# exit()

# Load the model
model = GPT2LMHeadModel.from_pretrained(args.model)
# model = GPT2LMHeadModel.from_pretrained("Data/SamBOT-gpt2")
model.resize_token_embeddings(len(tokenizer))
model.to(device)


# -------------------------------

logging.set_verbosity_error()


training_args = TrainingArguments(
    output_dir=args.results_dir,
    max_steps=5000,
    optim="adamw_torch",
    learning_rate=args.learning_rate,
    weight_decay=args.weight_decay,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    gradient_checkpointing=True,
    lr_scheduler_type="linear",
    evaluation_strategy="steps",
    eval_steps=50,
    logging_steps=50,
    warmup_steps=100,
    disable_tqdm=False,
    log_level="info",
    fp16=True,
    report_to="none",  # This ensures no reporting to any integrations
) 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)
result = trainer.train()
metrics = trainer.evaluate()
log_history = trainer.state.log_history


# =======================

# Save the model
model.save_pretrained(args.results_dir)
tokenizer.save_pretrained(args.results_dir)

print("Model saved successfully!")

# Plot Learning curve
SR.plot_learning_curve(log_history)
SR.save_metadata(tokenizer.name_or_path, training_args, metrics)
