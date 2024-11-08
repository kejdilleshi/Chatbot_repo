import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import load_dataset, load_metric

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {device}")

# Load the dataset
dataset = load_dataset("multi_woz_v22")


# Preprocess the dataset
tokenizer = AutoTokenizer.from_pretrained("t5-small")


def preprocess_function(examples):
    inputs = []
    targets = []
    for dialogue in examples["turns"]:
        context = ""
        for turn_id, speaker, utterance in zip(
            dialogue["turn_id"], dialogue["speaker"], dialogue["utterance"]
        ):
            if speaker == 0:  # Assuming 0 indicates the user
                context += " " + utterance
            elif speaker == 1:  # Assuming 1 indicates the system
                if context.strip():  # Ensure context is not empty
                    inputs.append(context.strip())
                    targets.append(utterance)
                    context = ""  # Reset context after system response

    # Debug: Print lengths of inputs and targets
    print(f"Processed {len(inputs)} inputs and {len(targets)} targets")

    if len(inputs) != len(targets):
        print(
            f"Warning: Inputs and targets length mismatch. Inputs: {len(inputs)}, Targets: {len(targets)}"
        )

    if len(inputs) == 0 or len(targets) == 0:
        return {"input_ids": [], "labels": []}

    model_inputs = tokenizer(
        inputs, max_length=512, truncation=True, padding="max_length"
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, max_length=128, truncation=True, padding="max_length"
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Map the preprocess function to the dataset
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=["dialogue_id", "services", "turns"],
)


# Filter out empty examples
def filter_empty(examples):
    return len(examples["input_ids"]) > 0 and len(examples["labels"]) > 0


tokenized_datasets = tokenized_datasets.filter(filter_empty)

# Load the model
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
model.to(device)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=10,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
)

# Define metric
metric = load_metric("rouge")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Replace newline tokens with spaces to avoid formatting issues
    decoded_preds = ["\n".join(pred.strip().split()) for pred in decoded_preds]
    decoded_labels = ["\n".join(label.strip().split()) for label in decoded_labels]

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )

    # Process results for ROUGE metrics which can return tuples
    result = {
        key: value.mid.fmeasure if isinstance(value, tuple) else value
        for key, value in result.items()
    }
    result = {k: round(v * 100, 4) for k, v in result.items()}
    return result


# Initialize the Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()


# Save the model
model.save_pretrained("./saved_new_modelT5")
tokenizer.save_pretrained("./saved_new_modelT5")
print("Model saved successfully!")

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")
