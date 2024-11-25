import matplotlib.pyplot as plt
import os
import json


class SaveResults:

    def __init__(self, config):
        self.config = config

    def plot_learning_curve(self, log_history):

        steps = sorted(set(log["step"] for log in log_history if "step" in log))
        losses = [log["loss"] for log in log_history if "loss" in log]
        val_losses = [log["eval_loss"] for log in log_history if "eval_loss" in log]
        # Truncate to the same length
        min_len = min(len(steps), len(losses))
        print(f"length of steps {len(steps)}, length of loss {len(losses)}")
        plt.figure(figsize=(12, 6))
        plt.plot(steps[:min_len], losses[:min_len], label="Training Loss")
        plt.plot(steps[:min_len], val_losses[:min_len], label="Validation Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Loss over Steps")
        plt.savefig(
            os.path.join(self.config.results_dir, "loss_curves.png"), format="png"
        )
        plt.show()

    def save_metadata(self, tokenizer_name, training_args, metrics):
        metadata = {
            "model_name": self.config.model,
            "tokenizer_name": tokenizer_name,
            "training_args": training_args.to_dict(),
            "metrics": metrics,
        }
        with open(os.path.join(self.config.results_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

    def print_conversation(self, tokenized_datasets, tokenizer,num_dial=20):
        # # Visualize the input and output for the first few dialogues
        for i in range(num_dial):  # Visualize 5 examples
            sample = tokenized_datasets["test"][i]
            decoded_input = tokenizer.decode(
                sample["input_ids"], skip_special_tokens=False
            )
            decoded_labels = tokenizer.decode(
                sample["labels"], skip_special_tokens=False
            )

            print(f"\nDialogue {i + 1}:")
            print("\nInput Sequence:")
            print(decoded_input)
            # print("\nLabel Sequence (Target Output):")
            # print(decoded_labels)
            print("-" * 80)
