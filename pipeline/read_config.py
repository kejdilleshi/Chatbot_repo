import argparse


def str2bool(v):
    """makes a string non-case-sensitive"""
    return v.lower() in ("true", "1")


def get_args():
    """
    read configuration file
    """
    parser = argparse.ArgumentParser(description="Transformer chatbot")

    parser.add_argument("--model", type=str, default="gpt2", help="Train a new model")

    parser.add_argument(
        "--train_new", type=str2bool, default=True, help="Train a new model"
    )
    parser.add_argument(
        "--save_lr", type=str2bool, default=True, help="Plot the learning rate"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Path of the results folder",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="batch size",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="gradient_accumulation_steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="learning_rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="weight_decay",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Seed for reproducibility"
    )

    # Parse the arguments
    args = parser.parse_args()

    return args
