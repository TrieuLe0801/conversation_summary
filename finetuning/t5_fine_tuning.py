import time

import torch
from datasets.load import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from processing_dataset import tokenize_function
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from utils.utils import print_number_of_trainable_model_parameters


# Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")

# Set the device
device = "cpu"

if torch.backends.mps.is_available():
    # Initialize the device
    device = "mps"
elif torch.cuda.is_available():
    # Initialize the device
    device = "cuda:1"

print(f"Using device: {device}")

# Load the model and tokenizer
model_name = "google/flan-t5-base"
original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32).to(
    device
)  # If not using mac, use bfloat16
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(print_number_of_trainable_model_parameters(original_model))

lora_config = LoraConfig(
    r=32,  # Rank
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,  # FLAN-T5
)
peft_model = get_peft_model(original_model, lora_config)
print(print_number_of_trainable_model_parameters(peft_model))

if __name__ == "__main__":
    dataset = load_dataset("knkarthick/dialogsum")
    tokenized_datasets = dataset.map(
        lambda example: tokenize_function(example, tokenizer),
        batched=True,
    )
    tokenized_datasets = tokenized_datasets.remove_columns(
        [
            "id",
            "topic",
            "dialogue",
            "summary",
        ]
    )
    print(tokenized_datasets)
    tokenized_datasets = tokenized_datasets.filter(
        lambda example, index: index % 100 == 0, with_indices=True
    )
    print(f"Shapes of the datasets:")
    print(f"Training: {tokenized_datasets['train'].shape}")
    print(f"Validation: {tokenized_datasets['validation'].shape}")
    print(f"Test: {tokenized_datasets['test'].shape}")

    print(tokenized_datasets)
    # output_dir = f'./models/dialogue-summary-training-{str(int(time.time()))}'

    # training_args = TrainingArguments(
    #     output_dir=output_dir,
    #     learning_rate=1e-5,
    #     num_train_epochs=1,
    #     weight_decay=0.01,
    #     logging_steps=1,
    #     max_steps=1
    # )

    # trainer = Trainer(
    #     model=original_model,
    #     args=training_args,
    #     train_dataset=tokenized_datasets['train'],
    #     eval_dataset=tokenized_datasets['validation']
    # )

    # trainer.train()

    output_dir = f"./models/peft-dialogue-summary-training-{str(int(time.time()))}"

    peft_training_args = TrainingArguments(
        output_dir=output_dir,
        auto_find_batch_size=True,
        learning_rate=1e-3,  # Higher learning rate than full fine-tuning.
        num_train_epochs=1,
        logging_steps=1,
        max_steps=1,
    )

    peft_trainer = Trainer(
        model=peft_model,
        args=peft_training_args,
        train_dataset=tokenized_datasets["train"],
    )

    peft_trainer.train()
