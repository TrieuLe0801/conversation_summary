import json

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "google/flan-t5-base"
# device = torch.device('cuda:0')
# original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
# tokenizer = AutoTokenizer.from_pretrained(model_name, device=device)


def zero_shot(model_name = "google/flan-t5-base"):
# Load the JSON data
    with open("./data/train.json", "r") as file:
        data = json.load(file)

    # Set the device you want to use
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create an instance of the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create an instance of the model with the specified device
    original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    # zero-shot
    prompt = f"""
    {data[0]["instruction"]}

    {data[0]["input"]}

    Summary:
    """

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors='pt').to(device)  # Move input tensors to the specified device

    # Generate the output
    output = original_model.generate(
        inputs["input_ids"],
        max_length=200,
        num_return_sequences=1,
    )[0]

    # Decode the generated output
    output = tokenizer.decode(output, skip_special_tokens=True)

    # Print the output and expected output
    print("Output:", output)
    print("Expectation:", data[0]["output"])

if __name__ == "__main__":
    zero_shot()
