import json
import os

from datasets import load_dataset
from tqdm import tqdm


def create_conversation_dataset_hf(
    dataset_name_huggingface: str = "knkarthick/dialogsum",
    dataset_type: str = "train",
    dir_path: str = "",
):
    """Collect data from huggingface

    Args:
        dataset_name_huggingface (str): Huggingface dataset
        dataset_type (str): data type for return (train|test|validation)
        dir_path (str): directory to saving
    """
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, 511)

    dataset = load_dataset(dataset_name_huggingface).get(dataset_type)
    print(dataset)

    data_list = []
    # save data
    for d in tqdm(dataset, position=0, total=dataset.num_rows):
        dictionary = {
            "instruction": "Summarize the conversation of the persons and keep the topic word of this conversation",
            "input": d.get("dialogue", ""),
            "output": d.get("summary", ""),
            "topic": d.get("topic", ""),
        }
        data_list.append(dictionary)
    with open(
        os.path.join(dir_path, f"{dataset_type}.json"), "w"
    ) as json_file:
        json.dump(data_list, json_file, indent=4, ensure_ascii=False)
    return data_list


if __name__ == "__main__":
    create_conversation_dataset_hf(dir_path="data")
