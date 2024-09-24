import base64
import json
import os
from typing import Any, Dict, List
import uuid

import datasets
import pandas as pd
from sklearn.model_selection import train_test_split

from internvl.tools.json2jsonl import json2jsonl


def format_dataset(dataset: pd.DataFrame, image_folder: str, label_column: str) -> List[Dict[str, Any]]:
    formatted_data = []
    for i, row in dataset.iterrows():
        id = str(uuid.uuid4())
        image_data = base64.b64decode(row['base_64'])
        image_path = f'{image_folder}/{id}.png'

        with open(image_path, 'wb') as file:
            file.write(image_data)

        formatted_data.append({
            'id': id,
            'image': image_path,
            'conversations': [
                {'from': 'human', 'value': row['instruction']},
                {'from': 'gpt', 'value': row[label_column]}
            ]
        })

    return formatted_data


def prepare_dataset_for_finetuning(dataset_names: List[str], combined_dataset_name: str, label_column: str):
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')

    ds = []
    for dataset_name in dataset_names:
        dataset = datasets.load_dataset(dataset_name)
        dataset_path = f'{root_dir}/data/datasets/{dataset_name}'
        if not os.path.exists(dataset_path):
            dataset.save_to_disk(dataset_path)
        ds.append(dataset['train'])
    df = pd.DataFrame(datasets.concatenate_datasets(ds))
    # TODO restore once the annotation has been completed
    complete = df.loc[:, ['id', 'document', 'instruction', label_column]]
    # complete = df.loc[df['status'] == 'completed', ['id', 'document', 'instruction', label_column]]

    if complete.shape[0] == 0:
        raise ValueError('No completed annotations found')

    complete.loc[:, 'base_64'] = complete['document'].apply(
        lambda x: x[x.find('base64,') + 7:x.find('\'', x.find('base64,') + 7)])

    train, test = train_test_split(complete, test_size=.2, shuffle=True, random_state=42)
    split_dataset = {'train': train, 'test': test}

    processed_folder = f'{root_dir}/data/processed_whole/{combined_dataset_name}'
    os.makedirs(processed_folder, exist_ok=True)

    for name in ['train', 'test']:
        subset = split_dataset[name]
        print(f"{name} size: {len(subset)}")

        image_folder = f'{root_dir}/data/images/{combined_dataset_name}/{name}'
        os.makedirs(image_folder, exist_ok=True)
        formatted_subset = format_dataset(subset, image_folder, label_column)

        out_filepath = f'{processed_folder}/{name}.json'
        json.dump(formatted_subset, open(out_filepath, 'w+'), indent=4)
        json2jsonl(out_filepath)

        if name == 'train':
            dataset_description_filepath = os.path.join(
                root_dir, 'shell', 'data', f'{combined_dataset_name.replace("/", "-")}_train.json')
            dataset_description = {
                f'{combined_dataset_name}-train': {
                    "root": "data/images/",
                    "annotation": f'data/processed_whole/{combined_dataset_name}/train.jsonl',
                    "data_augment": False,
                    "repeat_time": 1,
                    "length": len(formatted_subset)
                }
            }
            json.dump(dataset_description, open(dataset_description_filepath, 'w+'), indent=4)
