import datasets
import pandas as pd
import uuid
import base64
import json
import os

from sklearn.model_selection import train_test_split

from .json2jsonl import json2jsonl


def format_dataset(dataset, image_folder):
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
                {'from': 'gpt', 'value': row['claude-3_5-sonnet-prediction.suggestion']}
            ]
        })

    return formatted_data


root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

ds = []
dataset_names = [
    'jenesys-ai/ark-lvlm-dataset',
    'jenesys-ai/ark-lvlm-ib-books-dataset',
    'jenesys-ai/ark-lvlm-uhy-dataset'
]

for dataset_name in dataset_names:
    dataset = datasets.load_dataset(dataset_name)
    dataset_path = f'{root_dir}/data/datasets/{dataset_name}'
    if not os.path.exists(dataset_path):
        dataset.save_to_disk(dataset_path)
    ds.append(dataset['train'])

df = pd.DataFrame(datasets.concatenate_datasets(ds))
complete = df[df['status'] == 'completed'][['id', 'document', 'instruction', 'claude-3_5-sonnet-prediction.suggestion']]
complete['instruction'] = complete['instruction'].apply(lambda x:
    x.replace(
        'The invoice image is provided here:',
        'The document image is provided here:'
    ).replace(
        'categorizing financial data from invoice images.',
        'categorizing financial data from document images.'
    ))
complete['base_64'] = complete['document'].apply(lambda x: x[x.find('base64,')+7:x.find('\'', x.find('base64,')+7)])
train, test = train_test_split(complete, test_size=0.22, shuffle=True, random_state=42)
print(f"Train size: {len(train)}", f"Test size: {len(test)}")

dataset_name = 'ark-lvlm-combined'
train_image_folder = f'{root_dir}/data/images/{dataset_name}/train'
test_image_folder = f'{root_dir}/data/images/{dataset_name}/test'

os.makedirs(train_image_folder, exist_ok=True)
os.makedirs(test_image_folder, exist_ok=True)

formatted_train_dataset = format_dataset(train, train_image_folder)
formatted_test_dataset = format_dataset(test, test_image_folder)

out_folder = f'{root_dir}/data/processed_whole/{dataset_name}'
train_out_filepath = f'{out_folder}/train.json'
test_out_filepath = f'{out_folder}/test.json'

os.makedirs(out_folder, exist_ok=True)

json.dump(formatted_train_dataset, open(train_out_filepath, 'w+'), indent=4)
json.dump(formatted_test_dataset, open(test_out_filepath, 'w+'), indent=4)

json2jsonl(train_out_filepath)
json2jsonl(test_out_filepath)
