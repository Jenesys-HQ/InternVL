from internvl.tools.huggingface import prepare_dataset_for_finetuning

if __name__ == '__main__':
    dataset_names = [
        'jenesys-ai/ark-lvlm-dataset',
        'jenesys-ai/ark-lvlm-ib-books-dataset',
        'jenesys-ai/ark-lvlm-uhy-dataset'
    ]
    combined_dataset_name = 'jenesys-ai/ark-lvlm-combined'
    label_column = 'claude-3_5-sonnet-prediction.suggestion'

    prepare_dataset_for_finetuning(dataset_names, combined_dataset_name, label_column)
