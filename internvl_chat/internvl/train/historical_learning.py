from internvl_chat.tools.huggingface import prepare_dataset_for_finetuning

if __name__ == '__main__':
    dataset_names = ['jenesys-ai/historical_data_9dbed789-a3d4-4d69-95f5-7ff3fcbdbfb8']
    combined_dataset_name = dataset_names[0]
    label_column = 'lvlm-historical-annotation.suggestion'

    prepare_dataset_for_finetuning(dataset_names, combined_dataset_name, label_column)
