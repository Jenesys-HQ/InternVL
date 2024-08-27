import argparse
import json
import logging
from typing import Any, Dict

import mlflow
import torch
from dotenv import load_dotenv

from constants import PROMPT
from data_utils import extract_invoice_data, load_labelbox_data, transform_invoice_data, flatten_data, extract_json_data
from img_utils import get_pdf_base64_from_img_url, pdf_to_image_base64_function, load_image_bs64, \
    pdfs_to_images_base64_function, load_image
from internvl.model import load_model_and_tokenizer
from standardisation import standardise_data_models, standardise_data_value
from metrics import MetricsHelper

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

EXPERIMENT_NAME = 'Invoice Extraction'

def evaluate_by_item(model_path: str, gen_key: str, project_id: str):
    export_data = load_labelbox_data(gen_key, project_id)
    extracted_labeled_data = extract_invoice_data(export_data, project_id)
    transformed_labeled_data = [transform_invoice_data(data) for data in extracted_labeled_data]
    # remove doc transcript from labelbox transformed data
    transformed_labeled_data = [{k: v for k, v in record.items() if k != 'Doc Transcript'}
                                for record in transformed_labeled_data]

    flattened_labeled_data = []
    for data_row in transformed_labeled_data:
        flattened_row_data = {}
        flatten_data(data_row, flattened_row_data)
        if len(flattened_row_data) == 0:
            continue
        flattened_labeled_data.append(flattened_row_data)

    args = argparse.Namespace(
        checkpoint=model_path,
        root='./Your_Results',
        num_beams=1,
        top_k=50,
        top_p=0.9,
        sample=True,
        dynamic=False,
        max_num=6,
        load_in_8bit=False,
        load_in_4bit=False,
        auto=False,
    )

    model, tokenizer = load_model_and_tokenizer(args)

    standardised_labeled_data = []
    standardised_predicted_data = []
    for data_row in flattened_labeled_data:
        logger.info(data_row['Img_path'])
        df_base64_string = get_pdf_base64_from_img_url(data_row["Img_path"])
        images_base64 = pdf_to_image_base64_function(df_base64_string)

        standardised_labeled_data_row = {}
        standardised_predicted_data_row = {}
        for label_name, label_value in data_row.items():
            if label_name == 'Img_path':
                continue

            prompt = (f"Extract a value form this invoice: {label_name}.\n"
                      f"The response should just contain the value.")
            predicted_value = model.generate_value(prompt, images_base64)

            standardise_labeled_value = standardise_data_value(label_name, label_value)
            standardised_predicted_value = standardise_data_value(label_name, predicted_value)
            logger.debug(f'True label for {label_name}: {standardise_labeled_value}')
            logger.debug(f'Prediction for {label_name}: {standardised_predicted_value}')

            standardised_labeled_data_row[label_name] = standardise_labeled_value
            standardised_predicted_data_row[label_name] = standardised_predicted_value

        standardised_labeled_data.append(standardised_labeled_data_row)
        standardised_predicted_data.append(standardised_predicted_data_row)

    metrics_helper = MetricsHelper()
    metrics_helper.compare_true_pred(standardised_labeled_data, standardised_predicted_data)
    logger.info(f"Accuracy for Zero-shot extraction: {metrics_helper.accuracy}")


def evaluate_whole_json_labelbox(model_path: str, gen_key: str, project_id: str):
    export_data = load_labelbox_data(gen_key, project_id)
    extracted_labeled_data = extract_invoice_data(export_data, project_id)

    transformed_labeled_data = [transform_invoice_data(data) for data in extracted_labeled_data]
    # remove doc transcript from labelbox transformed data
    transformed_labeled_data = [{k: v for k, v in record.items() if k != 'Doc Transcript'}
                                for record in transformed_labeled_data]
    standardised_labeled_data = [standardise_data_models(info) for info in transformed_labeled_data]

    args = argparse.Namespace(
        checkpoint=model_path,
        num_beams=1,
        top_k=50,
        top_p=0.9,
        sample=True,
        dynamic=False,
        max_num=6,
        load_in_8bit=False,
        load_in_4bit=False,
        auto=False,
    )

    model, tokenizer = load_model_and_tokenizer(args)

    df_base64_strings = [get_pdf_base64_from_img_url(data["Img_path"]) for data in transformed_labeled_data]
    images_base64 = pdfs_to_images_base64_function(df_base64_strings)

    standardised_predicted_data = []
    for i, image_bs64 in enumerate(images_base64):
        logger.debug(f"True data")
        logger.debug(json.dumps(transformed_labeled_data[i], indent=4))

        pixel_values = load_image_bs64(image_bs64, max_num=args.max_num).to(torch.bfloat16).cuda()

        generation_config = dict(
            do_sample=args.sample,
            top_k=args.top_k,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=1024,
            eos_token_id=tokenizer.eos_token_id,
        )
        response = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=PROMPT,
            generation_config=generation_config,
            verbose=True
        )

        if "```" in response and "```json" not in response:
            # Extract just the dictionary-like part
            start = response.find("```") + 3
            end = response.find("\n```", start)

            response = response[start:end].strip()
        elif "```json" in response:
            # Extract just the dictionary-like part
            start = response.find("json") + len("json\n")
            end = response.find("\n```", start)

            response = response[start:end].strip()

        try:
            predicted_data_row = json.loads(response)
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            predicted_data_row = {}

        logger.debug(f"Predicted data")
        logger.debug(json.dumps(predicted_data_row, indent=4))
        logger.debug('-' * 50)
        standardised_predicted_data_row = standardise_data_models(predicted_data_row)
        standardised_predicted_data.append(standardised_predicted_data_row)

    metrics_helper = MetricsHelper()
    metrics_helper.compare_true_pred(standardised_labeled_data, standardised_predicted_data)
    logger.info(f"Accuracy for Zero-shot extraction: {metrics_helper.accuracy}")


@mlflow.trace
def evaluate_whole_json_data_row(model, tokenizer, eval_dataset_row: Dict[str, Any]):
    labeled_response = extract_json_data(eval_dataset_row['conversations'][1]['value'])
    standardised_labeled_response = standardise_data_models(labeled_response)
    logger.debug(f"True data")
    logger.debug(json.dumps(standardised_labeled_response, indent=4))

    img_path = eval_dataset_row['image']
    pixel_values = load_image(img_path)

    generation_config = dict(
        do_sample=args.sample,
        top_k=args.top_k,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=1024,
        eos_token_id=tokenizer.eos_token_id,
    )
    response = model.chat(
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        question=PROMPT,
        generation_config=generation_config,
        verbose=True
    )

    predicted_data_row = extract_json_data(response)
    standardised_predicted_response = standardise_data_models(predicted_data_row)
    logger.debug(f"Predicted data")
    logger.debug(json.dumps(standardised_predicted_response, indent=4))
    logger.debug('-' * 50)

    return standardised_labeled_response, standardised_predicted_response


@mlflow.trace
def evaluate_whole_json_dataset():
    with open(args.eval_dataset, 'r') as file:
        eval_dataset = [json.loads(line.strip()) for line in file]

    model, tokenizer = load_model_and_tokenizer(args)

    standardised_labeled_data = []
    standardised_predicted_data = []
    for i, eval_dataset_row in enumerate(eval_dataset):
        standardised_labeled_response, standardised_predicted_response = evaluate_whole_json_data_row(
            model, tokenizer, eval_dataset_row)

        standardised_labeled_data.append(standardised_labeled_response)
        standardised_predicted_data.append(standardised_predicted_response)

    metrics_helper = MetricsHelper()
    metrics_helper.compare_true_pred(standardised_labeled_data, standardised_predicted_data)
    logger.info(f"Accuracy for Zero-shot extraction: {metrics_helper.accuracy}")

    mlflow.log_param("model_path", args.model_path)
    mlflow.log_metric("accuracy", metrics_helper.accuracy)
    mlflow.log_table({
        "labeled_data": standardised_labeled_data,
        "predicted_data": standardised_predicted_data
    }, "data.json")
    mlflow.transformers.log_model(model, "model")


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", help="Path to the model for evaluation", type=str)
    parser.add_argument("--eval-dataset", help="Path to the dataset for evaluation", type=str)
    args = parser.parse_args()

    args2 = argparse.Namespace(
        checkpoint=args.model_path,
        num_beams=1,
        top_k=50,
        top_p=0.9,
        sample=True,
        dynamic=False,
        max_num=6,
        load_in_8bit=False,
        load_in_4bit=False,
        auto=False,
    )

    args = argparse.Namespace(**vars(args), **vars(args2))


    # LB_RAFT_GEN_KEY = os.getenv("LB_RAFT_GEN_KEY")
    # LB_PROJECT_ID = os.getenv("LB_PROJECT_ID")

    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        # evaluate_by_item(args.model_path, LB_RAFT_GEN_KEY, LB_PROJECT_ID)
        evaluate_whole_json_dataset()
