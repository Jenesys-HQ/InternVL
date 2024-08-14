import argparse
import json
import logging
import os

import torch
from dotenv import load_dotenv

from constants import PROMPT
from data_utils import extract_invoice_data, load_labelbox_data, transform_invoice_data, flatten_data
from img_utils import get_pdf_base64_from_img_url, pdf_to_image_base64_function, load_image_bs64, \
    pdfs_to_images_base64_function
from internvl.model import load_model_and_tokenizer
from standardisation import standardise_data_models, standardise_data_value

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def calculate_metrics(predicted, ground_truth):
    # iterate over predicted and ground truth keys and compare ground truth and predicted values
    # calculate averages of F1, precision and recall

    total_correct = 0
    total_count = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0

    for pred, true in zip(predicted, ground_truth):
        count = 0
        true_p = 0  # true positive
        false_p = 0  # false positive
        false_n = 0  # false negative

        for key in true:
            if true[key] is None:
                continue

            count += 1
            if key in pred and pred[key] == true[key]:
                true_p += 1
            else:
                if key in pred:
                    false_p += 1
                if key not in pred or pred[key] != true[key]:
                    false_n += 1

        total_correct += true_p
        total_count += count

        precision = true_p / (true_p + false_p) if (true_p + false_p) > 0 else 0
        recall = true_p / (true_p + false_n) if (true_p + false_n) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        total_precision += precision
        total_recall += recall
        total_f1 += f1

    total_accuracy = total_correct / total_count if total_count > 0 else 0
    total_precision = total_precision / total_count if total_count > 0 else 0
    total_recall = total_recall / total_count if total_count > 0 else 0
    total_f1 = total_f1 / total_count if total_count > 0 else 0

    return total_accuracy, total_precision, total_recall, total_f1


def evaluate_by_item(model_path: str):
    export_data = load_labelbox_data(LB_PROJECT_ID)
    extracted_labeled_data = extract_invoice_data(export_data, LB_PROJECT_ID)
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

    accuracy, precision, recall, f1 = calculate_metrics(standardised_labeled_data, standardised_predicted_data)
    logger.info("============= Metrics for Zero-shot extraction =============")
    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"F1 Score: {f1}")


def evaluate_whole_json(model_path: str):
    export_data = load_labelbox_data()
    extracted_labeled_data = extract_invoice_data(export_data, LB_PROJECT_ID)

    transformed_labeled_data = [transform_invoice_data(data) for data in extracted_labeled_data]
    # remove doc transcript from labelbox transformed data
    transformed_labeled_data = [{k: v for k, v in record.items() if k != 'Doc Transcript'}
                                for record in transformed_labeled_data]
    standardised_labeled_data = [standardise_data_models(info) for info in transformed_labeled_data]

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

    accuracy, precision, recall, f1 = calculate_metrics(standardised_predicted_data, standardised_labeled_data)
    logger.info("============= Metrics for Zero-shot extraction =============")
    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"F1 Score: {f1}")


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", help="Path to the model for evaluation", type=str)
    args = parser.parse_args()

    LB_RAFT_GEN_KEY = os.getenv("LB_RAFT_GEN_KEY")
    LB_PROJECT_ID = os.getenv("LB_PROJECT_ID")

    # evaluate_by_item(args.model_path)
    evaluate_whole_json(args.model_path)
