import logging
import json
import os
import argparse

from tqdm import tqdm
import labelbox
import torch
from dotenv import load_dotenv

# from infer import JackVision
from internvl.model import load_model_and_tokenizer
from standardisation import standardise_data_models, standardise_data_value, standardise_data_models_flat
from data_utils import load_ndjson, save_ndjson, extract_invoice_data, transform_invoice_data, flatten_data
from img_utils import load_image, get_pdf_base64_from_img_url, pdf_to_image_base64_function, pdfs_to_images_base64_function

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

JSON_STRUCTURE = {
    "Document Type": "",
    "VAT": "",
    "Total": "",
    "VAT %": "",
    "Category": "",
    "Currency": "",
    "Discount Total": "",
    "Payment Status": "",
    "Service Charge": "",
    "Delivery Charge": "",
    "VAT Exclusive": "",
    "Supplier": "",
    "Invoice ID": "",
    "Line Items": [{
        "VAT": "",
        "VAT %": "",
        "Total": "",
        "Category": "",
        "Quantity": "",
        "Discount": "",
        "Unit price": "",
        "Description": ""
    }],
    "VAT Number": "",
    "Date of Invoice": "",
    "Date Payment Due": "",
    "Supplier Address": "",
    "Billing Address": "",
    "Delivery Address": "",
    "Bank Details": [{
        "Company Name": "",
        "Account Number": "",
        "Sort Code": "",
        "Bank Name": "",
        "Bank Number": "",
        "IBAN": "",
        "SWIFT Code": "",
        "Account Type": ""
    }]
}


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


def load_labelbox_data():
    eval_filepath = f'data/exports/{LB_PROJECT_ID}.jsonl'

    if os.path.exists(eval_filepath):
        return load_ndjson(eval_filepath)

    client = labelbox.Client(api_key=LB_RAFT_GEN_KEY)
    project = client.get_project(LB_PROJECT_ID)

    export_task = project.export_v2(params={"attachments": True}, filters={"workflow_status": "InReview"})
    export_task.wait_till_done()
    export_json = export_task.result

    save_ndjson(export_json, eval_filepath)

    return export_json


def evaluate_by_item(model_path: str):
    export_data = load_labelbox_data()
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

    model, tokenizer = load_model_and_tokenizer(
        checkpoint=model_path,
        root='./Your_Results',
        num_beams=5,
        top_k=50,
        top_p=0.9,
        sample=False,
        # dynamic=True,
        max_num=6,
        # load_in_8bit=False,
        # load_in_4bit=False,
        # auto=False
    )

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

    num_beams = 5
    top_k = 50
    top_p = 0.9
    sample=False

    model, tokenizer = load_model_and_tokenizer(
        checkpoint=model_path,
        root='./Your_Results',
        num_beams=num_beams,
        top_k=top_k,
        top_p=top_p,
        sample=False,
        # dynamic=True,
        max_num=6,
        # load_in_8bit=False,
        # load_in_4bit=False,
        # auto=False
    )
    image_size = model.config.force_image_size or model.config.vision_config.image_size

    prompt = f"""
        # Extraction Agent
        The invoice image is provided here:
        <image>

        You are an AI bookkeeper tasked with extracting and categorizing financial data from invoice images. 
        Your goal is to accurately extract all relevant data from the image and format it into a structured JSON output. 
        Follow these instructions carefully:

        Perform the following steps:

        1. Document Type Classification:
        - Classify the document as either a "Bill", "Receipt", or "Credit Note".
        - Note that any type of invoice should be classified as a "Bill".

        2. Data Extraction:
        - Extract all relevant financial and metadata information from the image.
        - Pay close attention to details such as dates, amounts, tax information, and line items.

        3. VAT Calculation:
        - If VAT information is present, ensure it is calculated as a fraction of the total amount.
        - Determine if the invoice is VAT exclusive or inclusive.

        4. Payment Status:
        - Classify the payment status as either "Awaiting Payment" or "Paid" based on the information provided.

        5. Currency:
        - Identify and extract the currency used in the invoice.

        6. Line Items:
        - Extract individual line items, including their descriptions, quantities, unit prices, and totals.
        - Extract VAT for each line item if available. VAT might be labelled as 'Tax' on the Invoice.
        - If Unit Price is not provided, calculate it as Subtotal divided by Quantity. 
        
        7. Address Information:
        - Extract and categorize supplier address, billing address, and delivery address.
        - Do not include company name or staff name in the address fields.

        8. Bank Details:
        - Extract any bank account information provided in the invoice.

        9. Dates:
        - Extract the invoice date and due date (if available).

        10. Additional Charges:
        - Identify and extract any service charges, delivery charges, or discounts.

        11. Supplier Information:
        - Extract the supplier name and details carefully.
        - Be cautious not to confuse supplier information with customer information.
        - If the supplier name isn't clear, look for clues in the company's logo, contact email, invoice footer, or VAT number location.
        - Do not include address information in the supplier field.

        When extracting data, adhere to these guidelines:
        - If a field is not present in the invoice, leave it as an empty string in the JSON output.
        - For numerical values, extract them as numbers without currency symbols.
        - For dates, use the format "DD/MM/YYYY".
        - If there's uncertainty about a value, use your best judgment based on context and typical invoice structures.

        After extraction, format the data according to the following JSON schema:

        <json_schema>
        {{
            "Document Type": "",
            "VAT": "",
            "Total": "",
            "VAT %": "",
            "Category": "",
            "Currency": "",
            "Discount Total": "",
            "Payment Status": "",
            "Service Charge": "",
            "Delivery Charge": "",
            "VAT Exclusive": "",
            "Supplier": "",
            "Invoice ID": "",
            "Line Items": [{{
                "VAT": "",
                "VAT %": "",
                "Total": "",
                "Category": "",
                "Quantity": "",
                "Discount": "",
                "Unit price": "",
                "Description": ""
            }}],
            "VAT Number": "",
            "Date of Invoice": "",
            "Date Payment Due": "",
            "Supplier Address": "",
            "Billing Address": "",
            "Delivery Address": "",
            "Bank Details": [{{
                "Company Name": "",
                "Account Number": "",
                "Sort Code": "",
                "Bank Name": "",
                "Bank Number": "",
                "IBAN": "",
                "SWIFT Code": "",
                "Account Type": ""
            }}]
        }}
        </json_schema>

        Before finalizing your output, perform these quality checks:
        1. Ensure all extracted data is accurately placed in the correct fields.
        2. Verify that numerical calculations (totals, VAT, etc.) are consistent and accurate.
        3. Check that dates are formatted correctly.
        4. Confirm that the document type classification is appropriate.
        5. Validate that the VAT Exclusive field is correctly set to true or false.
        6. Make sure not to confuse the total VAT with the line item VAT, and that the sum of all the VAT values in the line items is equal to the total VAT.
        7. Double-check that the supplier information is correct and not confused with customer details.
        8. Verify that address fields do not contain company or staff names.
        9. Never conflict Subtotal with Total or Unit Price or Line Item Total or VAT or Discount Amount.

        If you encounter any ambiguities or missing information, use your best judgment to infer the most likely value based on the context of the invoice. If a value cannot be reasonably inferred, leave it as an empty string.

        Provide your final output as a valid JSON object within. Ensure that the JSON is properly formatted and contains no syntax errors.
    """

    logger.info(prompt)

    # df_base64_strings = [get_pdf_base64_from_img_url(data["Img_path"]) for data in transformed_labeled_data]
    # images_base64 = pdfs_to_images_base64_function(df_base64_strings)
    all_pixel_values = [load_image(data['Img_path'], image_size).cuda().to(torch.bfloat16)
                        for data in transformed_labeled_data]

    predicted_data = []
    for i, pixel_values in tqdm(enumerate(all_pixel_values)):
        logger.info(f"True data")
        logger.info(json.dumps(transformed_labeled_data[i], indent=4))

        generation_config = dict(
            do_sample=sample,
            top_k=top_k,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=20,
            eos_token_id=tokenizer.eos_token_id,
        )
        response = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=prompt,
            generation_config=generation_config,
            verbose=True
        )

        try:
            predicted_data_row = json.loads(response)
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            predicted_data_row = {}

        logger.info(f"Predicted data")
        logger.info(json.dumps(predicted_data_row, indent=4))
        logger.info('-' * 50)
        predicted_data.append(predicted_data_row)
    standardised_predicted_data = [standardise_data_models(predicted_data_row) for predicted_data_row in predicted_data]

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
