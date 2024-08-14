from dotenv import load_dotenv
import json
import os
import logging
from os.path import join, dirname, abspath, exists
from typing import Any, List, Dict
import boto3
import uuid
import pdf2image
import labelbox

from constants import JSON_STRUCTURE, PROMPT

DATA_ROW_MAPPING = {
    'what_is_the_invoice_number': 'Invoice ID',
    'what_is_the_invoice_date': 'Date of Invoice',
    'what_is_the_payment_due_date': 'Date Payment Due',
    'bank_details': 'Bank Details',
    'payment_email': None,
    'pay_pal': None,
    'payment_platform': None,
    'extra_charges': None,
    'extra_charges_text': None,
    'unit_total': 'Total',
    'total': 'Total',
    # 'category_code': 'Category',
    'supplier_name': 'Supplier',
    'supplier_address': 'Supplier Address',
    'billing_address': 'Billing Address',
    'delivery_address': 'Delivery Address',
    'vat_number': 'VAT Number',
    'handwritten': None,
    'identify_the_invoice_language': None,
    'document_text': None,
    'what_is_the_document_type': None,  # TODO should be mapped to Document Type but the annotated values are incorrect
    'currency': 'Currency'
}

LINE_ITEM_MAPPING = {
    'item_description': 'Description',
    'unit_price': 'Unit price',
    'quantity': 'Quantity',
    # 'gross_total': 'Total',
    'gross_total_incl_vat': 'Total',
    'category_code': 'Category',
    'vat': 'VAT',
    'tax_code': 'VAT %',  # do we want % or just the number?
}


LOCAL_IMAGE_FOLDER = join(
    dirname(abspath(__file__)),
    '..',
    'data',
    'images'
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def load_ndjson(file_path: str):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    return data


def save_ndjson(data: List[Dict], file_path: str):
    with open(file_path, 'w') as file:
        for data_dict in data:
            json_line = json.dumps(data_dict)
            file.write(json_line + '\n')


def load_labelbox_data(gen_key: str, project_id: str):
    eval_filepath = f'data/exports/{project_id}.jsonl'

    if os.path.exists(eval_filepath):
        return load_ndjson(eval_filepath)

    client = labelbox.Client(api_key=gen_key)
    project = client.get_project(project_id)

    export_task = project.export_v2(params={"attachments": True})
    export_task.wait_till_done()
    export_json = export_task.result

    save_ndjson(export_json, eval_filepath)

    return export_json


# Check the format of multiple strings use set to get only unique types returned
def detect_file_format(df_base64_strings):
    file_type = set()  # set used to return unique values from list

    for file in df_base64_strings:
        if file is None:
            return None

        if file.startswith('JVBER'):
            file_type.add('pdf')
        elif file.startswith('/9j/'):
            file_type.add('jpeg')
        elif file.startswith('iVBOR'):
            file_type.add('png')

    if not file_type:
        return None
    else:
        return file_type


def remove_duplicates(lst: List[Dict[str, str]]) -> List[Dict[str, str]]:
    str_lst = [str(i) for i in lst]
    count_dict = {}
    unique_lst = []

    for i in str_lst:
        if i in count_dict:
            count_dict[i] += 1
        else:
            count_dict[i] = 1

    for item, count in count_dict.items():
        if count == 1:
            unique_lst.append(eval(item))

    return unique_lst


def download_image(image_url: str, dataset_name: str) -> str:
    s3 = boto3.client('s3')

    bucket_name = image_url.split('/')[2].split('.')[0]
    object_name = '/'.join(image_url.split('/')[3:])

    local_file_name = f"{LOCAL_IMAGE_FOLDER}/{dataset_name}/{object_name.split('/')[-1]}"

    if exists(local_file_name):
        return local_file_name

    os.makedirs(dirname(local_file_name), exist_ok=True)

    print(f'Downloading {bucket_name} {object_name} to {local_file_name}')

    s3.download_file(bucket_name, object_name, local_file_name)

    return local_file_name


def expand_sentence(question: str, answer: str) -> str:
    if len(question.split()) > 2:
        return question

    if answer in ['Yes', 'No']:
        sentence = f'Is the document {question}?'
    else:
        sentence = f'What is the {question} of the document?'

    return sentence


def format_data_row(annotation, image_path: str) -> Dict[str, str]:
    question = annotation['name']

    if 'radio_answer' in annotation:
        answer = annotation['radio_answer']['name']
    elif 'text_answer' in annotation:
        answer = annotation['text_answer']['content']
    else:
        raise ValueError('No answer found')

    question = expand_sentence(question, answer)

    return {
        'id': str(uuid.uuid4()),
        'image': image_path,
        'conversations': [{
            'from': 'human',
            'value': f'<image>\n{question}'
        }, {
            'from': 'gpt',
            'value': answer
        }]
    }


def format_data_row_whole(data_row: Dict[str, Any], image_path: str) -> Dict[str, Any]:
    return {
        'id': str(uuid.uuid4()),
        'image': image_path,
        'conversations': [{
            'from': 'human',
            'value': PROMPT
        }, {
            'from': 'gpt',
            'value': f"```json\n{json.dumps(data_row, indent=4)}\n```"
        }]
    }


def convert_pdf_to_image(pdf_path: str) -> str:
    output_folder = os.path.dirname(pdf_path)
    images = pdf2image.convert_from_path(pdf_path, output_folder=output_folder)

    image_path = pdf_path.replace('.pdf', '.jpg')

    # TODO at the moment we only convert the first page of the pdf
    images[0].save(image_path, 'JPEG')

    return image_path
    
    
def preprocess_data(data_file: str, dataset_name: str):
    data = load_ndjson(data_file)

    formatted_data = []

    for data_row in data:
        annotations = data_row['projects']['clth4b4ys0byx07x0eb4odl50']['labels'][0]['annotations']

        for annotation in annotations['classifications']:
            image_url = data_row['data_row']['row_data']
            image_path = download_image(image_url, dataset_name)

            if image_url.endswith('.pdf'):
                image_path = convert_pdf_to_image(image_path)

            formatted_annotation = format_data_row(annotation, image_path)

            formatted_data.append(formatted_annotation)

        for obj in annotations['objects']:
            for annotation in obj['classifications']:
                image_url = data_row['data_row']['row_data']
                image_path = download_image(image_url, dataset_name)

                if image_url.endswith('.pdf'):
                    image_path = convert_pdf_to_image(image_path)

                formatted_annotation = format_data_row(annotation, image_path)

                formatted_data.append(formatted_annotation)

    formatted_data = remove_duplicates(formatted_data)

    return formatted_data


def preprocess_data_whole(gen_key: str, project_id: str, dataset_name: str) -> Dict[str, Any]:
    export_data = load_labelbox_data(gen_key, project_id)

    formatted_data = []
    for data_row in export_data:
        processed_data_row = {key: None for key in JSON_STRUCTURE.keys()}
        processed_data_row["Line Items"] = [{key: None for key in JSON_STRUCTURE["Line Items"][0].keys()}]
        processed_data_row["Bank Details"] = [{key: None for key in JSON_STRUCTURE["Bank Details"][0].keys()}]
        processed_data_row['Document Type'] = 'Bill'  # We assume all the documents in labelbox are bills

        image_url = data_row['data_row']['row_data']
        image_path = download_image(image_url, dataset_name)
        if image_url.endswith('.pdf'):
            image_path = convert_pdf_to_image(image_path)

        annotations = data_row['projects'][project_id]['labels'][0]['annotations']

        for annotation in annotations['classifications']:
            process_annotation(annotation, processed_data_row)
        for obj in annotations['objects']:
            for annotation in obj['classifications']:
                process_annotation(annotation, processed_data_row)

        logger.info(json.dumps(processed_data_row, indent=4))
        formatted_data_row = format_data_row_whole(processed_data_row, image_path)
        formatted_data.append(formatted_data_row)

    return formatted_data


def process_annotation(annotation: Dict, formatted_data_row: Dict) -> None:
    if not annotation['value'] in DATA_ROW_MAPPING and not annotation['value'] in LINE_ITEM_MAPPING:
        logger.warning(f'Annotation {annotation["name"]} - {annotation["value"]} not found in mapping')
        return None

    if annotation['value'] in DATA_ROW_MAPPING:
        key = DATA_ROW_MAPPING[annotation['value']]
        value_to_update = formatted_data_row

    if annotation['value'] in LINE_ITEM_MAPPING:
        key = LINE_ITEM_MAPPING[annotation['value']]
        value_to_update = formatted_data_row['Line Items'][0]

    if not key:
        return None
    if 'text_answer' in annotation:
        value_to_update[key] = annotation['text_answer']['content']
    if 'radio_answer' in annotation:
        # TODO handle radio answers
        pass


def extract_invoice_data(data, labelbox_project_id):
    extracted_data = []

    for data_row in data:
        invoice_details = {
            "line_items": [{
                "item_description": None,
                "unit_price": None,
                "quantity": None,
                "vat": None,
                "gross_total": None,
                "category_code": None,
                "tax_code": None
            }],
            "total": None,
            "invoice_date": None,
            "payment_due_date": None,
            "supplier_name": None,
            "document_text": None,
            "img_path": None
            # "billing_address": None,
            # "supplier_address": None,
            # "delivery_address": None,  # Assuming delivery address is needed
            # "bank_details": None
        }

        annotations = data_row['projects'][labelbox_project_id]['labels'][0]['annotations']['objects']
        invoice_details["img_path"] = data_row["data_row"]["row_data"]

        # # Extracting document text
        # if 'document_text' in data_row['projects'][labelbox_project_id]['labels'][0]:
        #     invoice_details["document_text"] = data_row['projects'][labelbox_project_id]['labels'][0]['text_answer']['content']

        classifications = data_row['projects'][labelbox_project_id]['labels'][0]['annotations']['classifications']

        for classification in classifications:
            name = classification['name']
            # text_answer = classification['text_answer']
            # content = text_answer.get('content') if text_answer else None

            if name == 'Document Text':
                content = classification['text_answer']['content']
                invoice_details["document_text"] = content

        for annotation in annotations:
            name = annotation['name']
            classifications = annotation.get('classifications', [])

            if not classifications:
                continue

            text_answer = classifications[0].get('text_answer')
            content = text_answer.get('content') if text_answer else None

            if name == 'Item Description' and content:
                invoice_details["line_items"][0]['item_description'] = content
            elif name == 'Unit Price' and content:
                invoice_details["line_items"][0]["unit_price"] = content
            elif name == 'Quantity' and content:
                invoice_details["line_items"][0]["quantity"] = content
            elif name == 'VAT' and content:
                invoice_details["line_items"][0]["vat"] = content
            elif name == 'Gross Total' and content:
                invoice_details["line_items"][0]["gross_total"] = content
            # elif name == 'Category Code' and content:
            #   invoice_details["line_items"][0]["category_code"] = content
            elif name == 'Tax Code' or name == 'VAT%' and content:
                invoice_details["line_items"][0]["tax_code"] = content
            elif name == 'Total' and content:
                invoice_details["total"] = content
            elif name == 'What is the invoice date?' and content:
                invoice_details["invoice_date"] = content
            elif name == 'What is the payment due date?' and content:
                invoice_details["payment_due_date"] = content
            elif name == 'Supplier Name' and content:
                invoice_details["supplier_name"] = content
            elif name == 'Billing Address' and content:
                invoice_details["billing_address"] = content
            elif name == 'Supplier Address' and content:
                invoice_details["supplier_address"] = content
            elif name == 'Delivery Address' and content:  # Assuming this key exists
                invoice_details["delivery_address"] = content
            elif name == 'Bank Details' and content:
                invoice_details["bank_details"] = content

        extracted_data.append(invoice_details)

    return extracted_data


def flatten_data(data_dict: Dict[str, Any], flat_data: Dict[str, Any], prefix: str = ''):
    for key, value in data_dict.items():
        if value is None:
            continue

        if type(value) is list:
            for i, item in enumerate(value):
                flatten_data(item, flat_data, f'{prefix}{key} {i+1} ')
        else:
            name = f'{prefix}{key}'
            flat_data[name] = value


def transform_invoice_data(extracted_invoice_data):
    # Transform the main document keys
    transformed_data = {
        "VAT": None,
        "Total": extracted_invoice_data.get('total', None),
        "VAT %": None,
        # "Category": None,
        "Currency": extracted_invoice_data.get('total')[0] if extracted_invoice_data.get('total', None) else None,
        "Supplier": extracted_invoice_data.get('supplier_name', None),
        "Doc Transcript": extracted_invoice_data.get('document_text', None),
        "Invoice ID": None,
        "Line Items": [],
        "VAT Number": None,
        "Date of Invoice": extracted_invoice_data.get('invoice_date', None),
        "Date Payment Due": extracted_invoice_data.get('payment_due_date', None),
        "Supplier Description": extracted_invoice_data.get('bank_details', None),
        "Expense Policy Review": None,
        "Expense Policy Review Status": None,
        "Img_path": extracted_invoice_data.get('img_path', None),
        "Bank Details": [extracted_invoice_data.get('bank_details')] if extracted_invoice_data.get('bank_details', None) else []
    }

    # Transform each line item
    for item in extracted_invoice_data.get('line_items', []):
        line_item = {
            "VAT": item.get('vat', None),
            "VAT%": item.get('tax_code', None),
            "Total": item.get('gross_total', None),
            # "Category": item.get('category_code', None),
            "Quantity": item.get('quantity', None),
            "Unit price": item.get('unit_price', None),
            "Description": item.get('item_description', None)
        }
        transformed_data["Line Items"].append(line_item)

    return transformed_data


if __name__ == "__main__":
    load_dotenv()

    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_REGION')
    LB_RAFT_GEN_KEY = os.getenv("LB_RAFT_GEN_KEY")
    LB_PROJECT_ID = 'clth4b4ys0byx07x0eb4odl50'

    dataset_name = 'cp-jack-reconciliation-knowledge'

    data = preprocess_data_whole(LB_RAFT_GEN_KEY, LB_PROJECT_ID, dataset_name)

    out_folder = join(
        dirname(abspath(__file__)),
        '..',
        'data',
        'processed_whole',
    )

    os.makedirs(out_folder, exist_ok=True)

    json.dump(data, open(join(
        out_folder,
        f'{dataset_name}.json'
    ), 'w+'), indent=4)
