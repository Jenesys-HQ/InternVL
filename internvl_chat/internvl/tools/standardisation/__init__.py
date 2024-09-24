from typing import Any, Dict, Optional

from .address import standardise_address
from .bank_detail import standardise_bank_details
from .base import standardise_float, standardise_integer, standardise_string
from .currency import standardise_currency
from .date import standardise_date

INTEGER = 'integer'
FLOAT = 'float'
STRING = 'string'
CURRENCY = 'currency'
DATE = 'date'
ADDRESS = 'address'
BANK_DETAILS = 'bank_details'

METHOD_MAPPING = {
    INTEGER: standardise_integer,
    FLOAT: standardise_float,
    STRING: standardise_string,
    CURRENCY: standardise_currency,
    DATE: standardise_date,
    ADDRESS: standardise_date,
    BANK_DETAILS: standardise_bank_details
}

DATA_TYPES = {
    'Img_path': STRING,
    'VAT': CURRENCY,
    'Total': CURRENCY,
    "VAT %": STRING,
    "Currency": STRING,
    "Supplier": STRING,
    "Supplier Description": STRING,
    "Invoice ID": STRING,
    "VAT Number": STRING,
    "Date of Invoice": DATE,
    "Date Payment Due": DATE,
    "Supplier Address": ADDRESS,
    "Billing Address": ADDRESS,
    "Delivery Address": ADDRESS,
    "Bank Details": BANK_DETAILS,
    # TODO make more generic to include multiple line items
    "Line Items 1 VAT": CURRENCY,
    "Line Items 1 VAT%": STRING,
    "Line Items 1 Total": CURRENCY,
    "Line Items 1 Quantity": FLOAT,
    "Line Items 1 Unit price": CURRENCY,
    "Line Items 1 Description": STRING
}


def standardise_data_value(name, value):
    data_type = DATA_TYPES[name]
    return METHOD_MAPPING[data_type](value) if value != '' else None


def standardise_backend_value(value: Optional[str]):
    if not value:
        return None

    return value.upper().replace(' ', '_')


def standardise_data_models(data: Dict[str, Any]) -> Dict[str, Any]:
    for name, value in data.items():
        if value == "":
            data[name] = None

    return {
        "Document Type": standardise_backend_value(data.get('Document Type', None)),
        "VAT": standardise_currency(data.get('VAT', None)),
        "Total": standardise_currency(data.get('Total', None)),
        "VAT %": data.get('VAT %', None),
        "Category": standardise_backend_value(data.get('Category', None)),
        "Currency": data.get('Currency', None),
        "Discount Total": standardise_currency(data.get('Discount Total', None)),
        "Payment Status": standardise_backend_value(data.get('Payment Status', None)),
        "Service Charge": standardise_currency(data.get('Service Charge', None)),
        "Delivery Charge": standardise_currency(data.get('Delivery Charge', None)),
        "VAT Exclusive": data.get('VAT Exclusive', None),
        "Supplier": data.get('Supplier', None),
        "Invoice ID": data.get('Invoice ID', None),
        "Line Items": [{
            "VAT": standardise_currency(line_item.get('VAT', None)),
            "VAT %": line_item.get('VAT %', None),
            "Total": standardise_currency(line_item.get('Total', None)),
            "Category": data.get('Category', None),
            "Quantity": standardise_float(line_item.get('Quantity', None)),
            "Discount": standardise_currency(line_item.get('Discount', None)),
            "Unit price": standardise_currency(line_item.get('Unit price', None)),
            "Description": line_item.get('Description', None)
        } for line_item in data.get('Line Items', [])],
        "VAT Number": data.get('VAT Number', None),
        "Date of Invoice": standardise_date(data.get('Date of Invoice', None)),
        "Date Payment Due": standardise_date(data.get('Date Payment Due', None)),
        "Supplier Address": standardise_address(data.get('Supplier Address', None)),
        "Billing Address": standardise_address(data.get('Billing Address', None)),
        "Delivery Address": standardise_address(data.get('Delivery Address', None)),
        "Bank Details": standardise_bank_details(data.get('Bank Details', [])),
    }


def standardise_data_models_flat(data: Dict[str, Any]) -> Dict[str, Any]:
    for name, value in data.items():
        if value == "":
            data[name] = None

    standardised_data = {}
    for name, value in data.items():
        standardised_data[name] = standardise_data_value(name, value)

    return standardised_data
