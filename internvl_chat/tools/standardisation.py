import logging
import re
from typing import Any, Dict

import pyap
import pycountry

from dateutil import parser

BANK_DETAILS_REGEX = (
    r'([^,]+),\s*'  # bank name       - look into refactoring regex using named capture groups for regex
    r'Sort code:\s*(\d{2}-\d{2}-\d{2}),\s*'  # sort code
    r'Account Number:\s*(\d+),\s*'  # account number
    r'Payment reference:\s*(.+)'  # payment reference
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def standardise_date(date):
    if not date:
        return None

    try:
        # convert date time string into object and return day-month-year
        parsed_date = parser.parse(date)
        return parsed_date.strftime("%d-%m-%Y")
    except ValueError as e:
        logger.error(f"Error parsing date '{date}': {e}")
        return None


def standardise_bank_details(bank_details: str):
    if not bank_details:
        return None

    match = re.match(BANK_DETAILS_REGEX, bank_details)
    if match:
        return {
            'Bank Name': match.group(1).strip(),
            'Sort Code': match.group(2).strip(),
            'Account Number': match.group(3).strip(),
            'Payment Reference': match.group(4).strip()
        }
    else:
        logger.error(f"Error parsing bank details: '{bank_details}'")
        return None


def get_country_code(country_identifier: str):
    try:
        if len(country_identifier) == 2:
            # Assuming country_identifier is a 2-letter country code
            country = pycountry.countries.get(alpha_2=country_identifier)
        elif len(country_identifier) == 3:
            # Assuming country_identifier is a 3-letter country code
            country = pycountry.countries.get(alpha_3=country_identifier)
        else:
            # Assuming country_identifier is a country name
            country = pycountry.countries.lookup(country_identifier)
        return country.alpha_2
    except LookupError as e:
        logger.warning(e)
        return None


def extract_country_from_address(address: str):
    last_component = address.split(',')[-1].strip()

    return get_country_code(last_component)


def standardise_address(address: str):
    if address is None:
        return None

    address = address.replace('\n', ', ')

    country_code = extract_country_from_address(address)
    addresses = pyap.parse(address, country=country_code)
    if addresses:
        address = addresses[0]
        return {
            "Street": address.full_street,
            "City": address.city,
            "Postal Code": address.postal_code,
            "Country": address.country
        }
    else:
        return None


def standardise_currency(number):
    if not number:
        return None

    try:
        trim = re.compile(r'[^\d.,]+')
        number_trim = trim.sub('', str(number))
        number = float(number_trim)
        return f"{float(number):.2f}"
    except ValueError as e:
        logger.error(f'Error standardising value {number}: {e}')
        return number


def standardise_integer(number):
    if not number:
        return None

    try:
        return str(int(number))
    except ValueError as e:
        return None


def standardise_float(number):
    if not number:
        return None

    try:
        return str(float(number.replace(',', '')))
    except ValueError as e:
        return None


def standardise_string(string: str):
    if not string:
        return None

    return string.lower()


def standardise_data_models(data: Dict[str, Any]) -> Dict[str, Any]:
    for name, value in data.items():
        if value == "":
            data[name] = None

    return {
        "VAT": standardise_currency(data.get('VAT', None)),
        "Total": standardise_currency(data.get('Total', None)),
        "VAT %": data.get('VAT %', None),
        "Currency": data.get('Currency', None),
        "Supplier": data.get('Supplier', None),
        "Invoice ID": data.get('Invoice ID', None),
        "VAT Number": data.get('VAT Number', None),
        "Date of Invoice": standardise_date(data.get('Date of Invoice', None)),
        "Date Payment Due": standardise_date(data.get('Date Payment Due', None)),
        "Supplier Address": standardise_address(data.get('Supplier Address', None)),
        "Billing Address": standardise_address(data.get('Billing Address', None)),
        "Delivery Address": standardise_address(data.get('Delivery Address', None)),
        "Bank Details": standardise_bank_details(data.get('Bank Details', None)),
        "Line Items": [{
            "VAT": standardise_currency(line_item.get('VAT', None)),
            "VAT%": standardise_integer(line_item.get('VAT %', None)),
            "Total": standardise_currency(line_item.get('Total', None)),
            "Quantity": standardise_float(line_item.get('Quantity', None)),
            "Unit price": standardise_currency(line_item.get('Unit price', None)),
            "Description": line_item.get('Description', None)
        } for line_item in data.get('Line Items', [])],
        "Img_path": data.get('Img_path', None)
    }


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


def standardise_data_models_flat(data: Dict[str, Any]) -> Dict[str, Any]:
    for name, value in data.items():
        if value == "":
            data[name] = None

    standardised_data = {}
    for name, value in data.items():
        standardised_data[name] = standardise_data_value(name, value)

    return standardised_data

