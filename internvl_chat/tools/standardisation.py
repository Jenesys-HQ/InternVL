import logging
import re
from typing import Any, Dict, List, Optional, Union

import pyap
from postal import parser as postal_parser
import pycountry

from dateutil import parser

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BANK_NAME = r'([^,]+),\s*'
SORT_CODE_REGEX = r'[S,s]ort [C,c]ode:\s*(\d{2}-\d{2}-\d{2}),*\s*'
ACCOUNT_NUMBER_REGEX = r'[A,a]ccount [N,n]umber:\s*(\d+),*\s*'
# PAYMENT_REFERENCE_REGEX = r'Payment reference:\s*(.+)'


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


def standardise_bank_details(bank_details: Union[List[Dict[str, str]], str, None]):
    if bank_details is None:
        return []

    if type(bank_details) is list:
        if len(bank_details) == 0:
            return []

        if len(bank_details) > 1:
            logger.warning(f"Multiple bank details found: {bank_details}")

        return bank_details

    if type(bank_details) is str:
        bank_name = re.search(BANK_NAME, bank_details)
        sort_code = re.search(SORT_CODE_REGEX, bank_details)
        account_number = re.search(ACCOUNT_NUMBER_REGEX, bank_details)
        # payment_reference = re.search(PAYMENT_REFERENCE_REGEX, bank_details)

        # TODO add regex to extract all fields
        return [{
            'Company Name': None,
            'Account Number': account_number.group(1) if account_number else None,
            'Sort Code': sort_code.group(1) if sort_code else None,
            'Bank Name': bank_name.group(1) if bank_name else None,
            'Bank Number': None,
            'IBAN': None,
            'SWIFT Code': None,
            'Account Type': None,
            # 'Payment Reference': payment_reference.group(1) if payment_reference else None
        }]


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


def standardise_address(address: Optional[Union[str, dict]]) -> Optional[Dict[str, str]]:
    if address is None:
        return None

    if type(address) is dict:
        return address

    address = address.replace('\n', ', ')

    country_code = extract_country_from_address(address)
    try:
        addresses = pyap.parse(address, country=country_code)
        address = addresses[0]

        street = address.full_street
        city = address.city
        post_code = address.postal_code
        country = address.country
    except Exception as e:
        try:
            postal_address = {k: v for v, k in postal_parser.parse_address(address)}

            street = None
            if 'house' in postal_address:
                street = f"{postal_address['house']}"
            elif 'house_number' in postal_address:
                street = f"{postal_address['house_number']}"
                if 'road' in postal_address:
                    street += f", {postal_address['road']}"

            city = postal_address.get('city', None)
            post_code = postal_address.get('postcode', None)
            country = postal_address.get('country', None)

        except Exception as e2:
            logger.error(f"Error parsing address '{address}': {e2}")
            return None

    return {
        "Street": street,
        "City": city,
        "Postal Code": post_code.replace(' ', '').upper() if post_code else None,
        "Country": country
    }
    

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
        return None


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
        return str(float(str(number).replace(',', '')))
    except ValueError as e:
        return None


def standardise_string(string: Optional[str]):
    if not string:
        return None

    return string.lower()


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

