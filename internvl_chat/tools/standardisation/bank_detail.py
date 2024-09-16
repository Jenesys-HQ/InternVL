import logging
import re
from typing import List, Dict, Union

BANK_NAME = r'([^,]+),\s*'
SORT_CODE_REGEX = r'[S,s]ort [C,c]ode:\s*(\d{2}-\d{2}-\d{2}),*\s*'
ACCOUNT_NUMBER_REGEX = r'[A,a]ccount [N,n]umber:\s*(\d+),*\s*'
# PAYMENT_REFERENCE_REGEX = r'Payment reference:\s*(.+)'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
