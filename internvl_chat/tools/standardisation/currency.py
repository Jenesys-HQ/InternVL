import logging
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
