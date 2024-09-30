from dateutil import parser
import logging

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
