from typing import Optional


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
