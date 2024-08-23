import pytest
from tools.standardisation import (
    standardise_date, standardise_bank_details,
    standardise_currency, standardise_integer, standardise_address
)


def test_standardise_date_valid():
    assert standardise_date("2023-01-01") == "01-01-2023"
    assert standardise_date("2023/01/01") == "01-01-2023"
    assert standardise_date("01/01/2023") == "01-01-2023"
    assert standardise_date("01-01-2023") == "01-01-2023"
    assert standardise_date("01/January/2023") == "01-01-2023"


def test_standardise_date_invalid():
    assert standardise_date("invalid-date") is None


def test_standardise_bank_details_valid():
    bank_details = "Bank Name, Sort code: 12-34-56, Account Number: 12345678, Payment reference: ref123"
    expected = [{
        'Company Name': None,
        'Account Number': '12345678',
        'Sort Code': '12-34-56',
        'Bank Name': 'Bank Name',
        'Bank Number': None,
        'IBAN': None,
        'SWIFT Code': None,
        'Account Type': None,
        # 'Payment Reference': 'ref123'
    }]
    assert standardise_bank_details(bank_details) == expected


def test_standardise_bank_details_invalid():
    assert standardise_bank_details(None) == []
    assert standardise_bank_details("invalid") == [{
        'Company Name': None,
        'Account Number': None,
        'Sort Code': None,
        'Bank Name': None,
        'Bank Number': None,
        'IBAN': None,
        'SWIFT Code': None,
        'Account Type': None,
        # 'Payment Reference': None,
    }]


def test_standardise_address_valid():
    dates = {
        '10 Alford Court\nLondon, London N17JW\nUnited Kingdom': {
            'Street': '10 Alford Court',
            'City': 'London',
            'Postal Code': 'N17JW',
            'Country': 'United Kingdom'
        },
        '84A Stapleton Hall Road London, London, London, N4 4QA, GB': {
            'Street': '84A Stapleton Hall Road London',
            'City': 'London',
            'Postal Code': 'N44QA',
            'Country': 'GB'
        },
        'Unit B, Staplehurst Nurseries, Staplehurst, Kent, TN12 0JT': {
            'Street': 'unit b staplehurst nurseries',
            'City': 'staplehurst',
            'Postal Code': 'TN120JT',
            'Country': None
        }
    }

    for address, expected in dates.items():
        assert standardise_address(address) == expected


def test_standardise_currency_valid():
    assert standardise_currency(100) == "100.00"
    assert standardise_currency("100") == "100.00"
    assert standardise_currency("£100") == "100.00"


def test_standardise_currency_invalid():
    assert standardise_currency("invalid") is None


def test_standardise_integer_valid():
    pass


def test_standardise_integer_invalid():
    pass
