import unittest
from tools.standardisation import (
    standardise_date, standardise_bank_details,
    standardise_currency, standardise_integer, standardise_address
)


class TestStandardisation(unittest.TestCase):

    def test_standardise_date_valid(self):
        self.assertEqual(standardise_date("2023-01-01"), "01-01-2023")
        self.assertEqual(standardise_date("2023/01/01"), "01-01-2023")
        self.assertEqual(standardise_date("01/01/2023"), "01-01-2023")
        self.assertEqual(standardise_date("01-01-2023"), "01-01-2023")
        self.assertEqual(standardise_date("01/January/2023"), "01-01-2023")

    def test_standardise_date_invalid(self):
        self.assertIsNone(standardise_date("invalid-date"))

    def test_standardise_bank_details_valid(self):
        bank_details = "Bank Name, Sort code: 12-34-56, Account Number: 12345678, Payment reference: ref123"
        expected = [{
            'Bank Name': 'Bank Name',
            'Sort Code': '12-34-56',
            'Account Number': '12345678',
            # 'Payment Reference': 'ref123'
        }]
        self.assertEqual(standardise_bank_details(bank_details), expected)

    def test_standardise_bank_details_invalid(self):
        self.assertEqual(standardise_bank_details(None), [])
        self.assertEqual(standardise_bank_details("invalid"), [{
            'Account Number': None,
            'Bank Name': None,
            # 'Payment Reference': None,
            'Sort Code': None
        }])

    def test_standardise_address_valid(self):
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
                'Postal Code': 'N4 4QA',
                'Country': 'GB'
            }
        }

        for address, expected in dates.items():
            # assert that the dictionary returned is equal to the expected dictionary
            self.assertEqual(standardise_address(address), expected)

    def test_standardise_currency_valid(self):
        self.assertEqual(standardise_currency(100), "100.00")
        self.assertEqual(standardise_currency("100"), "100.00")
        self.assertEqual(standardise_currency("£100"), "100.00")

    def test_standardise_currency_invalid(self):
        self.assertEqual(standardise_currency("invalid"), None)

    def test_standardise_integer_valid(self):
        pass

    def test_standardise_integer_invalid(self):
        pass


if __name__ == "__main__":
    unittest.main()
