import pytest
from tools.metrics import MetricsHelper


def test_calculate_metrics_address():
    ground_truth = [{
        'Address': {
            "Street": "some road",
            "City": "london",
            "Postal Code": "E33AB",
            "Country": 'UK'
        }
    }]
    predicted = [{
        'Address': {
            "Street": "some road",
            "City": "london",
            "Postal Code": "E33AB",
            "Country": None
        }
    }]

    metrics_helper = MetricsHelper()
    metrics_helper.compare_true_pred(ground_truth, predicted)

    assert metrics_helper.accuracy == 3/4


def test_calculate_metrics_line_items_different_order():
    ground_truth = [{
        'Line Items': [
            {'Total': 10, 'VAT': 1},
            {'Total': 20, 'VAT': 1},
            {'Total': 30, 'VAT': 1},
        ]
    }]
    predicted = [{
        'Line Items': [
            {'VAT': 1, 'Total': 10},
            {'Total': 20, 'VAT': 1},
            {'Total': 30, 'VAT': 1},
        ]
    }]

    metrics_helper = MetricsHelper()
    metrics_helper.compare_true_pred(ground_truth, predicted)

    assert metrics_helper.accuracy == 1.0


def test_calculate_metrics_line_items_none_values():
    ground_truth = [{
        'Line Items': [
            {'Total': 10, 'VAT': 1},
            {'Total': 20, 'VAT': None},
            {'Total': 30, 'VAT': 1},
        ]
    }]
    predicted = [{
        'Line Items': [
            {'Total': 10, 'VAT': 1},
            {'Total': 20, 'VAT': 1},
            {'Total': 30, 'VAT': 1},
        ]
    }]

    metrics_helper = MetricsHelper()
    metrics_helper.compare_true_pred(ground_truth, predicted)

    assert metrics_helper.accuracy == 1.0


def test_calculate_metrics_line_items_missing_line_item():
    ground_truth = [{
        'Line Items': [
            {'Total': 10, 'VAT': 1},
            {'Total': 20, 'VAT': None},
            {'Total': 30, 'VAT': 1},
        ]
    }]
    predicted = [{
        'Line Items': [
            {'Total': 10, 'VAT': 1},
            {'Total': 20, 'VAT': 1}
        ]
    }]

    metrics_helper = MetricsHelper()
    metrics_helper.compare_true_pred(ground_truth, predicted)

    assert metrics_helper.accuracy == .6


def test_calculate_metrics_line_items_wrong_values():
    ground_truth = [{
        'Line Items': [
            {'Total': 10, 'VAT': 1},
            {'Total': 20, 'VAT': 1},
            {'Total': 30, 'VAT': 1},
        ]
    }]
    predicted = [{
        'Line Items': [
            {'Total': 10, 'VAT': 1},
            {'Total': 20, 'VAT': 1},
            {'Total': 30, 'VAT': 2},
        ]
    }]

    metrics_helper = MetricsHelper()
    metrics_helper.compare_true_pred(ground_truth, predicted)

    assert metrics_helper.accuracy == 5/6


def test_calculate_metrics_complete_json():
    ground_truth = [{
        "Document Type": "BILL",
        "VAT": "7.70",
        "Total": "46.20",
        "VAT %": 20,
        "Category": "Purchases",
        "Currency": "GBP",
        "Discount Total": None,
        "Payment Status": "AWAITING_PAYMENT",
        "Service Charge": None,
        "Delivery Charge": None,
        "VAT Exclusive": True,
        "Supplier": "HGV Direct Ltd.",
        "Invoice ID": "INV705501",
        "Line Items": [
            {
                "VAT": "7.70",
                "VAT %": 20,
                "Total": "38.50",
                "Category": "Purchases",
                "Quantity": "1.0",
                "Discount": None,
                "Unit price": "38.50",
                "Description": "SNAKE MUDWING PLASTIC"
            }
        ],
        "VAT Number": "GB 411 1873 85",
        "Date of Invoice": "16-04-2024",
        "Date Payment Due": None,
        "Supplier Address": {
            "Street": "the barn",
            "City": "ashbourne",
            "Postal Code": "DE61HD",
            "Country": None
        },
        "Billing Address": {
            "Street": "sturwood uk ltd",
            "City": "ashbourne",
            "Postal Code": "DE62EF",
            "Country": None
        },
        "Delivery Address": None,
        "Bank Details": [
            {
                "Company Name": "",
                "Account Number": "",
                "Sort Code": "",
                "Bank Name": "",
                "Bank Number": "",
                "IBAN": "",
                "SWIFT Code": "",
                "Account Type": ""
            }
        ]
    }]

    predicted = [{
        "Document Type": "BILL",
        "VAT": "7.70",
        "Total": "46.20",
        "VAT %": 16.67,
        "Category": None,
        "Currency": "GBP",
        "Discount Total": None,
        "Payment Status": "AWAITING_PAYMENT",
        "Service Charge": None,
        "Delivery Charge": None,
        "VAT Exclusive": True,
        "Supplier": "HGV Direct Ltd",
        "Invoice ID": "INV705501",
        "Line Items": [
            {
                "VAT": None,
                "VAT %": 0,
                "Total": "38.50",
                "Category": None,
                "Quantity": "1.0",
                "Discount": None,
                "Unit price": "38.50",
                "Description": "SNAKE MUDWING PLASTIC"
            }
        ],
        "VAT Number": None,
        "Date of Invoice": "16-04-2024",
        "Date Payment Due": None,
        "Supplier Address": {
            "Street": "the elms mowbray roads anwick industrial estate anwick nr",
            "City": "bourne",
            "Postal Code": "DN173AE",
            "Country": None
        },
        "Billing Address": {
            "Street": "stubwood uk ltd",
            "City": "ashbourne",
            "Postal Code": "DE62EF",
            "Country": None
        },
        "Delivery Address": None,
        "Bank Details": [
            {
                "Company Name": "",
                "Account Number": "01333 246266",
                "Sort Code": "",
                "Bank Name": "",
                "Bank Number": "",
                "IBAN": "",
                "SWIFT Code": "",
                "Account Type": ""
            }
        ]
    }]

    metrics_helper = MetricsHelper()
    metrics_helper.compare_true_pred(ground_truth, predicted)

    assert metrics_helper.accuracy == 14/25