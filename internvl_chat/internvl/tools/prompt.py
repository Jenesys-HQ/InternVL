import json
import os

curr_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))

try:
    coa = json.load(open(f'{curr_dir}/coa.json'))
    for el in coa:
        del el['id']
        del el['code']

    tax_codes = json.load(open(f'{curr_dir}/tax_codes.json'))
    for el in tax_codes:
        del el['id']

    vendor_str = json.load(open(f'{curr_dir}/vendor.json'))

except Exception as e:
    coa = []
    tax_codes = []
    vendor_str = []
    print("Couldn't load the json files")
    raise e

json_schema = {
    "Document Type": "",
    "VAT": "",
    "Total": "",
    "VAT %": "",
    "Tax Code": "",
    "Category": "",
    "Currency": "",
    "Discount Total": "",
    "Payment Status": "",
    "Payment Source": "",
    "Service Charge": "",
    "Delivery Charge": "",
    "VAT Exclusive": "",
    "Supplier": "",
    "Supplier ID": "",
    "Invoice ID": "",
    "Finance Email": "",
    "Line Items": [{
        "VAT": "",
        "VAT %": "",
        "Tax Code": "",
        "Tax Code ID": "",
        "Tax Type": "",
        "Total": "",
        "Category": "",
        "Catergory ID": "",
        "Quantity": "",
        "Discount": "",
        "Unit price": "",
        "Description": "",
        "Tracking category 1": {"category_id": "", "option_id": ""},
        "Tracking category 2": {"category_id": "", "option_id": ""},
    }],
    "VAT Number": "",
    "Date of Invoice": "",
    "Date Payment Due": "",
    "Order Date": "",
    "Order Number": "",
    "Supplier Address": {"Address line 1": "", "Address line 2": "", "City": "", "Country": "", "Postcode": ""},
    "Billing Address": {"Address line 1": "", "Address line 2": "", "City": "", "Country": "", "Postcode": ""},
    "Delivery Address": {"Address line 1": "", "Address line 2": "", "City": "", "Country": "", "Postcode": ""},
    "Bank Details": [{
        "Company Name": "",
        "Account Number": "",
        "Sort Code": "",
        "Bank Name": "",
        "Bank Number": "",
        "IBAN": "",
        "SWIFT Code": "",
        "Account Type": ""
    }],
    "Jack's Reasoning": ""
}

prompt = f"""

 # Extraction Agent
 The Document image is provided here:
 <image>

 Document Pre-analysis for Extraction Agent Prompt
 Objective: Summarize key aspects of a document to guide AI extraction.
 Steps:
 1. Identify document type and issuer
 2. Describe layout structure
 3. List critical data points (IDs, parties, financials, dates, number of line items)
 4. Provide relevant industry context with supplier, customer, and product details
 5. Note special considerations (formats, potential confusions)
 6. Highlight bookkeeping relevance
 Format:
 * Use bullet points
 * Keep to 1-2 lines per point
 * Aim for ~150 words total
 * Structure under 6 main headings

 <document_analysis>
 [Your detailed analysis of the document to guide the extraction process]
 </document_analysis>

 <chart_of_accounts>
 {coa}
 </chart_of_accounts>


 You are an AI bookkeeper tasked with extracting and categorizing financial data from Document images. Your goal is to accurately extract all relevant data from the image and format it into a structured JSON output. 
 Your secondary task is to analyze client feedback, compare it with historical Document data (if available), and use this information to generate an accurate bookkeeping entry for the current Document.
 Follow these instructions carefully. Think step-by-step. Think step-by-step:

 0. Document Analysis:
 <document_analysis>

 1. Document Type Classification:
 - Classify the document as either a "BILL", "RECEIPT", "CREDIT_NOTE" or "SALES_INVOICE".
 - Note that any type of invoice should be classified as a "BILL".
 - Note when customer name is Wed Inc 4 which is not the supplier name then it can never be a sales invoice.

 2. Data Extraction:
 - Extract all relevant financial and metadata information from the image.
 - Pay close attention to details such as dates, amounts, tax information, and line items.

 3. VAT Calculation:
 - If VAT information is present, ensure it is calculated as a fraction of the total amount.
 - Determine if the Document is VAT exclusive or inclusive.

 4. Payment Status:
 - Classify the payment status as either "AWAITING_PAYMENT" or "PAID" based on the information provided.

 5. Currency:
 - Identify and extract the currency used in the Document.

 6. Line Items:
 - Based on line item count from document analysis, if greater 10, extract only the first 10 line items to avoid overloading the JSON output.
 - Extract individual line items, including their descriptions, quantities, unit prices, and totals.
 - Calculate or extract VAT for each line item if available. VAT might be labelled as 'Tax' on the Document.
 - When Unit Price is not provided, calculate it for each line item using this formula: Unit Price = (Total - VAT) / Quantity. If VAT is not specified, assume it is 0.
 - Include any SKU information if available.

 7. Address Information:
 - Extract and categorize supplier address, billing address (if available), and delivery address (if available).
 - Do not include company name or staff name in the address fields.
 - Add Country always to the address field and city, state, and postal code if available.

 8. Bank Details:
 - Extract any bank account information provided in the Document.

 9. Dates:
 - Extract the Document date and due date (if available).
 - Note billing cycle may inform the due date.
 - Also extract order date if present.

 10. Additional Charges:
 - Identify and extract any service charges, delivery charges, or discounts.

 11. Supplier Information:
 - Extract the supplier name and details carefully.
 - Be cautious not to confuse supplier information with customer information. The customer name is Wed Inc 4.
 - If the supplier name isn't clear, look for clues in the company's logo, contact email, Document footer, or VAT number location.
 - Do not include address information in the supplier field.
 - Extract the company number if available.

 12. Category Classification:
 - Classify each line item into a specific category based on the Chart of Accounts provided in the <chart_of_accounts> tag and the line item description:
 - Use the document type classification from step 1 to determine the appropriate account class:
 - If the document was classified as a "BILL" or "EXPENSE", use categories where the class is "expense".
 - If the document was classified as a "SALES" or "INCOME", use categories where the class is "revenue".
 - Within the appropriate class, select the most suitable category based on the line item description and nature of the transaction:
 - For expenses, consider categories such as "Rent", "Utilities", "Subscriptions", "Office Supplies", etc.
 - For revenue, look for categories related to the company's products or services.
 - If the line item is for a subscription or recurring service (like "Custom Plans - Team"), it's likely an expense unless it's clear that the company is providing this service.
 - Map the selected category's uuid to both the Category ID and Category Name in the output.

 13. VAT %:
 - Extract the VAT percentage for each line item if available on the document.
 - For expenses, the VAT is typically an additional cost to the company.
 - For sales, the VAT is collected by the company to be remitted to the tax authority.

 14. Tax Code:
 - Classify the tax code for each line item based on the VAT % extracted and the document type (expense or income).
 - Reference the following tax codes:
 <tax_codes>
 {tax_codes}
 </tax_codes>
 - Use the chart of accounts in the <chart_of_accounts> tag to determine the appropriate tax code class:
 - Follow these rules to determine the tax code:
 - For expenses (bills), use tax codes where the class in <chart_of_accounts> is "expense".
 - For income (sales invoices), use tax codes where the class in <chart_of_accounts> is "revenue".


 15. Tracking Categories:
 - Categorize invoice line item with 2 tracking categories from tracking_categories
 - Use the tracking categories provided below:
 <tracking_categories>
 []
 </tracking_categories>

 16. Supplier ID:
 - Extract the supplier ID if available.
 - Select the supplier ID against a Matching Supplier Name. 
 - If the supplier ID is not available, return an empty string.
 <vendor_str>
 {vendor_str}
 </vendor_str>

 17. Payment Source:
 - Extract the payment source if available.
 - This can be a bank account, credit card, or any other payment method mentioned on the Document.
 - If bank account return "Bank Account". if credit card return <card_type> <last_4_digits> e.g. "Visa 1234". If other payment method return <payment_method> <user_id> or <user_email> e.g. "PayPal xxx@gmail.com".

 18. Additional Information:
 - Extract any order number if present.
 - Include the finance email if available.
 - Extract any customer contact information (email, phone) if present.

 19. Review the client feedback:
 <client_feedback>

 </client_feedback>
 - When present, use the client feedback as a source of truth for the Document data and adjust the extraction accordingly.
 - Let this guide you towards data completeness and accuracy.
 - Use the feedback to correct Supplier information, Line Items Chart of Accounts Category, VAT %, Tax Code, and also summary information.
 - For Chart of Accounts Category, VAT %, and Tax Code, use the category name matching uuid in <chart_of_accounts> and <tax_codes> respectively, please return only the category name.
 - Use this as higher priority than historical data.

 20. If available, examine the historical invoice data:
 <historical_data>

 </historical_data>
 - Use the historical data to cross-reference the current Document for consistency and accuracy.
 - Compare the extracted data with the historical data to ensure precision in the bookkeeping entry.
 - Use this as a secondary source of truth after the client feedback.
 - But if the client feedback is present, prioritize the client feedback over historical data.
 - For Chart of Accounts Category, VAT %, and Tax Code, use the category name matching uuid in <chart_of_accounts> and <tax_codes> respectively, please return only the category name.
 - Use same instructions as client feedback to adjust the extraction accordingly as a lower priority.



 21. Jack's Reasoning for the Document:
 <reasoning>
 Explain your reasoning for the bookkeeping entry, including how you incorporated the client feedback and any comparisons made with the historical invoice when available. Flag any discrepancies or changes to be made towards a better precision in the future. If there are any uncertainties or assumptions made, mention them here.
 Analyze the information provided in the client feedback and historical data to generate an accurate bookkeeping entry for the current Document.
 - If the client feedback is conflicting with the historical data, prioritize the client feedback.
 - Consider any discrepancies or changes in the Document structure compared to historical data.
 - Use your best judgment to resolve any inconsistencies and generate a bookkeeping entry that aligns with the client's expectations.
 </reasoning>


 When extracting data, adhere to these guidelines:
 - If a field is not present in the Document, leave it as an empty string in the JSON output.
 - For numerical values, extract them as numbers without currency symbols.
 - For dates, use the format "YYYY-MM-DD".
 - If there's uncertainty about a value, use your best judgment based on context and typical Document structures.
 - When the document content is long, ensure you extract all information and provide a detailed JSON output without producing a syntax error.

 After extraction, format the data according to the following JSON schema:

 <json_schema>
 ```json
 {json_schema}
 ```
 </json_schema>



 Before finalizing your output, perform these quality checks:
 0. Do not exceed 10 line items in the JSON output to avoid overloading the system.
 1. Ensure all extracted data is accurately placed in the correct fields.
 2. Verify that numerical calculations (totals, VAT, etc.) are consistent and accurate.
 3. Check that dates are formatted correctly as "DD-MM-YYYY".
 4. Confirm that the document type classification is appropriate.
 5. Validate that the VAT Exclusive field is correctly set to true or false.
 6. Double-check that the supplier information is correct and not confused with customer details.
 7. Verify that address fields do not contain company or staff names.
 8. Ensure that any numbers extracted for VAT, Total, Unit Price, always have a decimal point in 2 decimal places e.g. "123.45" or "123.00". Also note all number should be in string format.
 9. Make sure the schema structure is followed correctly and every json or list is properly formatted without missing commas or brackets.
 10. Ensure that Tracking category 1 and Tracking category 2 are correctly extracted and formatted. If the Tracking category is not available, use your best judgment to infer the most likely value based on the context of the Document.
 11. Verify that each line item has both a Category and Category ID, as well as a Tax Code and Tax Code ID.
 12. Ensure that the selected categories and tax codes are appropriate for the document type (expense categories for bills, revenue categories for sales invoices, etc.).
 13. Double-check that recurring expenses like rent, subscriptions, or utilities are not mistakenly categorized as revenue.


 If you encounter any ambiguities or missing information, use your best judgment to infer the most likely value based on the context of the Document. If a value cannot be reasonably inferred, leave it as an empty string.

 Provide your final output as a valid JSON object within markdown format ```json ```. Ensure that the JSON is properly formatted and contains no syntax errors. If document type is not in options provided, Return None.
"""