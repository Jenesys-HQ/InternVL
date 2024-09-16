import logging

import pyap
import pycountry
from typing import Optional, Union, Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    from postal import parser as postal_parser
except ImportError:
    postal_parser = None
    logger.error("Error importing postal parser")

# COUNTRY_CODES = {
#     country.alpha_2 for country in pycountry.countries
# }.union({
#     country.alpha_3 for country in pycountry.countries
# })


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

        if country is None:
            return None

        return country.alpha_2
    except LookupError as e:
        logger.warning(e)
        return None


def extract_country_from_address(address: str):
    last_component = address.split(',')[-1].strip()

    if last_component is None:
        return None

    return get_country_code(last_component)


class AddressExtractor:
    pyap_address = None

    def __init__(self, address_str: str):
        self.address_str = address_str
        self.country_code = extract_country_from_address(self.address_str)

        print(f'Country code: {self.country_code}')

        if self.country_code:
            self.pyap_address = pyap.parse(self.address_str, country=self.country_code)[0]

        if postal_parser:
            self.postal_address = {k: v for v, k in postal_parser.parse_address(self.address_str)}

    def extract_address_line_1(self) -> Optional[str]:
        address_line_1 = ' '
        if self.pyap_address:
            if self.pyap_address.street_number:
                address_line_1 += self.pyap_address.street_number + ' '
            if self.pyap_address.street_name:
                address_line_1 += self.pyap_address.street_name + ' '
            if self.pyap_address.street_type:
                address_line_1 += self.pyap_address.street_type + ' '

        if address_line_1 is None and postal_parser:
            if 'house' in self.postal_address:
                address_line_1 = f"{self.postal_address['house']}"
            elif 'house_number' in self.postal_address:
                address_line_1 = f"{self.postal_address['house_number']}"
                if 'road' in self.postal_address:
                    address_line_1 += f", {self.postal_address['road']}"

        return address_line_1.strip()

    def extract_address_line_2(self) -> Optional[str]:
        address_line_2 = None
        if self.pyap_address:
            address_line_2 = self.pyap_address.occupancy

        if address_line_2 is None and postal_parser:
            if 'unit' in self.postal_address:
                address_line_2 = self.postal_address['unit']
            elif 'level' in self.postal_address:
                address_line_2 = self.postal_address['level']
            elif 'staircase' in self.postal_address:
                address_line_2 = self.postal_address['staircase']

        return address_line_2

    def extract_city(self) -> Optional[str]:
        city = None
        if self.pyap_address:
            city = self.pyap_address.city

        if city is None and postal_parser:
            city = self.postal_address.get('city', None)

        return city

    def extract_postcode(self) -> Optional[str]:
        postcode = None
        if self.pyap_address:
            postcode = self.pyap_address.postal_code

        if postcode is None and postal_parser:
            postcode = self.postal_address.get('postcode', None)

        if postcode:
            postcode = postcode.replace(' ', '').upper()

        return postcode

    def extract_country(self) -> Optional[str]:
        country = None
        if self.pyap_address:
            if hasattr(self.pyap_address, 'country'):
                country = self.pyap_address.country
            elif hasattr(self.pyap_address, 'country_id'):
                country = self.pyap_address.country_id

        if country is None and postal_parser:
            country = self.postal_address.get('country', None)

        return country

    def extract_address(self) -> Dict[str, str]:
        return {
            "Address Line 1": self.extract_address_line_1(),
            "Address Line 2": self.extract_address_line_2(),
            "City": self.extract_city(),
            "Postcode": self.extract_postcode(),
            "Country": self.extract_country()
        }


def standardise_address(address: Optional[Union[str, dict]]) -> Optional[Dict[str, str]]:
    if address is None:
        return None

    if type(address) is dict:
        return address

    address = address.replace('\n', ', ')

    address_extractor = AddressExtractor(address)

    return address_extractor.extract_address()
