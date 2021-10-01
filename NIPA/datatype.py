from enum import Enum


def get_country_name(country):
    if country == Country.US:
        return country.name.upper()
    else:
        return country.name.capitalize()


class Country(Enum):
    CHINA = 0
    INDIA = 1
    ITALY = 2
    US = 3
