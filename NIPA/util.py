from NIPA.io import load_dataset


def get_country_name(country):
    if country.upper() == 'US':
        return country.upper()
    else:
        return country.capitalize()


if __name__ == '__main__':
    country = 'italy'
    dataset = load_dataset(country)
