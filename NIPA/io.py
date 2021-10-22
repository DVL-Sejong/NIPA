from NIPA.datatype import get_country_name
from os.path import join, abspath, dirname, isfile
from pathlib import Path
from dataclasses import fields

import pandas as pd


ROOT_PATH = Path(abspath(dirname(__file__))).parent
DATASET_PATH = join(ROOT_PATH, 'dataset')
SETTING_PATH = join(ROOT_PATH, 'settings')
RESULT_PATH = join(ROOT_PATH, 'results')


def load_links(country=None):
    link_path = join(DATASET_PATH, 'links.csv')
    link_df = pd.read_csv(link_path, index_col='country')
    return link_df.loc[get_country_name(country), :] if country is not None else link_df


def load_population(country):
    population_path = join(DATASET_PATH, get_country_name(country), 'population.csv')
    population_df = pd.read_csv(population_path, index_col='regions')
    return population_df


def load_regions(country):
    population_df = load_population(country)
    regions = population_df.index.tolist()
    regions.sort()
    return regions


def load_dataset(country, sird_hash):
    population_df = load_population(country)
    regions = load_regions(country)

    i_path = join(DATASET_PATH, get_country_name(country), 'I', sird_hash)
    I_df = pd.read_csv(join(i_path, 'I.csv'), index_col='regions')

    dataset = {'population': population_df, 'regions': regions, 'I': I_df}
    return dataset


def get_result_path(country, sird_info, data_info, test_start):
    result_hash = f'{sird_info.get_hash()}_{data_info.get_hash()}'
    result_path = join(RESULT_PATH, get_country_name(country), result_hash)
    if test_start is not None:
        result_path = join(result_path, test_start)
    Path(result_path).mkdir(parents=True, exist_ok=True)
    return result_path


def save_parameters(country, sird_info, data_info, test_start, curing_df, B_df):
    result_path = get_result_path(country, sird_info, data_info, test_start)
    curing_df.to_csv(join(result_path, 'curing_probs.csv'))
    B_df.to_csv(join(result_path, 'B.csv'))
    print(f'saving parameters under {result_path}')


def save_results(country, sird_info, data_info, test_start, pred_I_df, pred_R_df):
    result_path = get_result_path(country, sird_info, data_info, test_start)
    pred_I_df.to_csv(join(result_path, 'pred_I.csv'))
    pred_R_df.to_csv(join(result_path, 'pred_R.csv'))
    print(f'saving test results under {result_path}')


def save_setting(param_class, class_name):
    new_param_dict = dict()
    new_param_dict.update({'hash': param_class.get_hash()})

    for field in fields(param_class):
        if field.name[0] == '_': continue
        new_param_dict.update({field.name: getattr(param_class, field.name)})

    param_df = pd.DataFrame(columns=list(new_param_dict.keys()))
    param_df = param_df.append(new_param_dict, ignore_index=True)
    param_df = param_df.set_index('hash')

    filename = f'{class_name}.csv'
    if isfile(join(SETTING_PATH, filename)):
        df = pd.read_csv(join(SETTING_PATH, filename), index_col='hash')
        if param_class.get_hash() not in df.index.tolist():
            df = param_df.append(df, ignore_index=False)
            df.to_csv(join(SETTING_PATH, filename))
            print(f'updating settings to {join(SETTING_PATH, filename)}')
    else:
        param_df.to_csv(join(SETTING_PATH, filename))
        print(f'saving settings to {join(SETTING_PATH, filename)}')
