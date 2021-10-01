from NIPA.datatype import get_country_name
from os.path import join, abspath, dirname
from pathlib import Path

import pandas as pd


ROOT_PATH = Path(abspath(dirname(__file__))).parent
DATASET_PATH = join(ROOT_PATH, 'dataset')
SETTING_PATH = join(ROOT_PATH, 'settings')
RESULT_PATH = join(ROOT_PATH, 'results')


def load_population(country):
    population_path = join(DATASET_PATH, get_country_name(country), 'population.csv')
    population_df = pd.read_csv(population_path, index_col='regions')
    return population_df


def load_regions(country):
    population_df = load_population(country)
    regions = population_df.index.tolist()
    return regions


def load_dataset(country):
    population_df = load_population(country)
    regions = load_regions(country)

    data_path = join(DATASET_PATH, get_country_name(country))
    train_df = pd.read_csv(join(data_path, 'train_I.csv'), index_col='regions')
    test_df = pd.read_csv(join(data_path, 'test_I.csv'), index_col='regions')

    dataset = {'population': population_df, 'regions': regions,
               'train': train_df, 'test': test_df}
    return dataset


def get_result_path(country):
    result_path = join(RESULT_PATH, get_country_name(country))
    Path(result_path).mkdir(parents=True, exist_ok=True)
    return result_path


def save_parameters(country, curing_df, B_df):
    result_path = get_result_path(country)

    curing_df.to_csv(join(result_path, 'curing_probs.csv'))
    B_df.to_csv(join(result_path, 'B.csv'))
    print(f'saving parameters under {result_path}')


def save_results(country, pred_I_df, pred_R_df):
    result_path = get_result_path(country)

    pred_I_df.to_csv(join(result_path, 'pred_I.csv'))
    pred_R_df.to_csv(join(result_path, 'pred_R.csv'))
    print(f'saving test results under {result_path}')


def get_fig_path(country):
    return join(get_result_path(country), 'result.png')
