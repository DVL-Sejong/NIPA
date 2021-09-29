from os.path import join, abspath, dirname
from pathlib import Path

import pandas as pd


def get_country_name(country):
    if country.upper() == 'US':
        return country.upper()
    else:
        return country.capitalize()


def load_dataset(country):
    root_path = Path(abspath(dirname(__file__))).parent
    data_path = join(root_path, 'dataset', get_country_name(country))

    population_df = pd.read_csv(join(data_path, 'population.csv'), index_col='regions')
    train_df = pd.read_csv(join(data_path, 'train_I.csv'), index_col='regions')
    test_df = pd.read_csv(join(data_path, 'test_I.csv'), index_col='regions')
    regions = population_df.index.to_list()

    dataset = {'population': population_df, 'regions': regions,
               'train': train_df, 'test': test_df}
    return dataset


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


def get_result_path(country):
    root_path = Path(abspath(dirname(__file__))).parent
    result_path = join(root_path, 'results', get_country_name(country))
    Path(result_path).mkdir(parents=True, exist_ok=True)
    return result_path


def get_fig_path(country):
    fig_path = join(get_result_path(country), 'result.png')
    return fig_path


if __name__ == '__main__':
    country = 'italy'
    dataset = load_dataset(country)
