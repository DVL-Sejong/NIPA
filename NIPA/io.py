from os.path import join, abspath, dirname
from pathlib import Path

import pandas as pd


ROOT_PATH = Path(abspath(dirname(__file__))).parent
DATASET_PATH = join(ROOT_PATH, 'dataset')
SETTING_PATH = join(ROOT_PATH, 'settings')
RESULT_PATH = join(ROOT_PATH, 'results')


def load_population(case_name):
    population_path = join(DATASET_PATH, case_name, 'population.csv')
    population_df = pd.read_csv(population_path, index_col='regions')
    return population_df


def load_regions(case_name):
    population_df = load_population(case_name)
    regions = population_df.index.tolist()
    return regions


def load_dataset(case_name):
    population_df = load_population(case_name)
    regions = load_regions(case_name)

    data_path = join(DATASET_PATH, case_name)
    train_df = pd.read_csv(join(data_path, 'train_I.csv'), index_col='regions')
    test_df = pd.read_csv(join(data_path, 'test_I.csv'), index_col='regions')

    dataset = {'population': population_df, 'regions': regions,
               'train': train_df, 'test': test_df}
    return dataset


def get_result_path(case_name):
    result_path = join(RESULT_PATH, case_name)
    Path(result_path).mkdir(parents=True, exist_ok=True)
    return result_path


def save_parameters(case_name, curing_df, B_df):
    result_path = get_result_path(case_name)

    curing_df.to_csv(join(result_path, 'curing_probs.csv'))
    B_df.to_csv(join(result_path, 'B.csv'))
    print(f'saving parameters under {result_path}')


def save_results(case_name, pred_I_df, pred_R_df):
    result_path = get_result_path(case_name)

    pred_I_df.to_csv(join(result_path, 'pred_I.csv'))
    pred_R_df.to_csv(join(result_path, 'pred_R.csv'))
    print(f'saving test results under {result_path}')


def get_fig_path(case_name):
    return join(get_result_path(case_name), 'result.png')
