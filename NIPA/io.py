from os.path import join, abspath, dirname, isfile
from pathlib import Path
from dataclasses import fields

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
    I_df = pd.read_csv(join(data_path, 'I.csv'), index_col='regions')

    dataset = {'population': population_df, 'regions': regions, 'I': I_df}
    return dataset


def get_result_path(data_info, case_name, index):
    result_path = join(RESULT_PATH, case_name, data_info.get_hash(), str(index).zfill(2))
    Path(result_path).mkdir(parents=True, exist_ok=True)
    return result_path


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


def save_parameters(data_info, case_name, index, curing_df, B_df):
    result_path = get_result_path(data_info, case_name, index)

    curing_df.to_csv(join(result_path, 'curing_probs.csv'))
    B_df.to_csv(join(result_path, 'B.csv'))
    print(f'saving parameters under {result_path}')


def save_results(data_info, case_name, index, pred_I_df, pred_R_df):
    result_path = get_result_path(data_info, case_name, index)

    pred_I_df.to_csv(join(result_path, 'pred_I.csv'))
    pred_R_df.to_csv(join(result_path, 'pred_R.csv'))
    print(f'saving test results under {result_path}')


def get_fig_path(data_info, case_name, index):
    return join(get_result_path(data_info, case_name, index), 'result.png')
