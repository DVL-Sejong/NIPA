from NIPA.datatype import Country, PreprocessInfo, DatasetInfo
from NIPA.io import load_links, save_setting, load_dataset, save_parameters, save_results
from NIPA.loader import DataLoader
from NIPA.nipa import train_nipa, predict
from NIPA.util import get_predict_period_from_dataset

import argparse
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(description='test')

    parser.add_argument(
        "--country", type=str, default='italy',
        choices=['italy', 'india', 'us', 'china'],
        help="Country name"
    )

    parser.add_argument(
        "--standardization", type=bool, default=False,
        help="standardization"
    )

    parser.add_argument(
        "--x_frames", type=int, default=15,
        help="Number of x frames for generating dataset"
    )

    parser.add_argument(
        "--y_frames", type=int, default=3,
        help="Number of x frames for generating dataset"
    )

    args = parser.parse_args()
    return args


def main(args):
    country = Country[args.country.upper()]
    link_df = load_links(country)
    sird_info = PreprocessInfo(country=country, start=link_df['start_date'], end=link_df['end_date'],
                               increase=True, daily=True, remove_zero=True,
                               smoothing=True, window=5, divide=True)
    save_setting(sird_info, 'sird_info')

    dataset = load_dataset(country, sird_info.get_hash())

    predict_dates = get_predict_period_from_dataset(dataset, args.x_frames, args.y_frames)
    data_info = DatasetInfo(x_frames=args.x_frames, y_frames=args.y_frames,
                            test_start=predict_dates[0], test_end=predict_dates[-1])
    save_setting(data_info, 'data_info')

    loader = DataLoader(data_info, dataset)

    train_I, test_I, test_dates = loader[0]
    regions = train_I.index.tolist()
    I_df = pd.DataFrame(index=regions, columns=predict_dates)
    I_df.index.name = 'regions'
    R_df = pd.DataFrame(index=regions, columns=predict_dates)
    R_df.index.name = 'regions'

    for i in range(len(loader)):
        train_I, test_I, test_dates = loader[i]
        dataset.update({'train': train_I})
        dataset.update({'test': test_I})

        print(f'Predicting {test_dates[0]} to {test_dates[-1]} ===============================')
        curing_df, B_df = train_nipa(dataset, args.standardization)
        save_parameters(country, sird_info, data_info, test_dates[0], curing_df, B_df)
        pred_I_df, pred_R_df = predict(dataset, curing_df, B_df)
        I_df.loc[:, test_dates[0]] = pred_I_df.loc[:, test_dates[0]]
        R_df.loc[:, test_dates[0]] = pred_R_df.loc[:, test_dates[0]]
        save_results(country, sird_info, data_info, test_dates[0], pred_I_df, pred_R_df)

    save_results(country, sird_info, data_info, None, I_df, R_df)


if __name__ == '__main__':
    args = get_args()
    main(args)
