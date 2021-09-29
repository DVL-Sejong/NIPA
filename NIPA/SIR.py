from os.path import join
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error

import os
import pandas as pd
import numpy as np
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt


class Compartment:

    @staticmethod
    def get_infected(country):
        path = join(Path(os.getcwd()), 'dataset', country, f'{country}.csv')
        infected_df = pd.read_csv(path, index_col='regions')
        return infected_df

    @staticmethod
    def get_population(country):
        path = join(Path(os.getcwd()), 'dataset', country, 'population.csv')
        population_df = pd.read_csv(path, index_col='regions')
        return population_df

    @staticmethod
    def get_regions(compartment_df):
        return compartment_df.index.tolist()

    @staticmethod
    def get_dates(compartment_df):
        return compartment_df.columns.to_list()

    @staticmethod
    def get_curing_probs(regions, curing_min=0.01, curing_max=0.1, size=50):
        curing_prob_df = pd.DataFrame(index=regions, columns=[i for i in range(len(regions))])
        curing_prob_df = curing_prob_df.rename_axis('regions')

        for i, region in enumerate(regions):
            np.random.seed(i)
            curing_probs = np.random.uniform(low=curing_min, high=curing_max, size=(size,))
            curing_probs = np.sort(curing_probs)

            for j in range(size):
                curing_prob_df.loc[region, j] = curing_probs[j]

        return curing_prob_df

    @staticmethod
    def get_I_compartment(country, n_train=None, duration=3):
        infected_df = Compartment.get_infected(country)
        population_df = Compartment.get_population(country)

        regions = Compartment.get_regions(infected_df)
        dates = Compartment.get_dates(infected_df)
        I_df = pd.DataFrame(index=regions, columns=dates)

        for region in regions:
            population = population_df.loc[region, 'population']

            for date in dates:
                infected_number = infected_df.loc[region, date]
                i_value = infected_number / population
                I_df.loc[region, date] = i_value

        train_I, test_i = Compartment.split_compartment(I_df, n_train, duration)
        return train_I, test_i

    @staticmethod
    def get_R_compartment(I_region, curing_prob):
        R_region = [0] * len(I_region)

        for i in range(1, len(I_region)):
            prev_R = R_region[i - 1]
            prev_I = I_region[i - 1]
            R_region[i] = prev_R + (curing_prob * prev_I)

        return R_region

    @staticmethod
    def get_S_compartment(I_region, R_resion):
        S_region = [0] * len(I_region)

        for i in range(len(I_region)):
            S_region[i] = 1 - I_region[i] - R_resion[i]

        return S_region

    @staticmethod
    def get_prev_I(I):
        return I.iloc[:, -1:]

    @staticmethod
    def get_prev_R(curing_probs, I):
        regions = Compartment.get_regions(I)

        prev_R = pd.DataFrame()
        r_values = []

        for region in regions:
            curing_prob = curing_probs.loc[0, region]
            I_region = I.loc[region]
            R_region = Compartment.get_R_compartment(I_region, curing_prob)

            r_values.append(R_region[-1])

        prev_R[I.columns[-1]] = pd.Series(r_values, index=I.index)
        return prev_R

    @staticmethod
    def predict_I_compartment(curing_prob, prev_I, prev_R, infection_probs, region):
        not_cured = (1 - curing_prob) * prev_I.loc[region][-1]
        s_value = 1 - prev_I.loc[region][-1] - prev_R

        regions = infection_probs.index.tolist()
        new_infected = 0
        for i in range(len(regions)):
            weight = infection_probs.loc[region, regions[i]]
            i_region_prev = prev_I.loc[regions[i]][-1]
            new_infected += weight * i_region_prev

        i_value = not_cured + (s_value * new_infected)
        return i_value

    @staticmethod
    def predict_R_compartment(curing_prob, prev_I, prev_R):
        r_value = prev_R + (curing_prob * prev_I)
        return r_value

    @staticmethod
    def init_for_pred(I, curing_probs):
        date = datetime.strptime(I.columns[-1], '%Y-%m-%d')
        prev_I = Compartment.get_prev_I(I)
        prev_R = Compartment.get_prev_R(curing_probs, I)

        return date, prev_I, prev_R

    @staticmethod
    def predict_sir(I, curing_probs, infection_probs, duration=10):
        date, prev_I, prev_R = Compartment.init_for_pred(I, curing_probs)
        regions = Compartment.get_regions(I)

        for i in range(duration):
            str_date = (date + timedelta(days=i+1)).strftime('%Y-%m-%d')
            prev_I, prev_R = Compartment.next_sir(str_date, regions, prev_I, prev_R, curing_probs, infection_probs)

        prev_I = prev_I.drop([date.strftime('%Y-%m-%d')], axis=1)
        prev_R = prev_R.drop([date.strftime('%Y-%m-%d')], axis=1)

        return prev_I, prev_R

    @staticmethod
    def next_sir(date, regions, prev_I, prev_R, curing_probs, infection_probs):
        I_values, R_values = [], []

        for region in regions:
            curing_prob = curing_probs.loc[0, region]
            I_region = prev_I.loc[region][-1]
            R_region = prev_R.loc[region][-1]

            new_I = Compartment.predict_I_compartment(curing_prob, prev_I, R_region, infection_probs, region)
            new_R = Compartment.predict_R_compartment(curing_prob, I_region, R_region)
            I_values.append(new_I)
            R_values.append(new_R)
        prev_I[date] = pd.Series(I_values, index=regions)
        prev_R[date] = pd.Series(R_values, index=regions)

        return prev_I, prev_R

    @staticmethod
    def split_compartment(compartment_df, n_train=None, duration=3):
        if n_train is None:
            n_train = len(compartment_df.columns) - duration

        train_df = compartment_df.iloc[:, :n_train]
        test_df = compartment_df.iloc[:, n_train:n_train+duration]

        return train_df, test_df

    @staticmethod
    def plot_train_test_set(train, test, pred=None, duration=3, n_cols=2):
        regions = Compartment.get_regions(train)
        dates = Compartment.get_dates(train) + Compartment.get_dates(test)

        n_rows = len(regions) // n_cols
        if len(regions) % n_cols != 0: n_rows += 1

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 25))
        fig.tight_layout(h_pad=3, w_pad=5)
        locator = mticker.MultipleLocator(10)

        for i, region in enumerate(regions):
            train_cases = train.loc[region].to_list() + [np.nan for j in range(duration)]
            test_cases = [np.nan for j in range(len(train_cases) - duration)] + test.loc[region].to_list()
            axs[i // n_cols, i % n_cols].plot(dates, train_cases, label='train')
            axs[i // n_cols, i % n_cols].plot(dates, test_cases, label='test')

            if pred is not None:
                pred_cases = [np.nan for j in range(len(train_cases) - duration)] + pred.loc[region].to_list()
                axs[i // n_cols, i % n_cols].plot(dates, pred_cases, label='pred')

            axs[i // n_cols, i % n_cols].legend(loc='upper left')

        for i, ax in enumerate(axs.flat):
            ax.set(xlabel='date', ylabel='infected cases')
            ax.xaxis.set_major_locator(locator)

            if i < len(regions):
                ax.title.set_text(regions[i])

    @staticmethod
    def plot_test_pred_set(test, pred, n_cols=2):
        regions = Compartment.get_regions(test)
        dates = Compartment.get_dates(test)

        n_rows = len(regions) // n_cols
        if len(regions) % n_cols != 0: n_rows += 1

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 25))
        fig.tight_layout(h_pad=3, w_pad=5)
        locator = mticker.MultipleLocator(10)

        for i, region in enumerate(regions):
            pred_in_region = pred.loc[region].to_list()
            true_in_region = test.loc[region].to_list()

            axs[i // n_cols, i % n_cols].plot(dates, pred_in_region, label='pred')
            axs[i // n_cols, i % n_cols].plot(dates, true_in_region, label='true')
            axs[i // n_cols, i % n_cols].set_title(regions[i])
            axs[i // n_cols, i % n_cols].legend(loc='lower left')

        for ax in axs.flat:
            ax.set(xlabel='date', ylabel='Infected')

        plt.legend()

    @staticmethod
    def plot_sir(S_region, I_region, R_region, region=None, dates=None):
        fig = plt.figure()

        if region is not None:
           fig.suptitle(region)

        if dates is None:
            dates = [i for i in range(len(S_region))]

        ax = fig.add_subplot(111)
        locator = mticker.MultipleLocator(10)

        ax.plot(dates, S_region, 'b', label='Susceptible')
        ax.plot(dates, I_region, 'r', label='Infected')
        ax.plot(dates, R_region, 'g', label='Recovered')
        ax.xaxis.set_major_locator(locator)
        ax.set_xlabel('Time/days')
        ax.set_ylabel('Number')
        plt.legend(loc='upper right')
        plt.show()

    @staticmethod
    def plot_mse_by_dates(test, pred):
        dates = Compartment.get_dates(test)
        mse_list = []

        for i, date in enumerate(dates):
            true_values = test.iloc[:, i].to_list()
            pred_values = pred.iloc[:, i].to_list()
            mse_list.append(mean_squared_error(true_values, pred_values))

        plt.plot(dates, mse_list)
        plt.suptitle('MSE by dates')
        plt.xlabel('dates')
        plt.ylabel('MSE')
        plt.show()

    @staticmethod
    def plot_mse_by_regions(test, pred):
        regions = Compartment.get_regions(test)
        mse_list = []

        for i, region in enumerate(regions):
            true_values = test.loc[region].to_list()
            pred_values = pred.loc[region].to_list()
            mse_list.append(mean_squared_error(true_values, pred_values))

        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.bar(regions, mse_list)
        plt.xticks(rotation=90)
        plt.show()


if __name__ == '__main__':
    loader = Compartment()
    train_I, test_I = loader.get_I_compartment('test', n_train=90, duration=10)
    loader.plot_train_test_set(train_I, test_I, duration=10)
