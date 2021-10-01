from NIPA.io import load_dataset, save_parameters, save_results, get_fig_path
from NIPA.plotting import plot_multiple_graph

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
from numpy import linalg as LA

import pandas as pd
import numpy as np
import time

np.random.rand(0)


def generate_curing_probabilities(low=0.01, high=1, size=(150,)):
    curing_probs = np.random.uniform(low=low, high=high, size=size)
    return curing_probs


def get_R_df(I_df, curing_df):
    R_df = pd.DataFrame(index=I_df.index, columns=I_df.columns)

    for region in I_df.index:
        curing_prob = curing_df.loc[region, 'curing_prob']
        I_region = I_df.loc[region, :].to_numpy()
        R_df.loc[region, :] = get_R_region(curing_prob, I_region)

    return R_df


def get_R_region(curing_prob, I_region):
    R_region = np.zeros_like(I_region)

    for i in range(len(I_region) - 1):
        R_region[i+1] = R_region[i] + (curing_prob * I_region[i])

    return R_region


def get_S_region(I_region, R_region):
    S_region = 1 - I_region - R_region
    return S_region


def get_region_V(curing_prob, I_region):
    v_list = []
    for i in range(len(I_region) - 1):
        v_elem = I_region[i + 1] - ((1 - curing_prob) * I_region[i])
        v_list.append(v_elem)

    v_np = np.asarray(v_list)
    v_np = v_np.reshape(-1, 1)
    return v_np


def get_region_F(S_region, I_df):
    regions = I_df.index.tolist()

    f_list = []
    for i in range(len(S_region) - 1):
        row = []
        for j, region in enumerate(regions):
            elem = S_region[i] * I_df.iloc[j, i]
            row.append(elem)
        f_list.append(row)

    f_np = np.asarray(f_list)
    return f_np


def standardize(F_region, V_region):
    F_transposed = np.transpose(F_region)
    V_transposed = np.transpose(V_region)

    scaler = StandardScaler()
    scaler = scaler.fit(F_transposed)

    F_transposed = scaler.transform(F_transposed)
    V_transposed = scaler.transform(V_transposed)

    F_region = np.transpose(F_transposed)
    V_region = np.transpose(V_transposed)

    V_region = V_region.reshape((-1,))

    return F_region, V_region


def network_inference(curing_prob, vector, I_df, standardization):
    S_region, I_region, R_region = vector

    V_region = get_region_V(curing_prob, I_region)
    F_region = get_region_F(S_region, I_df)

    F_transposed = np.transpose(F_region)
    F_tV = np.dot(F_transposed, V_region)

    p_max = 2 * LA.norm(F_tV, np.inf)
    p_min = 0.0001 * p_max
    p_candidates = np.logspace(p_min, p_max, num=300)

    if standardization:
        F_region, V_region = standardize(F_region, V_region)
    else:
        V_region = V_region.reshape((-1,))

    lasso = LassoCV(alphas=p_candidates, max_iter=100000,
                    copy_X=True, cv=3, n_jobs=-1).fit(F_region, V_region)
    region_B = lasso.coef_
    region_mse = mean_squared_error(V_region, lasso.predict(F_region))

    return region_B, region_mse


def train_nipa(country, standardization):
    dataset = load_dataset(country)
    train_I_df = dataset['train']
    regions = dataset['regions']

    curing_df = pd.DataFrame(index=train_I_df.index, columns=['curing_prob'])
    B_df = pd.DataFrame(index=train_I_df.index, columns=regions)

    curing_probs = generate_curing_probabilities()

    for region in regions:
        print(region, end='')

        B_list = []
        mse_list = []
        curing_list = []

        ts = time.time()
        for curing_prob in curing_probs:
            I_region = train_I_df.loc[region, :].to_numpy()
            R_region = get_R_region(curing_prob, I_region)
            S_region = get_S_region(I_region, R_region)

            anomaly_R = [elem for elem in R_region if elem > 1]

            if len(anomaly_R) > 0:
                continue

            vector = [S_region, I_region, R_region]
            region_B, region_mse = network_inference(curing_prob, vector, train_I_df, standardization)
            B_list.append(region_B)
            mse_list.append(region_mse)
            curing_list.append(curing_prob)

        min_index = mse_list.index(min(mse_list))
        curing_df.loc[region, 'curing_prob'] = curing_list[min_index]
        B_df.loc[region, :] = B_list[min_index]

        te = time.time()
        print(f', took {(te - ts):.4f} sec', end='')
        print(f', curing_prob: {curing_probs[min_index]}')
        print(f'B values: {B_list[min_index]}')
        print()

    save_parameters(country, curing_df, B_df)
    return curing_df, B_df


def predict(country, curing_df, B_df):
    dataset = load_dataset(country)
    train_I_df = dataset['train']
    train_R_df = get_R_df(train_I_df, curing_df)
    test_I_df = dataset['test']
    regions = dataset['regions']

    I_last_day = train_I_df.iloc[:, -1]
    R_last_day = train_R_df.iloc[:, -1]

    pred_start_date = datetime.strptime(test_I_df.columns.to_list()[0], '%Y-%m-%d') + timedelta(days=-1)
    pred_end_date = datetime.strptime(test_I_df.columns.to_list()[-1], '%Y-%m-%d')
    pred_duration = (pred_end_date - pred_start_date).days + 1
    pred_period = [(pred_start_date + timedelta(days=elem)).strftime('%Y-%m-%d') for elem in range(pred_duration)]

    pred_I_df = pd.DataFrame(index=test_I_df.index, columns=pred_period)
    pred_I_df.iloc[:, 0] = I_last_day
    pred_R_df = pd.DataFrame(index=test_I_df.index, columns=pred_period)
    pred_R_df.iloc[:, 0] = R_last_day

    for k, pred_day in enumerate(pred_period):
        if k == 0: continue

        for i, region in enumerate(regions):
            prev_day = pred_period[k - 1]
            curing_prob = curing_df.loc[region, 'curing_prob']

            elem1 = (1 - curing_prob) * pred_I_df.loc[region, prev_day]

            s_elem = 1 - pred_I_df.loc[region, prev_day] - pred_R_df.loc[region, prev_day]
            infect_sum = 0
            for j, region_2 in enumerate(regions):
                infect_elem = B_df.loc[region, region_2] * pred_I_df.loc[region_2, prev_day]
                infect_sum += infect_elem
            elem2 = s_elem * infect_sum

            pred_I_df.loc[region, pred_day] = elem1 + elem2
            pred_R_df.loc[region, pred_day] = pred_R_df.loc[region, prev_day] + (
                        curing_prob * pred_I_df.loc[region, prev_day])

    pred_I_df = pred_I_df.iloc[:, 1:]
    pred_R_df = pred_R_df.iloc[:, 1:]

    save_results(country, pred_I_df, pred_R_df)
    return pred_I_df, pred_R_df


if __name__ == '__main__':
    country = 'italy'
    standardization = False

    dataset = load_dataset(country)
    curing_df, B_df = train_nipa(country, standardization)
    pred_I_df, pred_R_df = predict(country, curing_df, B_df)

    data = [pred_I_df, dataset['test']]
    names = ['pred', 'true']
    plot_multiple_graph(data, names, fig_path=get_fig_path(country), figsize=(15, 20))
