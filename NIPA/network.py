from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

from NIPA.SIR import Compartment


class Network:

    @staticmethod
    def get_V_compartment(curing_prob, I_region):
        V_region = []

        for i in range(len(I_region) - 1):
            V_region.append(I_region[i + 1] - ((1 - curing_prob) * I_region[i]))

        return np.array(V_region)

    @staticmethod
    def get_F_compartment(S_region, I):
        dates = I.columns.to_list()
        regions = I.index.tolist()

        F_matrix = []

        for day in range(len(dates) - 1):
            row = []

            for i, region in enumerate(regions):
                row.append(S_region[day] * I.loc[region, dates[day]])

            F_matrix.append(row)

        return np.array(F_matrix)

    @staticmethod
    def network_inference(curing_probs, I, region):
        infection_probs_list, mse_list = [], []

        for curing_prob in curing_probs:
            I_region = I.loc[region].to_list()
            R_region = Compartment.get_R_compartment(I_region, curing_prob)
            S_region = Compartment.get_S_compartment(I_region, R_region)
            vector = [S_region, I_region, R_region]

            infection_probs, mse = Network.estimate_infection_probs(curing_prob, vector, I)
            infection_probs_list.append(infection_probs)
            mse_list.append(mse)

        min_mask = mse_list.index(min(mse_list))
        infection_probs = infection_probs_list[min_mask]

        return infection_probs, curing_probs[min_mask]

    @staticmethod
    def estimate_infection_probs(curing_prob, vector, I):
        S_region, I_region, R_region = vector

        V_region = Network.get_V_compartment(curing_prob, I_region)
        F_matrix = Network.get_F_compartment(S_region, I)

        lasso = LassoCV(eps=1e-4, max_iter=1000000, tol=1e-4, copy_X=True, cv=3, n_jobs=-1).fit(F_matrix, V_region)
        infection_probs = lasso.coef_
        mse = mean_squared_error(V_region, lasso.predict(F_matrix))
        print(lasso.n_iter_, lasso.alpha_)

        return infection_probs, mse





