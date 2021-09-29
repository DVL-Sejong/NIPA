from sklearn.metrics import mean_squared_error

import numpy as np


class KFold:
    def __init__(self, n_splits):
        self.n_splits = n_splits

    def split(self, X, y=None):
        fold_size = int(len(X) / self.n_splits)

        indices = np.arange(len(X))
        for i in range(self.n_splits):
            test_index = indices[fold_size * i: fold_size * (i + 1)]
            train_mask = indices[list(np.delete(indices, test_index))]
            test_mask = indices[test_index]
            yield train_mask, test_mask

    def get_dataset(self, X, y):
        for train_mask, test_mask in self.split(X, y):
            yield X[train_mask], X[test_mask], y[train_mask], y[test_mask]


class Lasso:
    def __init__(self, p, region_index, cv=KFold(1), iterations=1000, learning_rate=0.001):
        self.p = p
        self.region_index = region_index
        self.cv = cv
        self.iterations = iterations
        self.learning_rate = learning_rate

    def cross_validate(self, X, y):
        infection_prob_list = []
        mse_list = []

        for X_train, X_test, y_train, y_test in self.cv.get_dataset(X, y):
            self.train_model(X_train, y_train)
            mse = self.evaluate_model(X_test, y_test)
            infection_prob_list.append(self.b)
            mse_list.append(mse)

        min_index = mse_list.index(min(mse_list))
        infection_prob = infection_prob_list[min_index]
        mse = min(mse_list)

        return infection_prob, mse

    def train_model(self, X_train, y_train):
        self.m, self.n = X_train.shape
        self.b = np.zeros(self.n)
        self.X = X_train
        self.Y = y_train

        for i in range(self.iterations):
            self.update_weights()

    def evaluate_model(self, X_test, y_test):
        Y_pred = self.predict(X_test)

        mse = self.get_mse(Y_pred, y_test)
        return mse

    def update_weights(self):
        Y_pred = self.predict(self.X)

        db = np.zeros(self.n)

        for i in range(self.n):
            db[i] = (-2 / self.m) * self.X[:, i].dot(self.Y - Y_pred)

            if self.b[i] > 0:
                self.b[i] -= self.learning_rate * db[i]
            else:
                self.b[i] += self.learning_rate * db[i]

        self.b -= self.learning_rate * db

        return self

    def predict(self, X):
        pred = X.dot(self.b)
        return pred

    def get_mse(self, Y_pred, y_test):
        b = self.b
        b[self.region_index] = 0
        pb = self.p * b
        mse = mean_squared_error(Y_pred, y_test) + sum(pb)

        return mse

# parameter = Parameter()

# for i, reg_param in enumerate(reg_params):
#     model = Lasso(p=reg_param, region_index=region_index, cv=KFold(3))
#     infection_prob, mse = model.cross_validate(F_matrix, V_region)
#     parameter.add_elements(reg_param, infection_prob, mse)
#
# reg_param, infection_probs, mse = parameter.get_min_values()

# return infection_probs, mse


class Parameter:
    def __init__(self):
        self.infection_probs_list = []
        self.reg_param_list = []
        self.mse_list = []

    def add_elements(self, reg_param, infection_probs, mse):
        self.reg_param_list.append(reg_param)
        self.infection_probs_list.append(infection_probs)
        self.mse_list.append(mse)

    def get_min_values(self):
        min_mask = self.mse_list.index(min(self.mse_list))
        reg_param = self.reg_param_list[min_mask]
        infection_probs = self.infection_probs_list[min_mask]
        mse = self.mse_list[min_mask]

        return reg_param, infection_probs, mse
