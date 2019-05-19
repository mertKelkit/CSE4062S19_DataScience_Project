import numpy as np
import pandas as pd

from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class Experimenter:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def test_linear_regression(self, method='train_test_split', test_size=0.3, k=10, random_state=123):

        X = self.df.iloc[:, 1:-1].values
        y = self.df.iloc[:, -1].values

        if method == 'train_test_split':

            regressor = LinearRegression()

            print('- Starting to train linear regression model with test size {} and random state {}...\n'
                  .format(test_size, random_state))

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            regressor.fit(X_train, y_train)
            preds = regressor.predict(X_test)

            msqe, mabse, r2 = mean_squared_error(y_test, preds), \
                              mean_absolute_error(y_test, preds), \
                              r2_score(y_test, preds)

            print('- Mean squared error of linear regression model is {}.'
                  .format(msqe))
            print('- Mean absolute error of linear regression model is {}.'
                  .format(mabse))
            print('- R squared score of linear regression model is {}.'
                  .format(r2))
            print('\n\n')

            return regressor, msqe, mabse, r2

        elif method == 'k_fold':

            kfold = KFold(n_splits=k, random_state=random_state, shuffle=True)

            print('- Starting to train linear regression model with kfold with k={} and random state {}...\n'
                  .format(k, random_state))

            iteration = 1

            models = []

            msqes = []
            mabses = []
            r2_scores = []

            for train_index, test_index in kfold.split(X):

                regressor = LinearRegression()

                print('- Starting iteration {}...'.format(iteration))
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                regressor.fit(X_train, y_train)
                preds = regressor.predict(X_test)

                models.append(regressor)

                msqe = mean_squared_error(y_test, preds)
                mabse = mean_absolute_error(y_test, preds)
                r2 = r2_score(y_test, preds)

                msqes.append(msqe)
                mabses.append(mabse)
                r2_scores.append(r2)

                print('- Mean squared error of linear regression model is {}.'
                      .format(msqe))
                print('- Mean absolute error of linear regression model is {}.'
                      .format(mabse))
                print('- R squared score of linear regression model is {}.'
                      .format(r2))
                print('\n\n')

                iteration += 1

            msqes = np.array(msqes)
            mabses = np.array(mabses)
            r2_scores = np.array(r2_scores)

            norm_msqes = msqes / np.linalg.norm(msqes)
            norm_mabses = mabses / np.linalg.norm(mabses)
            inverted_r2 = np.ones(r2_scores.shape) - r2_scores

            scores = norm_msqes + norm_mabses + inverted_r2

            min_index = int(np.argmin(scores))

            return models[min_index], msqes[min_index], mabses[min_index], r2_scores[min_index]

    def test_SVR(self, method='train_test_split', C=1.0, epsilon=0.0, tol=0.0001, max_iter=1000, random_state=123,
                 k=10, test_size=0.3):

        regressor = LinearSVR(C=C, epsilon=epsilon, tol=tol, max_iter=max_iter, random_state=random_state)

        print('- Regressor object created with C={}, epsilon={}, tol={}.\n'.format(C, epsilon, tol))

        X = self.df.iloc[:, 1:-1].values
        y = self.df.iloc[:, -1].values

        if method == 'train_test_split':

            print('- Starting to train linear SVR model with test size {} and random state {}...\n'
                  .format(test_size, random_state))

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            regressor.fit(X_train, y_train)
            preds = regressor.predict(X_test)

            msqe, mabse, r2 = mean_squared_error(y_test, preds), \
                              mean_absolute_error(y_test, preds), \
                              r2_score(y_test, preds)

            print('- Mean squared error of linear SVR model is {}.'
                  .format(msqe))
            print('- Mean absolute error of linear SVR model is {}.'
                  .format(mabse))
            print('- R squared score of linear SVR model is {}.'
                  .format(r2))
            print('\n\n')

            return regressor, msqe, mabse, r2

        elif method == 'k_fold':

            kfold = KFold(n_splits=k, random_state=random_state, shuffle=True)

            print('- Starting to train linear SVR model with kfold with k={} and random state {}...\n'
                  .format(k, random_state))

            models = []

            msqes = []
            mabses = []
            r2_scores = []

            iteration = 1

            for train_index, test_index in kfold.split(X):
                print('- Starting iteration {}...'.format(iteration))
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                regressor.fit(X_train, y_train)
                preds = regressor.predict(X_test)

                models.append(regressor)

                msqe = mean_squared_error(y_test, preds)
                mabse = mean_absolute_error(y_test, preds)
                r2 = r2_score(y_test, preds)

                msqes.append(msqe)
                mabses.append(mabse)
                r2_scores.append(r2)

                print('- Mean squared error of linear SVR model is {}.'
                      .format(mean_squared_error(y_test, preds)))
                print('- Mean absolute error of linear SVR model is {}.'
                      .format(mean_absolute_error(y_test, preds)))
                print('- R squared score of linear SVR model is {}.'
                      .format(r2_score(y_test, preds)))
                print('\n\n')
                iteration += 1

            msqes = np.array(msqes)
            mabses = np.array(mabses)
            r2_scores = np.array(r2_scores)

            norm_msqes = msqes / np.linalg.norm(msqes)
            norm_mabses = mabses / np.linalg.norm(mabses)
            inverted_r2 = np.ones(r2_scores.shape) - r2_scores

            scores = norm_msqes + norm_mabses + inverted_r2

            min_index = int(np.argmin(scores))

            return models[min_index], msqes[min_index], mabses[min_index], r2_scores[min_index]

    def test_SGD(self, method='train_test_split', alpha=0.0001, epsilon=0.1, tol=0.0001, max_iter=1000, random_state=123,
                 penalty='l2', learning_rate='invscaling', k=10, test_size=0.3):

        regressor = SGDRegressor(alpha=alpha, penalty=penalty, epsilon=epsilon, tol=tol, max_iter=max_iter,
                                 learning_rate=learning_rate, random_state=random_state)

        print('- Regressor object created with penalty={}, learning rate={}, alpha={}, epsilon={}, tol={}.\n'
              .format(penalty, learning_rate, alpha, epsilon, tol))

        X = self.df.iloc[:, 1:-1].values
        y = self.df.iloc[:, -1].values

        if method == 'train_test_split':

            print('- Starting to train SGD Regressor model with test size {} and random state {}...\n'
                  .format(test_size, random_state))

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            regressor.fit(X_train, y_train)
            preds = regressor.predict(X_test)

            msqe, mabse, r2 = mean_squared_error(y_test, preds), \
                              mean_absolute_error(y_test, preds), \
                              r2_score(y_test, preds)

            print('- Mean squared error of SGD Regressor model is {}.'
                  .format(msqe))
            print('- Mean absolute error of SGD Regressor model is {}.'
                  .format(mabse))
            print('- R squared score of SGD Regressor model is {}.'
                  .format(r2))
            print('\n\n')

            return regressor, msqe, mabse, r2

        elif method == 'k_fold':

            kfold = KFold(n_splits=k, random_state=random_state, shuffle=True)

            print('- Starting to train SGD Regressor model with kfold with k={} and random state {}...\n'
                  .format(k, random_state))

            iteration = 1

            models = []

            msqes = []
            mabses = []
            r2_scores = []

            for train_index, test_index in kfold.split(X):
                print('- Starting iteration {}...'.format(iteration))
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                regressor.fit(X_train, y_train)
                preds = regressor.predict(X_test)

                models.append(regressor)

                msqe = mean_squared_error(y_test, preds)
                mabse = mean_absolute_error(y_test, preds)
                r2 = r2_score(y_test, preds)

                msqes.append(msqe)
                mabses.append(mabse)
                r2_scores.append(r2)

                print('- Mean squared error of SGD Regressor model is {}.'
                      .format(mean_squared_error(y_test, preds)))
                print('- Mean absolute error of SGD Regressor model is {}.'
                      .format(mean_absolute_error(y_test, preds)))
                print('- R squared score of SGD Regressor model is {}.'
                      .format(r2_score(y_test, preds)))
                print('\n\n')
                iteration += 1

            msqes = np.array(msqes)
            mabses = np.array(mabses)
            r2_scores = np.array(r2_scores)

            norm_msqes = msqes / np.linalg.norm(msqes)
            norm_mabses = mabses / np.linalg.norm(mabses)
            inverted_r2 = np.ones(r2_scores.shape) - r2_scores

            scores = norm_msqes + norm_mabses + inverted_r2

            min_index = int(np.argmin(scores))

            return models[min_index], msqes[min_index], mabses[min_index], r2_scores[min_index]

    def test_polynomial_regression(self, method='train_test_split', degree=2, test_size=0.3, k=10, random_state=123):
        # polynomial features are unit price and change in inflation

        regressor = LinearRegression()
        pf = PolynomialFeatures(degree=degree, include_bias=False)

        X_before = self.df.iloc[:, :-1].values
        y = self.df.iloc[:, -1].values

        X_poly = pf.fit_transform(X_before[:, -2:])
        X = np.concatenate((X_before[:, :-2], X_poly), axis=1)

        if method == 'train_test_split':

            print('- Starting to train Polynomial Regression model with test size {} and random state {}...\n'
                  .format(test_size, random_state))

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            regressor.fit(X_train, y_train)
            preds = regressor.predict(X_test)

            msqe, mabse, r2 = mean_squared_error(y_test, preds), \
                              mean_absolute_error(y_test, preds), \
                              r2_score(y_test, preds)

            print('- Mean squared error of Polynomial Regression model is {}.'
                  .format(msqe))
            print('- Mean absolute error of Polynomial Regression model is {}.'
                  .format(mabse))
            print('- R squared score of Polynomial Regression model is {}.'
                  .format(r2))
            print('\n\n')

            return regressor, msqe, mabse, r2

        elif method == 'k_fold':

            kfold = KFold(n_splits=k, random_state=random_state, shuffle=True)

            print('- Starting to train Polynomial Regression model with kfold with k={} and random state {}...'
                  .format(k, random_state))

            iteration = 1

            models = []

            msqes = []
            mabses = []
            r2_scores = []

            for train_index, test_index in kfold.split(X):
                print('- Starting iteration {}...'.format(iteration))
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                regressor.fit(X_train, y_train)
                preds = regressor.predict(X_test)

                models.append(regressor)

                msqe = mean_squared_error(y_test, preds)
                mabse = mean_absolute_error(y_test, preds)
                r2 = r2_score(y_test, preds)

                msqes.append(msqe)
                mabses.append(mabse)
                r2_scores.append(r2)

                print('- Mean squared error of Polynomial Regression model is {}.'
                      .format(mean_squared_error(y_test, preds)))
                print('- Mean absolute error of Polynomial Regression model is {}.'
                      .format(mean_absolute_error(y_test, preds)))
                print('- R squared score of Polynomial Regression model is {}.'
                      .format(r2_score(y_test, preds)))
                print('\n\n')

                iteration += 1

            msqes = np.array(msqes)
            mabses = np.array(mabses)
            r2_scores = np.array(r2_scores)

            norm_msqes = msqes / np.linalg.norm(msqes)
            norm_mabses = mabses / np.linalg.norm(mabses)
            inverted_r2 = np.ones(r2_scores.shape) - r2_scores

            scores = norm_msqes + norm_mabses + inverted_r2
            min_index = int(np.argmin(scores))

            return models[min_index], msqes[min_index], mabses[min_index], r2_scores[min_index]
