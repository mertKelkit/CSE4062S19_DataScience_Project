import numpy as np

from experiment import Experimenter
from sklearn.preprocessing import PolynomialFeatures


class Evaluator:

    def __init__(self, df):
        self.__df = df
        self.__models = []       # list of models tested
        self.__msqes = []        # list of mean squared errors
        self.__mabses = []       # list of mean absolute errors
        self.__r2_scores = []    # list of R square scores
        self.__linearities = []  # list of linearities of models - False for polynomial regression, True for others
        self.__degrees = []      # list of polynomial degrees, -1 if model is linear

    # Experiment
    def find_best_estimator(self):

        experimenter = Experimenter(df=self.__df)

        random_state = np.random.randint(500)

        # Test linear regressor with specified parameters

        # method='train_test_split'
        test_sizes = [0.2, 0.3, 0.4, 0.5]
        for s in test_sizes:
            result = experimenter.test_linear_regression(method='train_test_split', test_size=s,
                                                         random_state=random_state)
            self.__models.append(result[0])
            self.__msqes.append(result[1])
            self.__mabses.append(result[2])
            self.__r2_scores.append(result[3])
            self.__linearities.append(True)
            self.__degrees.append(-1)

        # method='k_fold'
        k_values = np.arange(5, 11)
        for k in k_values:
            result = experimenter.test_linear_regression(method='k_fold', k=k, random_state=random_state)

            self.__models.append(result[0])
            self.__msqes.append(result[1])
            self.__mabses.append(result[2])
            self.__r2_scores.append(result[3])
            self.__linearities.append(True)
            self.__degrees.append(-1)

        '''
        # Test linear SVR with specified parameters

        # method='train_test_split'
        test_sizes = [0.3]
        for s in test_sizes:
            C_vals = [1e-2, 1e-1, 1, 10, 100]
            for C in C_vals:
                max_iters = [1000]
                for max_iter in max_iters:
                    result = experimenter.test_SVR(method='train_test_split', test_size=s,
                                                   C=C, epsilon=0.0, max_iter=max_iter,
                                                   random_state=random_state)
                    self.__models.append(result[0])
                    self.__msqes.append(result[1])
                    self.__mabses.append(result[2])
                    self.__r2_scores.append(result[3])
                    self.__linearities.append(True)
                    self.__degrees.append(-1)
        
        # method='k_fold'
        k_values = np.arange(2, 11)
        for k in k_values:
            C_vals = [1e-1, 1, 10, 100]
            for C in C_vals:
                tols = [1e-5, 1e-4, 1e-3, 1e-2]
                for tol in tols:
                    max_iters = [2000, 2500]
                    for max_iter in max_iters:
                        result = experimenter.test_SVR(method='k_fold', k=k, C=C, epsilon=0.0,
                                                       tol=tol, max_iter=max_iter, random_state=random_state)
                        self.__models.append(result[0])
                        self.__msqes.append(result[1])
                        self.__mabses.append(result[2])
                        self.__r2_scores.append(result[3])
                        self.__linearities.append(True)
                        self.__degrees.append(-1)
        
        # Test SGD regressor with specified parameters
        
        # method='train_test_split'
        test_sizes = [0.3]
        for s in test_sizes:
            alpha_vals = [1e-4, 1e-3, 1e-2, 1e-1, 1]
            for alpha in alpha_vals:
                max_iters = [1000]
                for max_iter in max_iters:
                    result = experimenter.test_SGD(method='train_test_split', test_size=s, max_iter=max_iter, alpha=alpha,
                                                   random_state=random_state)

                    print('Best {} - {} - {}.'.format(result[1], result[2], result[3]))

                    self.__models.append(result[0])
                    self.__msqes.append(result[1])
                    self.__mabses.append(result[2])
                    self.__r2_scores.append(result[3])
                    self.__linearities.append(True)
                    self.__degrees.append(-1)
        
        # method='k_fold'
        k_values = np.arange(10, 11)
        for k in k_values:
            alpha_vals = [1e-4, 1e-3, 1e-2, 1e-1, 1]
            for alpha in alpha_vals:
                max_iters = [1000]
                for max_iter in max_iters:
                    result = experimenter.test_SGD(method='k_fold', k=k, max_iter=max_iter, alpha=alpha,
                                                   random_state=random_state)
                    self.__models.append(result[0])
                    self.__msqes.append(result[1])
                    self.__mabses.append(result[2])
                    self.__r2_scores.append(result[3])
                    self.__linearities.append(True)
                    self.__degrees.append(-1)
        '''
        # Test polynomial regressor with specified parameters
        
        # method='train_test_split'
        test_sizes = [0.2, 0.3, 0.4, 0.5]
        degrees = [2, 3, 4, 5]
        for s in test_sizes:
            for d in degrees:
                result = experimenter.test_polynomial_regression(method='train_test_split', test_size=s,
                                                                 random_state=random_state, degree=d)
                self.__models.append(result[0])
                self.__msqes.append(result[1])
                self.__mabses.append(result[2])
                self.__r2_scores.append(result[3])
                self.__linearities.append(False)
                self.__degrees.append(d)

        # method='k_fold'
        k_values = np.arange(5, 11)
        for k in k_values:
            for d in degrees:
                result = experimenter.test_polynomial_regression(method='k_fold', k=k, random_state=random_state,
                                                                 degree=d)
                self.__models.append(result[0])
                self.__msqes.append(result[1])
                self.__mabses.append(result[2])
                self.__r2_scores.append(result[3])
                self.__linearities.append(False)
                self.__degrees.append(d)

        return self.__eval_results()

    def __eval_results(self):
        msqes = np.array(self.__msqes)
        mabses = np.array(self.__mabses)
        r2_scores = np.array(self.__r2_scores)

        norm_msqes = msqes / np.linalg.norm(msqes)
        norm_mabses = mabses / np.linalg.norm(mabses)
        inverted_r2 = np.ones(r2_scores.shape) - r2_scores

        scores = norm_msqes + norm_mabses + inverted_r2

        min_index = int(np.argmin(scores))

        best_estimator = Estimator(df=self.__df, model=self.__models[min_index], mean_squared_error=msqes[min_index],
                                   mean_absolute_error=mabses[min_index], r2_score=r2_scores[min_index],
                                   linear=self.__linearities[min_index], degree=self.__degrees[min_index])

        return best_estimator


class Estimator:

    def __init__(self, df, model, mean_squared_error, mean_absolute_error, r2_score, linear, degree):
        self.df = df
        self.best_model = model
        self.mean_squared_error = mean_squared_error
        self.mean_absolute_error = mean_absolute_error
        self.r2_score = r2_score
        self.linear = linear
        self.degree = degree

    def product_based_prediction(self, X):
        # X contains <- 'Product Number', 'Day', 'Month', 'Year', 'Change in Inflation'
        # and it needs to be 2d array

        preds = []

        for x in X:
            mask = np.array((self.df['Product Number'] == x[0]) & (self.df['Gün'] == x[1]) & (self.df['Ay'] == x[2]))
            fltr = np.repeat(False, mask.shape[0])

            if np.array_equal(mask, fltr):
                print('Product {} is no longer sold.'.format(x[0]))
                continue

            brand_num = self.df[mask].iloc[0, 1]
            profile_num = self.df[mask].iloc[0, 2]

            previous_total_sales_revenues = self.df[mask].iloc[:, 6].values
            previous_unit_prices = self.df[mask].iloc[:, 7].values
            '''
            shp = previous_unit_prices.shape
            
            previous_change_in_inflations = self.df[mask].iloc[:, -2].values
            previous_years = self.df[mask].iloc[:, 5].values
            
            features = np.concatenate((previous_change_in_inflations.reshape((shp[0], 1)),
                                       previous_years.reshape((shp[0], 1))), axis=1)
            current_total_sales_revenue_pred = LinearRegression().fit(features, previous_total_sales_revenues)\
                .predict(np.array([[x[-1], x[3]]]))[0]
            current_unit_price_pred = LinearRegression().fit(features, previous_unit_prices) \
                .predict(np.array([[x[-1], x[3]]]))[0]
            '''
            current_total_sales_revenue_pred = np.mean(previous_total_sales_revenues)
            current_unit_price_pred = np.mean(previous_unit_prices)

            if self.linear:
                preds.append(self.best_model.predict(np.array([
                    [
                        x[0], brand_num, profile_num, x[1], x[2], x[3], current_total_sales_revenue_pred,
                        current_unit_price_pred, x[4]
                    ]
                ])))
            else:
                pf = PolynomialFeatures(degree=self.degree, include_bias=False)
                # Polynomial features are inflation change percentage and unit price
                X_not_poly = np.array([
                    [
                        x[0], brand_num, profile_num, x[1], x[2], x[3], current_total_sales_revenue_pred
                    ]
                ])
                X_poly = pf.fit_transform(np.array([[current_unit_price_pred, x[4]]]))

                preds.append(self.best_model.predict(np.concatenate((X_not_poly, X_poly), axis=1)))
        return preds
