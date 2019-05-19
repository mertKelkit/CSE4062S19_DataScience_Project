import pandas as pd

from regression import Evaluator

df = pd.read_excel('final_data.xlsx').drop(['Alıcı Ürün Kodu'], axis=1)

'''
experimenter = Experimenter(df)
best_est = experimenter.test_linear_regression(method='k_fold', k=2, random_state=321)
'''
evaluator = Evaluator(df=df)
best_estimator = evaluator.find_best_estimator()
