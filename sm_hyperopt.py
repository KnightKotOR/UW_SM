import numpy as np
import pandas as pd
import torch
import warnings

from hyperopt import tpe
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from nw_kernel import NWScikit
import model_search

from hpsklearn import HyperoptEstimator, random_forest_classifier

warnings.filterwarnings(action="ignore")

df = pd.read_csv('data/synth_ds_0.csv')

# Преобразование в массивы
x_data = np.array(df[['x1', 'x2', 'x3']].values.tolist())
y_data = np.array(df['y'].values.tolist()).reshape([-1, 1])

# Разделение на обучающую и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, shuffle=True)

# Нормализация
scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# Байесовская оптимизация над параметрами модели
model_classes = [RandomForestRegressor]

# BO = model_search.MultiModelBayesianSearchCV(model_classes, cv=x_train.shape[0], n_iter=60, random_state=42)
# BO.fit(x_train, y_train)
# BO.score(x_test, y_test)
# results = BO.get_results()
# results.to_csv('data/results.csv')
# print(results.to_string())

# Байесовская оптимизация над аппроксимируемой функцией
# BO.find_max(scaler)

# BO.save_models()


estim = HyperoptEstimator(classifier=random_forest_classifier("my_clf"),
                              preprocessing=[],
                              algo=tpe.suggest,
                              max_evals=10,
                              trial_timeout=300)

estim.fit(x_train, y_train)
print(estim.score(x_test, y_test))
