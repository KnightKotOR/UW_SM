import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from nw_kernel import NWScikit
import model_search

from catboost import CatBoostRegressor

warnings.filterwarnings(action="ignore")

df = pd.read_csv('data/real_ds_0.csv')

# Преобразование в массивы
x_data = np.array(df[['Pc', 'U', 't', 'L', 'B', 'Sc', 'Pp']].values.tolist())
y_data = np.array(df['D'].values.tolist()).reshape([-1, 1])

# Разделение на обучающую и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, shuffle=True)

# Нормализация
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Байесовская оптимизация

model_classes = [RandomForestRegressor, CatBoostRegressor]

BO = model_search.MultiModelBayesianSearchCV(model_classes, cv=4, n_iter=100, random_state=42)
BO.fit(x_train, y_train)
BO.score(x_test, y_test)
results = BO.get_results()
results.to_csv('data/results.csv')
print(results.to_string())

# Байесовская оптимизация над аппроксимируемой функцией
BO.find_max(scaler)
