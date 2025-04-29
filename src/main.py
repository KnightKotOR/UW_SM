import numpy as np
import pandas as pd
import warnings
import model_search

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, \
    HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from nw_kernel import NWScikit

warnings.filterwarnings(action="ignore")

# Загрузка данных
df = pd.read_csv('data/synth_ds_0.csv')
#df = pd.read_csv('data/real_ds_0.csv')
columns = [*df]

# Преобразование в массивы
x_data = np.array(df[columns[:-1]].values.tolist())
#x_data = np.array(df[columns[:-3]].values.tolist())
y_data = np.array(df[columns[-1]].values.tolist()).reshape([-1, 1])

# Разделение на обучающую и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, shuffle=True)

# Нормализация
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

if __name__ == "__main__":
    # Классы моделей
    model_classes = [NWScikit, HistGradientBoostingRegressor, XGBRegressor, GradientBoostingRegressor,
                     ExtraTreesRegressor,
                     RandomForestRegressor, CatBoostRegressor]

    # Гиперпараметрический поиск и оценка моделей
    BO = model_search.MultiModelBayesianSearchCV(model_classes, n_iter=50, random_state=42)
    BO.fit(x_train, y_train)
    BO.score(x_test, y_test)
    results = BO.get_results()
    results.to_csv('results/results.csv')
    #results.to_csv('results/real_results.csv')
    print(results.to_string(), '\n')

    # Оптимизация над аппроксимируемой функцией
    BO.find_max(scaler)

    # BO.save_models()
