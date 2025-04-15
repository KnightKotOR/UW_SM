import numpy as np
import pandas as pd
import torch
import warnings

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from nw_kernel import NWScikit
import model_search

from catboost import CatBoostRegressor

warnings.filterwarnings(action="ignore")

df = pd.read_csv('data/synth_ds_0.csv')

# Преобразование в массивы
x_data = np.array(df[['x1', 'x2', 'x3']].values.tolist())
y_data = np.array(df['y'].values.tolist()).reshape([-1, 1])

# Разделение на обучающую и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, shuffle=True)

# Нормализация
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


if __name__ == "__main__":
    # Байесовская оптимизация над параметрами модели
    # model_classes = [NWScikit, GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor, CatBoostRegressor]
    #
    # BO = model_search.MultiModelBayesianSearchCV(model_classes, cv=4, n_iter=80, random_state=42)
    # BO.fit(x_train, y_train)
    # BO.score(x_test, y_test)
    # results = BO.get_results()
    # results.to_csv('data/results.csv')
    # print(results.to_string())
    #
    # # Байесовская оптимизация над аппроксимируемой функцией
    # BO.find_max(scaler)
    #
    # # BO.save_models()

    model = NWScikit(
        dist_mode='mlp',
        batch_size=24,
        kernel_fit_background=True,
        optimizer='Adam',
        lr=1e-3,
        weight_decay=0,
        background_lr=1e-3,
        background_weight_decay=0,
        epoch_n=100,
        pred_batch_size=16,
        verbose=False,
        verbose_tqdm=True,
        n_neurons=8,
        n_layers=3,
        batch_norm=True,
        problem_mode='reg'
    )


