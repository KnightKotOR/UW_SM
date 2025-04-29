from inspect import signature
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.model_selection import cross_validate, LeaveOneOut
from bayes_opt import BayesianOptimization


class BayesianSearchCV:
    def __init__(self, model_class, param_bounds, cv=LeaveOneOut(), n_iter=20, random_state=None):
        """
        Инициализация класса BayesianOptimizationWrapper

        :param model_class: Класс модели для оптимизации
        :param param_bounds: Границы параметров для оптимизации в формате {'param_name': (min, max)}
        :param cv: Количество фолдов для кросс-валидации
        :param n_iter: Количество итераций для байесовкой оптимизации
        :param random_state: Случайное состояние для воспроизводимости результатов
        """
        self.best_model = None

        self.y = None
        self.X = None
        self.model_class = model_class
        self.param_bounds = param_bounds
        self.cv = cv
        self.n_iter = n_iter
        self.random_state = random_state
        self.optimizer = None
        self.results_df = pd.DataFrame(columns=['Model', 'Parameters', 'Mean CV Score'])

    def _objective_function(self, **params):
        """
        Целевая функция для байесовкой оптимизации

        :param params: Параметры модели
        :return: Среднее значение метрики на кросс-валидации
        """
        # Получение списка параметров модели
        p = list(signature(self.model_class).parameters.keys())

        # При наличии параметра verbose, он указывается с 0
        if 'verbose' in p:
            model = self.model_class(**params, verbose=0)
        else:
            model = self.model_class(**params)

        cv_results = cross_validate(model, self.X, self.y, cv=self.cv,
                                    scoring=['neg_mean_squared_error', 'neg_mean_absolute_percentage_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error'],
                                    return_estimator=True, n_jobs=-1)

        nmse, mape, nmsle, nmedae = cv_results['test_neg_mean_squared_error'], cv_results[
            'test_neg_mean_absolute_percentage_error'], cv_results[
            'test_neg_mean_squared_log_error'], cv_results[
            'test_neg_median_absolute_error']
        mean_cv_score = np.mean(nmedae)

        return mean_cv_score

    def fit(self, X, y):
        """
        Обучение модели с использованием байесовской оптимизации

        :param X: Признаки
        :param y: Вектор целевых значений
        """
        self.X = X
        self.y = y

        # Создаем объект байесовской оптимизации
        self.optimizer = BayesianOptimization(
            f=self._objective_function,
            pbounds=self.param_bounds,
            random_state=self.random_state,
            verbose=2
        )

        # Запускаем оптимизацию
        self.optimizer.maximize(init_points=20, n_iter=self.n_iter)

        # Устанавливаем лучшие найденные параметры
        best_params = {key: int(value) if value.is_integer() else value for key, value in
                       self.optimizer.max['params'].items()}

        # Обучаем модель с лучшими параметрами на всем датасете
        self.best_model = self.model_class(**best_params, verbose=0)
        self.best_model.fit(X, y)

        self.results_df = pd.concat([self.results_df, pd.DataFrame({
            'Model': type(self.best_model).__name__,
            'Parameters': [best_params],
            'Mean CV Score': self.optimizer.max['target']
        })], ignore_index=True)

    def predict(self, X):
        if self.best_model is None:
            raise ValueError("Модель еще не обучена. Сначала вызовите метод fit.")
        return self.best_model.predict(X)

    def get_results(self):
        """
        Получение результатов кросс-валидации

        :return: Датафрейм с результатами
        """
        return self.results_df

    def save_model(self, score):
        """
        Сохранение лучшей модели
        """
        name = type(self.best_model).__name__
        self.best_model.save_model(name)


class MultiModelBayesianSearchCV:
    def __init__(self, model_classes, cv=LeaveOneOut(), n_iter=10, random_state=None):
        """
        Инициализация класса для оптимизации нескольких моделей

        :param model_classes: Список классов моделей для оптимизации
        :param cv: Количество фолдов для кросс-валидации
        :param n_iter: Количество итераций для байесовской оптимизации
        :param random_state: Случайное состояние для воспроизводимости
        """
        self.model_classes = model_classes
        self.best_models = []
        self.cv = cv
        self.n_iter = n_iter
        self.random_state = random_state
        self.results_df = pd.DataFrame(columns=['Model', 'Parameters', 'Mean CV Score'])

        # Предопределённые границы параметров для различных классов моделей
        self.param_bounds_templates = {
            'RandomForestRegressor': {
                'n_estimators': (10, 200, int),
                'max_depth': (1, 30, int),
                'min_samples_split': (2, 20, int),
                'min_samples_leaf': (1, 20, int)
            },
            'CatBoostRegressor': {
                'iterations': (50, 400, int),
                'learning_rate': (0.01, 0.3),
                'depth': (1, 10, int),
                'l2_leaf_reg': (1, 30),
            },
            'GradientBoostingRegressor': {
                'n_estimators': (10, 300, int),
                'learning_rate': (0.01, 0.3),
                'max_depth': (1, 30, int),
                'min_samples_split': (2, 20, int),
                'min_samples_leaf': (1, 20, int)
            },
            'XGBRegressor': {
                'n_estimators': (10, 300, int),
                'learning_rate': (0.001, 0.3),
                'max_depth': (1, 30, int),
                'max_leaves': (1, 30, int),
                'reg_alpha':(0, 1)
            },
            'ExtraTreesRegressor': {
                'n_estimators': (10, 300, int),
                'max_depth': (1, 30, int),
                'min_samples_split': (2, 20, int),
                'min_samples_leaf': (1, 20, int)
            },
            'NWScikit': {
                'batch_size': (40, 80, int),
                'epoch_n': (100, 500, int),
                'n_neurons': (50, 256, int),
                'n_layers': (1, 2, int),
                'lr': (1e-5, 5e-4)
            },
            'Ridge': {
                'alpha': (0.8, 1.2)
            },
            'ElasticNet': {
                'alpha': (0.1, 1.2),
                'l1_ratio': (0, 1),
                'max_iter': (100, 1000, int)
            },
            'HistGradientBoostingRegressor': {
                'learning_rate': (0.01, 0.2),
                'max_iter': (10, 500, int),
                'max_leaf_nodes': (15, 50, int),
                'max_depth': (1, 30, int),
                'min_samples_leaf': (1, 20, int)
            }
        }

    def _get_param_bounds(self, model_class):
        """
        Получение границ параметров для заданного класса модели

        :param model_class: Класс модели
        :return: Словарь с границами параметров
        """
        model_name = model_class.__name__
        if model_name in self.param_bounds_templates:
            return self.param_bounds_templates[model_name]
        else:
            raise ValueError(f"Границы параметров для модели {model_name} не определены.")

    def fit(self, X, y):
        """
        Проведение байесовской оптимизации для всех моделей

        :param X: Признаки для обучения
        :param y: Целевые значения
        """
        for model_class in self.model_classes:
            param_bounds = self._get_param_bounds(model_class)

            print(f"Оптимизация модели {model_class.__name__}")
            # Создание экземпляра BayesianSearchCV для текущей модели
            optimizer = BayesianSearchCV(
                model_class=model_class,
                param_bounds=param_bounds,
                cv=self.cv,
                n_iter=self.n_iter,
                random_state=self.random_state
            )

            # Обучение и оптимизация
            optimizer.fit(X, y)
            self.best_models.append(optimizer.best_model)

            # Добавление результатов в общий датафрейм
            self.results_df = pd.concat([self.results_df, optimizer.get_results()], ignore_index=True)

    def get_results(self):
        """
        Получение результатов оптимизации

        :return: Датафрейм с результатами
        """
        return self.results_df

    def score(self, X, y):
        """
        Получение R2 Score для всех моделей
        :param X: Параметры
        :param y: Точное значение
        :return: R2 Score
        """
        if len(self.best_models) == 0:
            raise ValueError(f"Нет обученных моделей для оценки")

        model_scores = []

        for model in self.best_models:
            _score = r2_score(y, model.predict(X))
            model_scores.append(_score)

        self.results_df['Test R2 Score'] = model_scores

    def find_max(self, scaler, real_ds=False, init_points=20, n_iter=80):
        """
        Поиск максимума.
        :param scaler: scaler, использовавшийся для нормализации обучающей и тестовой выборок
        :param real_ds: Флаг для выбора границ оптимизации реального датасета
        :param init_points: Количество начальных точек байесовской оптимизации
        :param n_iter: Количество итераций для байесовской оптимизации

        :return:
        """

        def func(**params):
            x = np.array(list(params.values())).reshape(1, -1)
            x_normalized = scaler.transform(x)
            y_pred = _model.predict(x_normalized).flatten()[0]
            return y_pred

        if real_ds:
            opt_bounds = {
                'Pc': (2, 4, int),
                'U': (50, 100),
                't': (2, 5, int),
                'L': (8, 22),
                'B': (8, 22)
            }
        else:
            opt_bounds = {
                'x1': (0, 140),
                'x2': (0, 140),
                'x3': (0, 140),
            }

        for _model in self.best_models:
            optimizer = BayesianOptimization(
                f=func,
                pbounds=opt_bounds,
                verbose=0
            )

            optimizer.maximize(init_points, n_iter)
            optimal_x = optimizer.max['params']
            max_y = optimizer.max['target']

            print(f"Model {type(_model).__name__} found maximum {max_y} with parameters {optimal_x}")

    def save_models(self):
        """
        Сохранение моделей
        """
        for model in self.best_models:
            name = str(type(model).__name__)
            model.save_model(name)
