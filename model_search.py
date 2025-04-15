import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_validate
from bayes_opt import BayesianOptimization


class BayesianSearchCV:
    def __init__(self, model_class, param_bounds, cv=4, n_iter=20, random_state=None):
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
        self.results_df = pd.DataFrame(columns=['Model', 'Parameters', 'Mean CV R2 Score'])

    def _objective_function(self, **params):
        """
        Целевая функция для байесовкой оптимизации

        :param params: Параметры модели
        :return: Среднее значение метрики на кросс-валидации
        """
        # Устанавливаем параметры модели
        if 'loss_function' in dir(self.model_class):
            model = self.model_class(**params, verbose=0, loss_function='RMSE')
        else:
            model = self.model_class(**params, verbose=0)

        cv_results = cross_validate(model, self.X, self.y, cv=self.cv, scoring=['r2', 'neg_mean_squared_error'])
        r2, nmse = cv_results['test_r2'], cv_results['test_neg_mean_squared_error']
        mean_cv_score = np.mean(r2)

        return mean_cv_score

    def fit(self, X, y):
        """
        Обучение модели с использованием байесовской оптимизации

        :param X: Матрица признаков
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
        self.optimizer.maximize(init_points=10, n_iter=self.n_iter)

        # Устанавливаем лучшие найденные параметры
        best_params = {key: int(value) if value.is_integer() else value for key, value in
                       self.optimizer.max['params'].items()}

        # Обучаем модель с лучшими параметрами на всем датасете
        self.best_model = self.model_class(**best_params, verbose=0)
        self.best_model.fit(X, y)

        self.results_df = pd.concat([self.results_df, pd.DataFrame({
            'Model': type(self.best_model).__name__,
            'Parameters': [best_params],
            'Mean CV R2 Score': self.optimizer.max['target']
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
        name = type(self.best_model).__name__
        self.best_model.save_model(name)


class MultiModelBayesianSearchCV:
    def __init__(self, model_classes, cv=4, n_iter=20, random_state=None):
        """
        Инициализация класса для оптимизации нескольких моделей.

        :param model_classes: Список классов моделей для оптимизации.
        :param cv: Количество фолдов для кросс-валидации.
        :param n_iter: Количество итераций для байесовской оптимизации.
        :param random_state: Случайное состояние для воспроизводимости.
        """
        self.model_classes = model_classes
        self.best_models = []
        self.cv = cv
        self.n_iter = n_iter
        self.random_state = random_state
        self.results_df = pd.DataFrame(columns=['Model', 'Parameters', 'Mean CV R2 Score'])

        # Предопределённые границы параметров для различных классов моделей
        self.param_bounds_templates = {
            'RandomForestRegressor': {
                'n_estimators': (10, 200, int),
                'max_depth': (1, 30, int),
                'min_samples_split': (2, 20, int),
                'min_samples_leaf': (1, 20, int)
            },
            'CatBoostRegressor': {
                'iterations': (50, 500, int),
                'learning_rate': (0.01, 0.3),
                'depth': (1, 10, int),
                'l2_leaf_reg': (1, 20),
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
                'max_leaves': (1, 30, int)
            },
            'ExtraTreesRegressor': {
                'n_estimators': (10, 300, int),
                'max_depth': (1, 30, int),
                'min_samples_split': (2, 20, int),
                'min_samples_leaf': (1, 20, int)
            },
            'NWScikit': {
                'batch_size': (10, 100, int),
                'epoch_n': (10, 500, int),
                'n_neurons': (8, 256, int),
                'n_layers': (1, 4, int)
            }
        }

    def _get_param_bounds(self, model_class):
        """
        Получение границ параметров для заданного класса модели.

        :param model_class: Класс модели.
        :return: Словарь с границами параметров.
        """
        model_name = model_class.__name__
        if model_name in self.param_bounds_templates:
            return self.param_bounds_templates[model_name]
        else:
            raise ValueError(f"Границы параметров для модели {model_name} не определены.")

    def fit(self, X, y):
        """
        Проведение байесовской оптимизации для всех моделей.

        :param X: Признаки для обучения.
        :param y: Целевые значения.
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
        Получение результатов оптимизации.

        :return: Датафрейм с результатами.
        """
        return self.results_df

    def score(self, X, y):
        """
        Получение R2 Score для всех моделей
        :param X: Параметры
        :param y: Точное значение
        :return:
        """
        if len(self.best_models) == 0:
            raise ValueError(f"Нет обученных моделей для оценки")

        model_scores = []

        for model in self.best_models:
            _score = r2_score(y, model.predict(X))
            model_scores.append(_score)

        self.results_df['Test R2 Score'] = model_scores

    def find_max(self, scaler, init_points=20, n_iter=100):
        """
        Поиск максимума

        :return:
        """

        def func(**params):
            x = np.array(list(params.values())).reshape(1, -1)
            x_normalized = scaler.transform(x)
            return _model.predict(x_normalized)[0]

        opt_bounds = {
            'x1': (0, 100),
            'x2': (0, 100),
            'x3': (0, 100),
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

            print(f"Model {type(_model).__name__} has found maximum {max_y} with parameters {optimal_x}")

    def save_models(self):
        """
        Сохранение моделей
        """
        for model in self.best_models:
            name = str(type(model).__name__)
            model.save_model(name)
