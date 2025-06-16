import numpy as np
import optuna
import optunahub
import pandas as pd
import warnings

from catboost import CatBoostRegressor
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from nw_kernel import NWScikit

from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from plotly.io import show

warnings.filterwarnings('ignore')


class Objective(object):
    """
    Целевая функция оптимизации гиперпараметров с кросс-валидацией
    """

    def __init__(self, X, y, model_name, cv):
        self.X, self.y = X, y
        self.model_name = model_name
        self.cv = cv
        self.model_results_df = pd.DataFrame(
            columns=['n', 'CV_mode', 'Model', 'R2_val', 'Parameters'])
        self.best_model = None
        self.best_model_est = None
        self.best_score = float('-inf')
        self.test_score = float('-inf')

    def __call__(self, trial):
        warnings.filterwarnings('ignore')
        # Определение модели и ее параметров
        # regressor_name = trial.suggest_categorical("regressor", self.models_list)
        regressor_name = self.model_name
        if regressor_name == "CatBoostRegressor":
            lr = trial.suggest_float("learning_rate", 1e-4, 3e-1, log=True)
            depth = trial.suggest_int("depth", 3, 10)
            l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1e-2, 10, log=True)
            bagging_temp = trial.suggest_float('bagging_temperature', 1e-6, 10.0, log=True)
            random_strength = trial.suggest_float('random_strength', 1e-6, 10, log=True)
            grow_policy = trial.suggest_categorical("grow_policy", ['SymmetricTree', 'Depthwise', 'Lossguide'])
            border_count = trial.suggest_int("border_count", 32, 255)

            regressor_obj = CatBoostRegressor(iterations=300, learning_rate=lr, depth=depth, l2_leaf_reg=l2_leaf_reg,
                                              bagging_temperature=bagging_temp, random_strength=random_strength,
                                              grow_policy=grow_policy, border_count=border_count, verbose=False,
                                              early_stopping_rounds=40)
        elif regressor_name == "XGBRegressor":
            lr = trial.suggest_float("learning_rate", 1e-4, 5e-1, log=True)
            max_depth = trial.suggest_int("max_depth", 1, 40)
            min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
            reg_alpha = trial.suggest_float('reg_alpha', 1e-8, 10, log=True)
            reg_lambda = trial.suggest_float('reg_lambda', 1e-8, 10, log=True)
            subsample = trial.suggest_float('subsample', 0.5, 1e3, log=True)
            colsample = trial.suggest_float('colsample_bytree', 0.5, 1e3, log=True)
            regressor_obj = XGBRegressor(
                max_depth=max_depth, n_estimators=200, learning_rate=lr, reg_alpha=reg_alpha,
                reg_lambda=reg_lambda, subsample=subsample, colsample_bytree=colsample,
                min_child_weight=min_child_weight, n_jobs=-1
            )
        elif regressor_name == "RandomForestRegressor":
            n = trial.suggest_int("n_estimators", 50, 2000)
            depth = trial.suggest_int("max_depth", 1, 50)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
            max_features = trial.suggest_float('max_features', 0.2, 1.0, log=True)
            bootstrap = trial.suggest_categorical("bootstrap", [True, False])
            regressor_obj = RandomForestRegressor(n_estimators=n, max_depth=depth, min_samples_split=min_samples_split,
                                                  min_samples_leaf=min_samples_leaf, max_features=max_features,
                                                  bootstrap=bootstrap, verbose=0)
        elif regressor_name == "HistGradientBoostingRegressor":
            learning_rate = trial.suggest_float("learning_rate", 1e-3, 2e-1, log=True)
            max_depth = trial.suggest_int("max_depth", 1, 10)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 100)
            max_features = trial.suggest_float('max_features', 0.1, 1.0, log=True)
            l2_regularization = trial.suggest_float('l2_regularization', 0.0, 1.0)
            n_iter_no_change = trial.suggest_int("n_iter_no_change", 10, 40)
            regressor_obj = HistGradientBoostingRegressor(
                max_depth=max_depth, max_iter=500, learning_rate=learning_rate, n_iter_no_change=n_iter_no_change,
                max_features=max_features, min_samples_leaf=min_samples_leaf, l2_regularization=l2_regularization,
                verbose=0
            )
        elif regressor_name == "ANN":
            lr = trial.suggest_float("learning_rate", 1e-6, 5e-4, log=True)
            batch_size = trial.suggest_int("batch_size", 50, 80)
            n_neurons = trial.suggest_int("n_neurons", 20, 200)
            n_layers = trial.suggest_int('n_layers', 1, 6)
            mode = trial.suggest_categorical("dist_mode", ['mlp', 'nam'])
            optimizer = trial.suggest_categorical("optimizer", ['SGD', 'Adam'])
            regressor_obj = NWScikit(lr=lr, batch_size=batch_size, n_neurons=n_neurons, n_layers=n_layers,
                                     dist_mode=mode, optimizer=optimizer, epoch_n=250)
        elif regressor_name == "ElasticNet":
            alpha = trial.suggest_float('alpha', 1e-4, 10, log=True)
            l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
            max_iter = trial.suggest_int('max_iter', 50, 500)
            fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
            positive = trial.suggest_categorical("positive", [True, False])
            regressor_obj = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, fit_intercept=fit_intercept,
                                       positive=positive)
        elif regressor_name == "BayesianRidge":
            alpha_1 = trial.suggest_float("alpha_1", 1e-8, 1e-1, log=True)
            alpha_2 = trial.suggest_float("alpha_2", 1e-8, 1e-1, log=True)
            lambda_1 = trial.suggest_float("lambda_1", 1e-8, 1e-1, log=True)
            lambda_2 = trial.suggest_float("lambda_2", 1e-8, 1e-1, log=True)
            compute_score = trial.suggest_categorical("compute_score", [True, False])
            fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
            regressor_obj = BayesianRidge(alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1, lambda_2=lambda_2,
                                          compute_score=compute_score, fit_intercept=fit_intercept)
        else:
            raise ValueError(f"Unknown model: {regressor_name}")

        if isinstance(self.cv, int):
            cv_mode = f"kfold_{self.cv}"
        elif isinstance(self.cv, LeaveOneOut):
            cv_mode = "loo"
        else:
            raise "Wrong CV mode"

        # Кросс-валидация
        y_pred = cross_val_predict(regressor_obj, self.X, self.y, cv=self.cv, verbose=0, n_jobs=-1)
        r2_val = r2_score(self.y, y_pred)

        if r2_val > self.best_score:
            self.best_model_est = regressor_obj
            self.best_score = r2_val

        # Обновление дф
        self.model_results_df = pd.concat([self.model_results_df, pd.DataFrame({
            'n': trial.number,
            'CV_mode': cv_mode,
            'Model': regressor_name,
            'Parameters': [trial.params],
            'R2_val': r2_val
        })], ignore_index=True)

        return r2_val


class OptunaSearchCV:
    """
    Класс для гиперпараметрического поиска по множеству моделей.
    В результате датафрейм с подробной информацией для анализа
    """

    def __init__(self, models_list, compare_kfold):
        self.models_list = models_list
        self.results_df = pd.DataFrame(columns=['n', 'CV_mode', 'Model', 'R2_val', 'Parameters'])
        self.best_models = []
        self.best_models_y_pred = {}
        self.best_models_r_test = []
        self.best_models_r_val = []
        self.opt_study_storage = optuna.storages.InMemoryStorage()
        self.kfold = compare_kfold

    def fit(self, x, y, n_trials=100, n_startup_trials=20):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        hyperopt_storage = optuna.storages.InMemoryStorage()
        for model in self.models_list:
            print(f"\n{model} hyperoptimization")

            objective = Objective(x, y, model, cv=8)
            study = optuna.create_study(storage=hyperopt_storage, study_name=f"synth_{model}_LOOCV",
                                        direction="maximize",
                                        sampler=optuna.samplers.TPESampler(multivariate=True,
                                                                           n_startup_trials=n_startup_trials))
            study.set_metric_names(["R2_val"])
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=-1)
            self.best_models.append(objective.best_model)
            self.best_models_r_test.append(objective.test_score)
            self.best_models_r_val.append(objective.best_score)
            self.results_df = pd.concat([self.results_df, objective.model_results_df], ignore_index=True)


class SurrogateModel:
    """
    Класс для оптимизации гиперпараметров суррогатной модели
    """

    def __init__(self, model_name):
        self.model_name = model_name
        self.previous_trials = None
        self.best_model = None
        self.best_model_est = None
        self.scaler = StandardScaler()

    def _optimize_hyperparams(self, x, y, n_trials=150, n_startup_trials=20):
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # sampler = optuna.samplers.CmaEsSampler(n_startup_trials=n_startup_trials, source_trials=self.previous_trials, lr_adapt=True, warn_independent_sampling=False)
        sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)
        # sampler = optunahub.load_module("samplers/auto_sampler").AutoSampler()
        objective = Objective(x, y, self.model_name, cv=LeaveOneOut())
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.set_metric_names(["R2_val"])
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=-1)
        self.best_model_est = objective.best_model_est
        self.previous_trials = study.get_trials()

        return self.best_model_est

    def predict(self, params):
        if self.best_model == None:
            raise ValueError('Using predict before fit')
        x = np.array(params).reshape(1, -1)
        x_normalized = self.scaler.transform(x)

        if self.model_name == 'CatBoostRegressor':
            preds = self.best_model.virtual_ensembles_predict(x_normalized, prediction_type='TotalUncertainty',
                                                              virtual_ensembles_count=10)
            m = preds[:, 0]
            s = preds[:, 1]
            return m, s
        else:
            pred = self.best_model.predict(x_normalized).flatten()[0]
            return pred

    def fit(self, x, y):
        est = self._optimize_hyperparams(x, y)
        x_train = self.scaler.fit_transform(x, y)

        if self.model_name == 'CatBoostRegressor':
            est.set_params(loss_function='RMSEWithUncertainty', posterior_sampling=True)

        self.best_model = est.fit(x_train, y)


class Experiment:
    """
    Класс для итеративного проведения экспериментов.
    2 сценария:
        - автоматический (simulate_exp) для синтетических экспериментов
        - ручной (get_new_points + add_new_data) для реальных экспериментов
    """

    def __init__(self, x, y, problem, model_name, direction, n_trials, n_startup_trials):
        self.problem = problem
        self.x, self.y = x, y
        self.model_name = model_name
        self.direction = direction
        self.n_trials = n_trials
        self.n_startup_trials = n_startup_trials

        self.previous_trials = None
        self.surrogate_model = SurrogateModel(self.model_name)

    def _acquisition_function(self, params, kappa: float = 1.96) -> float:
        """
        Upper / Lower Confidence Bound
        Параметры:
        params : Iterable[float]
            Вектор параметров-кандидатов (в том же порядке, что и при обучении).
        kappa : float
            Коэффициент «смелости» (размер доверительного интервала).
            1.96 ≈ 95 %-доверие.
        """
        x = np.array(params, dtype=float).reshape(1, -1)

        # Значения по умолчанию — детерминированная модель

        if self.model_name == 'CatBoostRegressor':
            mu, var = self.surrogate_model.predict(x)
            sig = np.sqrt(var)
        else:
            mu = self.surrogate_model.predict(x)
            sig = 0.0

        # UCB / LCB в зависимости от цели оптимизации
        if self.direction == 'maximize':
            return mu + kappa * sig
        else:
            return mu - kappa * sig

    def _optimization_objective(self, trial):
        """
        Целевая функция для Optuna — значение суррогатной модели на сгенерированной точке
        """
        if self.problem == 'real':
            Pp = trial.suggest_int('Pp', 2, 4)
            U = trial.suggest_categorical('U', [50, 75, 100])
            t = trial.suggest_categorical('t', [2, 3, 5])
            L = trial.suggest_float('L', 8, 22)
            B = trial.suggest_float('B', 12, 18.5)
            params = [Pp, U, t, L, B]
        elif self.problem == 'synth':
            x1 = trial.suggest_float('x1', 0, 100)
            x2 = trial.suggest_float('x2', 0, 100)
            x3 = trial.suggest_float('x3', 0, 100)
            params = [x1, x2, x3]
        elif self.problem == 'branin':
            x1 = trial.suggest_float('x1', -5, 10)
            x2 = trial.suggest_float('x2', 0, 15)
            params = [x1, x2]
        elif self.problem == 'rosenbrock':
            x1 = trial.suggest_float('x1', -2, 2)
            x2 = trial.suggest_float('x2', -2, 2)
            params = [x1, x2]
        else:
            raise ValueError(f"Unknown problem: {self.problem}")

        score = self._acquisition_function(params, kappa=2.4)

        return score

    def get_new_point(self, sampler_name: str = 'tpe', plot: bool = False, retrain: bool = False):
        """
        Функция вычисления новой точки для исследования. В качестве целевой функции — прогноз суррогатной модели.
        Гиперпараметры суррогатной модели оптимизируются автоматически
        :param sampler_name: оптимизатор из optuna ('tpe','brute','gp','cmaes')
        :param plot: True/False построения графиков
        :return: вектор параметров x и значение y
        """
        # Нормализация и обучение суррогатной модели с оптимизацией гиперпараметров
        # self._sm_train(force=retrain)
        self.surrogate_model.fit(self.x, self.y)

        start_sampler = optuna.samplers.QMCSampler(scramble=True, warn_asynchronous_seeding=False)

        # Оптимизация целевой функции

        if sampler_name == 'tpe':
            exploit_sampler = optuna.samplers.TPESampler(multivariate=True)
        elif sampler_name == 'brute':
            exploit_sampler = optuna.samplers.BruteForceSampler(multivariate=True)
        elif sampler_name == 'gp':
            exploit_sampler = optuna.samplers.GPSampler()
        elif sampler_name == 'cmaes':
            # exploit_sampler = optuna.samplers.CmaEsSampler(n_startup_trials=self.n_startup_trials)
            exploit_sampler = optuna.samplers.CmaEsSampler(lr_adapt=True, n_startup_trials=self.n_startup_trials,
                                                           restart_strategy='ipop', inc_popsize=2)
        elif sampler_name == 'auto':
            exploit_sampler = optunahub.load_module("samplers/auto_sampler").AutoSampler()
        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")

        optimize_study = optuna.create_study(direction=self.direction, sampler=exploit_sampler)
        optimize_study.optimize(self._optimization_objective, n_trials=self.n_trials)
        self.previous_trials = optimize_study.get_trials()

        if plot:
            # Гистограмма влияния параметров
            params_importances = optuna.visualization.plot_param_importances(optimize_study)
            show(params_importances)

            # График истории оптимизации
            history = optuna.visualization.plot_optimization_history(optimize_study)
            show(history)

            # Контурный график зависимости y от значений x
            p = ["Pp", "U", "t", "L", "B"] if self.problem == 'real' else ["x1", "x2", "x3"]
            counter = optuna.visualization.plot_contour(optimize_study, params=p)
            show(counter)

        y = optimize_study.best_value
        x = optimize_study.best_trial.params

        return x, y

    def add_new_data(self, x_new, y_new):
        """
        Функция добавления новых точек к датасету
        :param x_new: вектор x
        :param y_new: значение y
        """
        self.x = np.vstack([self.x, x_new.reshape(1, -1)])
        self.y = np.append(self.y, y_new).reshape(-1, 1)

    def simulate_exp(self, synth_func, global_y_min, tol, max_experiment_iterations, sampler_name='tpe'):
        """
        Функция для симуляции эксперимента
        :param synth_func: функция, симулирующая эксперимент
        :param global_y_min: глобальный минимум функции
        :param tol: достаточное отклонение от глобального минимума
        :param max_experiment_iterations: максимальное количество итераций
        :return: x_history, y_pred_history, y_true_history, min_history, accuracy_history
        """
        accuracy_history = []
        x_history = []
        y_pred_history = []
        y_true_history = []
        min_history = []

        y_cur_min = float('inf')
        dy_min = float('inf')
        remaining_iter = max_experiment_iterations

        while dy_min >= tol and remaining_iter > 0:
            # Оптимизация на основе суррогатной модели
            x_params, y_pred = self.get_new_point(sampler_name=sampler_name, retrain=True)
            x_params = np.fromiter(x_params.values(), dtype=float)

            # Определение точного значения
            y_true = synth_func(x_params)
            y_cur_min = min(y_cur_min, y_true)

            # Вычисление разностей
            dy = abs(y_true - y_pred)
            dy_min = abs(global_y_min - y_cur_min)

            # Логирование интересующих значений
            x_history.append(x_params)
            y_pred_history.append(y_pred)
            y_true_history.append(y_true)
            accuracy_history.append(dy)
            min_history.append(dy_min)

            # Добавление точного значения в тренировочную выборку
            self.add_new_data(x_params, y_true)

            remaining_iter -= 1
        return x_history, y_pred_history, y_true_history, min_history, accuracy_history

    def simulate_exp_clustered(self, synth_func, global_y_min: float, tol: float, max_experiment_iterations: int,
                               n_clusters: int = 5, n_over: int | None = None, sampler_name: str = "tpe",
                               verbose: bool = False):
        """
        Батчевая симуляция эксперимента на основе get_new_points().
        За один «экспериментальный» шаг проверяем сразу по одной
        лучшей точке из каждого кластера.
        """
        accuracy_history: list[float] = []
        x_history: list[np.ndarray] = []
        y_pred_history: list[float] = []
        y_true_history: list[float] = []
        min_history: list[float] = []

        current_min = float("inf")
        dy_min = float("inf")
        remaining_batches = max_experiment_iterations

        # Ключи параметров (нужны для сортировки, чтобы совпадал порядок)
        param_order: list[str] | None = None

        while dy_min >= tol and remaining_batches > 0:
            # 1. Получаем батч точек-кандидатов
            batch = self.get_new_points(
                n_clusters=n_clusters,
                sampler_name=sampler_name,
                n_over=n_over,
            )

            # 2. Для каждой точки вычисляем предсказание и «истинное» значение
            for params_dict in batch:
                if param_order is None:
                    param_order = sorted(params_dict.keys())
                x_vec = np.fromiter(
                    (params_dict[k] for k in param_order), dtype=float
                )

                # surrogate prediction «на выданных» точках
                y_pred = self._sm_pred(x_vec)

                # истинное значение функции-эксперимента
                y_true = synth_func(x_vec)

                # Лог-и обновление метрик
                current_min = min(current_min, y_true)
                dy = abs(y_true - y_pred)
                dy_min = abs(global_y_min - current_min)

                x_history.append(x_vec)
                y_pred_history.append(y_pred)
                y_true_history.append(y_true)
                accuracy_history.append(dy)
                min_history.append(dy_min)

                # Добавляем новую точку в датасет и сбрасываем surrogate
                self.add_new_data(x_vec, y_true)

                # Если достигли нужной точности ― досрочно заканчиваем
                if dy_min < tol:
                    break

            remaining_batches -= 1

        return (
            x_history,
            y_pred_history,
            y_true_history,
            min_history,
            accuracy_history,
        )
