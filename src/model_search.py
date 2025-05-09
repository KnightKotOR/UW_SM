from copy import deepcopy

import numpy as np
import optuna
import pandas as pd
import warnings

import scipy.optimize
from catboost import CatBoostRegressor
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, \
    HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression, SGDRegressor, Lasso, LassoLars, ARDRegression, \
    BayesianRidge
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from nw_kernel import NWScikit

from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict, StratifiedKFold
from scipy.optimize import fmin
from plotly.io import show

warnings.filterwarnings('ignore')


class Objective(object):
    def __init__(self, X, y, X_test, y_test, models_list, cv):
        self.X, self.y = X, y
        self.X_test, self.y_test = X_test, y_test
        self.models_list = models_list[0]
        self.cv = cv
        self.model_results_df = pd.DataFrame(
            columns=['n', 'CV_mode', 'Model', 'R2_val', 'R2_test', 'Parameters'])
        self.best_model = None
        self.best_score = float('-inf')
        self.test_score = float('-inf')
        self.y_pred = None

    def __call__(self, trial):
        warnings.filterwarnings('ignore')
        # Определение модели и ее параметров
        # regressor_name = trial.suggest_categorical("regressor", self.models_list)
        regressor_name = self.models_list
        if regressor_name == "CatBoostRegressor":
            lr = trial.suggest_float("learning_rate", 1e-4, 5e-1, log=True)
            depth = trial.suggest_int("depth", 1, 8)
            l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1e-6, 10, log=True)
            bagging_temp = trial.suggest_float('bagging_temperature', 0.0, 1.0)
            random_strength = trial.suggest_float('random_strength', 1e-6, 10, log=True)
            grow_policy = trial.suggest_categorical("grow_policy", ['SymmetricTree', 'Depthwise', 'Lossguide'])
            regressor_obj = CatBoostRegressor(iterations=200, learning_rate=lr, depth=depth, l2_leaf_reg=l2_leaf_reg,
                                              bagging_temperature=bagging_temp, random_strength=random_strength,
                                              grow_policy=grow_policy, verbose=0)
        elif regressor_name == "XGBRegressor":
            lr = trial.suggest_float("learning_rate", 1e-4, 5e-1, log=True)
            max_depth = trial.suggest_int("max_depth", 1, 40)
            max_leaves = trial.suggest_int('max_leaves', 1, 40)
            min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
            reg_alpha = trial.suggest_float('reg_alpha', 1e-8, 10, log=True)
            reg_lambda = trial.suggest_float('reg_lambda', 1e-8, 10, log=True)
            subsample = trial.suggest_float('subsample', 0.5, 1.0)
            colsample = trial.suggest_float('colsample_bytree', 0.5, 1.0)
            regressor_obj = XGBRegressor(
                max_depth=max_depth, max_leaves=max_leaves, n_estimators=100, learning_rate=lr, reg_alpha=reg_alpha,
                reg_lambda=reg_lambda, subsample=subsample, colsample_bytree=colsample,
                min_child_weight=min_child_weight, n_jobs=-1
            )
        elif regressor_name == "RandomForestRegressor":
            n = trial.suggest_int("n_estimators", 50, 500)
            depth = trial.suggest_int("max_depth", 1, 50)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 50)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
            max_features = trial.suggest_float('max_features', 0.1, 1.0, log=True)
            regressor_obj = RandomForestRegressor(n_estimators=n, max_depth=depth, min_samples_split=min_samples_split,
                                                  min_samples_leaf=min_samples_leaf, max_features=max_features,
                                                  verbose=0)
        elif regressor_name == "GradientBoostingRegressor":
            lr = trial.suggest_float("learning_rate", 1e-4, 5e-1, log=True)
            depth = trial.suggest_int("max_depth", 1, 60)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 60)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
            max_features = trial.suggest_float('max_features', 0.1, 1.0, log=True)
            subsample = trial.suggest_float('subsample', 0.5, 1.0)
            regressor_obj = GradientBoostingRegressor(
                max_depth=depth, min_samples_split=min_samples_split, n_estimators=250, learning_rate=lr,
                min_samples_leaf=min_samples_leaf, max_features=max_features, subsample=subsample, verbose=0
            )
        elif regressor_name == "ExtraTreesRegressor":
            n = trial.suggest_int("n_estimators", 50, 500)
            depth = trial.suggest_int("max_depth", 1, 50)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 50)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
            max_features = trial.suggest_float('max_features', 0.1, 1.0, log=True)
            regressor_obj = ExtraTreesRegressor(n_estimators=n, max_depth=depth, min_samples_split=min_samples_split,
                                                min_samples_leaf=min_samples_leaf, max_features=max_features,
                                                verbose=0)
        elif regressor_name == "HistGradientBoostingRegressor":
            lr = trial.suggest_float("learning_rate", 1e-4, 5e-1, log=True)
            depth = trial.suggest_int("max_depth", 1, 10)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
            max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 10, 100)
            max_features = trial.suggest_float('max_features', 0.1, 1.0, log=True)
            l2_regularization = trial.suggest_float('l2_regularization', 0.0, 1.0)
            regressor_obj = HistGradientBoostingRegressor(
                max_depth=depth, max_leaf_nodes=max_leaf_nodes, max_iter=500, learning_rate=lr,
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
        elif regressor_name == "Ridge":
            alpha = trial.suggest_float('alpha', 1e-4, 10, log=True)
            regressor_obj = Ridge(alpha=alpha)
        elif regressor_name == "ElasticNet":
            alpha = trial.suggest_float('alpha', 1e-4, 10, log=True)
            l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
            max_iter = trial.suggest_int('max_iter', 50, 500)
            regressor_obj = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter)
        elif regressor_name == "LinearRegression":
            regressor_obj = LinearRegression(n_jobs=-1)
        elif regressor_name == "SGDRegressor":
            alpha = trial.suggest_float("alpha", 1e-6, 1.0, log=True)
            lr_schedule = trial.suggest_categorical("learning_rate", ["constant", "optimal", "invscaling", "adaptive"])
            eta0 = trial.suggest_float("eta0", 1e-4, 1.0, log=True)
            penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])
            max_iter = trial.suggest_int("max_iter", 1000, 10000)
            regressor_obj = SGDRegressor(alpha=alpha, learning_rate=lr_schedule, eta0=eta0,
                                         penalty=penalty, max_iter=max_iter)
        elif regressor_name == "Lasso":
            alpha = trial.suggest_float("alpha", 1e-4, 1.0, log=True)
            max_iter = trial.suggest_int("max_iter", 1000, 10000)
            regressor_obj = Lasso(alpha=alpha, max_iter=max_iter)
        elif regressor_name == "LassoLars":
            alpha = trial.suggest_float("alpha", 1e-4, 1.0, log=True)
            max_iter = trial.suggest_int("max_iter", 500, 5000)
            regressor_obj = LassoLars(alpha=alpha, max_iter=max_iter)
        elif regressor_name == "ARDRegression":
            alpha_1 = trial.suggest_float("alpha_1", 1e-8, 1e-1, log=True)
            alpha_2 = trial.suggest_float("alpha_2", 1e-8, 1e-1, log=True)
            lambda_1 = trial.suggest_float("lambda_1", 1e-8, 1e-1, log=True)
            lambda_2 = trial.suggest_float("lambda_2", 1e-8, 1e-1, log=True)
            regressor_obj = ARDRegression(alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1, lambda_2=lambda_2)
        elif regressor_name == "BayesianRidge":
            alpha_1 = trial.suggest_float("alpha_1", 1e-8, 1e-1, log=True)
            alpha_2 = trial.suggest_float("alpha_2", 1e-8, 1e-1, log=True)
            lambda_1 = trial.suggest_float("lambda_1", 1e-8, 1e-1, log=True)
            lambda_2 = trial.suggest_float("lambda_2", 1e-8, 1e-1, log=True)
            regressor_obj = BayesianRidge(alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1, lambda_2=lambda_2)
        elif regressor_name == "GaussianProcessRegressor":
            alpha = trial.suggest_float("alpha", 1e-10, 1e-1, log=True)
            regressor_obj = GaussianProcessRegressor(alpha=alpha)
        elif regressor_name == "KNeighborsRegressor":
            n = trial.suggest_int("n_neighbors", 1, 50)
            weights = trial.suggest_categorical("weights", ["uniform", "distance"])
            leaf = trial.suggest_int("leaf_size", 10, 100)
            p = trial.suggest_int("p", 1, 2)
            regressor_obj = KNeighborsRegressor(n_neighbors=n, weights=weights, leaf_size=leaf, p=p)
        elif regressor_name == "RadiusNeighborsRegressor":
            radius = trial.suggest_float("radius", 1.2, 5.0, log=True)
            weights = trial.suggest_categorical("weights", ["uniform", "distance"])
            leaf = trial.suggest_int("leaf_size", 10, 100)
            regressor_obj = RadiusNeighborsRegressor(radius=radius, weights=weights, leaf_size=leaf)
        elif regressor_name == "SVR":
            C = trial.suggest_float("C", 1e-3, 1e3, log=True)
            kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
            gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
            degree = trial.suggest_int("degree", 2, 5)
            regressor_obj = SVR(C=C, kernel=kernel, gamma=gamma, degree=degree)
        elif regressor_name == "KernelRidge":
            alpha = trial.suggest_float("alpha", 1e-10, 1e-1, log=True)
            kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
            degree = trial.suggest_int("degree", 2, 5)
            regressor_obj = KernelRidge(alpha, kernel=kernel, degree=degree)
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

        regressor_obj.fit(self.X, self.y)
        y_test_pred = regressor_obj.predict(self.X_test)
        r2_test = r2_score(self.y_test, y_test_pred)

        if r2_val > self.best_score:
            self.best_model = regressor_obj
            self.best_score = r2_val
            self.test_score = r2_test
            self.y_pred = y_test_pred

        # Обновление дф
        self.model_results_df = pd.concat([self.model_results_df, pd.DataFrame({
            'n': trial.number,
            'CV_mode': cv_mode,
            'Model': regressor_name,
            'Parameters': [trial.params],
            'R2_val': r2_val,
            'R2_test': r2_test
        })], ignore_index=True)

        return r2_val

    def get_results(self):
        return self.model_results_df


class OptunaSearchCV:
    def __init__(self, models_list, compare_kfold):
        self.models_list = models_list
        self.results_df = pd.DataFrame(columns=['n', 'CV_mode', 'Model', 'R2_val', 'R2_test', 'Parameters', 'object'])
        self.best_models = []
        self.best_models_y_pred = {}
        self.best_models_r_test = []
        self.best_models_r_val = []
        self.opt_study_storage = optuna.storages.InMemoryStorage()
        self.kfold = compare_kfold

    def fit(self, x, y, x_test, y_test, cv_list, n_trials=100, n_startup_trials=20):

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        hyperopt_storage = optuna.storages.InMemoryStorage()
        for model in self.models_list:
            print(f"\n{model} hyperoptimization")
            for cv_mode in cv_list:
                print(f"{cv_mode} cross-validation")
                cv = LeaveOneOut() if cv_mode == 'loo' else int(cv_mode)
                objective = Objective(x, y, x_test, y_test, [model], cv=cv)
                study = optuna.create_study(storage=hyperopt_storage, study_name=f"synth_{model}_LOOCV",
                                            direction="maximize",
                                            sampler=optuna.samplers.TPESampler(multivariate=True,
                                                                               n_startup_trials=n_startup_trials))
                study.set_metric_names(["R2_val"])
                study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
                self.best_models.append(objective.best_model)
                self.best_models_y_pred[type(objective.best_model).__name__] = objective.y_pred
                self.best_models_r_test.append(objective.test_score)
                self.best_models_r_val.append(objective.best_score)
                self.results_df = pd.concat([self.results_df, objective.get_results()], ignore_index=True)

    def optimize(self, model_list, scaler, direction='maximize', problem='real', plot=False, continue_study=False,
                 n_trials=100, n_startup_trials=50, verbose=False):
        y, x = None, None

        def objective(trial):
            if problem == 'real':
                Pp = trial.suggest_int('Pp', 2, 4)
                U = trial.suggest_categorical('U', [50, 75, 100])
                t = trial.suggest_categorical('t', [2, 3, 5])
                L = trial.suggest_float('L', 8, 22)
                B = trial.suggest_float('B', 12, 18.5)
                params = [Pp, U, t, L, B]
            else:
                x1 = trial.suggest_float('x1', 0, 100)
                x2 = trial.suggest_float('x2', 0, 100)
                x3 = trial.suggest_float('x3', 0, 100)
                params = [x1, x2, x3]

            y_pred = func(params)
            return y_pred

        for i, _model in enumerate(model_list):
            def func(params):
                """
                Аппроксимирующая функция суррогатной модели
                :param params: вектор x
                :return: результат прогнозирования суррогатной модели
                """
                x = np.array(params).reshape(1, -1)
                x_normalized = scaler.transform(x)
                y_pred = _model.predict(x_normalized).flatten()[0]
                return y_pred

            sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=n_startup_trials, constant_liar=True)
            #sampler = optuna.samplers.GPSampler(n_startup_trials=n_startup_trials)
            #sampler = optuna.samplers.CmaEsSampler(n_startup_trials=n_startup_trials)
            study_name = f"optimize_study_{type(_model).__name__}"

            optimize_study = optuna.create_study(direction=direction, study_name=study_name, sampler=sampler)
            optimize_study.optimize(objective, n_trials=n_trials)

            if plot:
                # Гистограмма влияния параметров
                params_importances = optuna.visualization.plot_param_importances(optimize_study)
                show(params_importances)

                # График истории оптимизации
                history = optuna.visualization.plot_optimization_history(optimize_study)
                show(history)

                # Контурный график зависимости y от значений x
                p = ["Pp", "U", "t", "L", "B"] if problem == 'real' else ["x1", "x2", "x3"]
                counter = optuna.visualization.plot_contour(optimize_study, params=p)
                show(counter)

            model_name = type(_model).__name__
            y = optimize_study.best_value
            x = optimize_study.best_trial.params
            
            if verbose:
                print(f'{model_name} found {y} with params {x}')

        return y, x

    def simulate_experiment(self, initial_func, surrogate_model, scaler, x_train, y_train, x_min, y_min, tol=0.1,
                            max_iter=50, n_trials=50, n_startup_trials=50, direction='minimize'):
        """
        Моделирование итеративной оптимизации физического эксперимента с использованием суррогатной модели
        """
        # _surrogate_model = deepcopy(surrogate_model)
        _scaler = deepcopy(scaler)
        _x_train_scaled = deepcopy(x_train)

        continue_study = False
        accuracy_history = []
        x_history = []
        y_history = []
        y_true_history = []
        min_history = []
        dx = float('inf')
        remaining_iter = max_iter

        while dx >= tol and remaining_iter > 0:
            _surrogate_model = deepcopy(surrogate_model)
            _surrogate_model.fit(_x_train_scaled, y_train)

            y_pred, x_params = self.optimize([_surrogate_model], _scaler, direction=direction, problem='synth',
                                             continue_study=continue_study, n_trials=n_trials,
                                             n_startup_trials=n_startup_trials)
            
            x_params = np.fromiter(x_params.values(), dtype=float)
            y_true = initial_func(x_params)
            dx = abs(x_params[0] - x_min)
            x_history.append(x_params)
            y_history.append(y_pred)
            y_true_history.append(y_true)

            x_params_norm = _scaler.transform(x_params.reshape(1, -1))
            _x_train_scaled = np.vstack([_x_train_scaled, x_params_norm])
            y_train = np.append(y_train, y_true).reshape(-1, 1)

            dy = abs(y_true - y_pred)
            dy_min = abs(y_min - y_pred)

            accuracy_history.append(dy)
            min_history.append(dy_min)

            # _surrogate_model.fit(x_train, y_train)

            continue_study = True
            remaining_iter -= 1

        return accuracy_history, min_history, x_history, y_history, y_true_history
