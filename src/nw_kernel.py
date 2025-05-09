import copy
import dataclasses
from typing import List, Iterator, Optional, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
from torch import Tensor
from torch.nn import Parameter
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class MLP(torch.nn.Module):
    @staticmethod
    def get_layer_set(in_features, out_features, batch_norm: float) -> List[torch.nn.Module]:
        res_l = [
            torch.nn.Linear(in_features, out_features),
            torch.nn.ReLU(),
        ]
        if batch_norm:
            res_l.insert(1, torch.nn.BatchNorm1d(out_features))

        return res_l

    def __init__(self, in_size: int, out_size: int, n_neurons: int, n_layers: int, batch_norm: int):
        super().__init__()
        assert n_neurons is not None
        self.layers = torch.nn.Sequential(
            *[
                module
                for i in range(n_layers)
                for module in self.get_layer_set(
                    in_features=in_size if i == 0 else n_neurons,
                    out_features=n_neurons,
                    batch_norm=batch_norm
                )
            ],
            torch.nn.Linear(n_neurons, out_size)
        )

    def forward(self, X):
        return self.layers(X)


class DistModel(torch.nn.Module):
    def parameters(self, **kwargs) -> List[torch.Tensor]:
        raise Exception("Should be implemented")

    def get_b_vectors(self, X: torch.Tensor) -> Iterator[Parameter]:
        raise Exception("Should be implemented")


class LinearDistModel(DistModel):
    def __init__(self, init_vect: np.ndarray):
        super().__init__()
        self.b = torch.tensor(init_vect, requires_grad=True, dtype=torch.float)

    def parameters(self, **kwargs) -> Iterator[Tensor]:
        return iter([self.b])

    def get_b_vectors(self, X) -> torch.Tensor:
        return self.b.unsqueeze(0).unsqueeze(0)


class MLPDistModel(DistModel):
    def __init__(self, in_size: int, out_size: int, n_layers: int, n_neurons: int, batch_norm: bool):
        super().__init__()
        self.mlp = MLP(in_size=in_size, out_size=out_size, n_layers=n_layers, batch_norm=batch_norm,
                       n_neurons=n_neurons)

    def parameters(self, **kwargs) -> Iterator[Parameter]:
        return self.mlp.parameters()

    def get_b_vectors(self, X) -> torch.Tensor:
        return self.mlp(X).unsqueeze(1)


class NAMDistModel(DistModel):
    def __init__(self, in_size: int, n_layers: int, n_neurons: int, batch_norm: bool):
        super().__init__()
        self.nams = [
            MLP(in_size=1, out_size=1, n_layers=n_layers, batch_norm=batch_norm, n_neurons=n_neurons)
            for i in range(in_size)
        ]

    def parameters(self, **kwargs) -> Iterator[Parameter]:
        return iter(p for nam in self.nams for p in nam.parameters())

    def get_b_vectors(self, X) -> torch.Tensor:
        res = [self.nams[dim](X[:, [dim]]) for dim in range(X.shape[-1])]
        return torch.stack(res, dim=1).swapaxes(1, 2)


def get_dist_model_by_name(name: str, f_num, n_layers: int, n_neurons: int,
                           batch_norm: bool) -> DistModel:
    if name == 'linear':
        return LinearDistModel(init_vect=np.ones(f_num) * 0.01)
    elif name == 'mlp':
        return MLPDistModel(in_size=f_num, n_layers=n_layers, n_neurons=n_neurons, batch_norm=batch_norm,
                            out_size=f_num)
    elif name == 'nam':
        return NAMDistModel(in_size=f_num, n_layers=n_layers, n_neurons=n_neurons, batch_norm=batch_norm)
    else:
        raise Exception(f"Undefined mode = {name}")


@dataclasses.dataclass
class NwKernelModel(torch.nn.Module):
    x_background: torch.Tensor
    y_background: torch.Tensor
    fit_background: bool
    dist_mode: str
    problem_mode: str
    n_layers: int
    n_neurons: int
    batch_norm: bool

    def __post_init__(self):
        super(NwKernelModel, self).__init__()

        self.dist_model = get_dist_model_by_name(
            name=self.dist_mode, f_num=self.x_background.shape[-1],
            n_layers=self.n_layers, n_neurons=self.n_neurons,
            batch_norm=self.batch_norm
        )

        self.softmax = torch.nn.Softmax()

        if self.fit_background:
            self.y_background = torch.nn.Parameter(self.y_background, requires_grad=True)
        else:
            pass

    def parameters(self, **kwargs) -> List[Tensor]:
        if self.fit_background:
            return [self.y_background, *list(self.softmax.parameters()), *list(self.dist_model.parameters())]
        else:
            return [*list(self.softmax.parameters()), *list(self.dist_model.parameters())]

    def forward(self, X: torch.Tensor):
        xps_rep = X.unsqueeze(1)
        x_background_rep = self.x_background.unsqueeze(0)
        b_vectors = self.dist_model.get_b_vectors(X=X)

        distances = (torch.abs(b_vectors) * torch.abs(xps_rep - x_background_rep)).sum(axis=-1)

        weights = torch.exp(-distances)
        weights_norm = weights / weights.sum(axis=1).unsqueeze(1)

        y_background_rep = self.y_background.swapaxes(0, 1)
        if self.problem_mode == 'reg':
            return (y_background_rep * weights_norm).sum(axis=1).unsqueeze(1)
        else:
            raise ValueError(self.problem_mode)


@dataclasses.dataclass
class NWScikit(BaseEstimator, RegressorMixin):
    batch_size: int
    epoch_n: int = 200
    pred_batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 0
    background_lr: float = 1e-3
    background_weight_decay: float = 0
    dist_mode: str = 'mlp'
    kernel_fit_background: bool = True
    problem_mode: str = 'reg'
    optimizer: str = 'SGD'
    momentum: Optional[float] = 0.95
    n_layers: Optional[int] = None
    n_neurons: Optional[int] = None
    batch_norm: Optional[bool] = False
    background_part: Optional[float] = None
    verbose: bool = None
    verbose_tqdm: bool = None
    _model = None
    _fit_history = None

    def __post_init__(self):
        if self.verbose is None:
            self.verbose = False
        if self.verbose_tqdm is None:
            self.verbose_tqdm = False

    @staticmethod
    def _torch_cast(arr) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            return arr
        elif isinstance(arr, pd.DataFrame):
            return torch.tensor(arr.to_numpy(), dtype=torch.float)
        elif isinstance(arr, np.ndarray):
            return torch.tensor(arr, dtype=torch.float32)
        else:
            raise TypeError(type(arr))

    def fit(self, X: Union[pd.DataFrame, np.ndarray, torch.Tensor], y: Union[pd.DataFrame, np.ndarray, torch.Tensor]):
        X = self._torch_cast(X)
        y = self._torch_cast(y)

        def verbose_print(*args):
            if self.verbose:
                print(*args)

        if self.problem_mode == 'reg':
            criterion = torch.nn.MSELoss()
        else:
            raise ValueError(f"Unexpected problem_mode = {self.problem_mode}")

        dataloader = DataLoader(
            dataset=TensorDataset(X, y),
            batch_size=self.batch_size,
            shuffle=False
        )

        if self.background_part is not None:
            assert 0 < self.background_part <= 1., self.background_part
            if self.problem_mode == 'reg':
                save_ids = np.linspace(0, len(X) - 1, int(len(X) * self.background_part), dtype=int)
                x_background = X[save_ids].detach().clone()
                y_background = y[save_ids].detach().clone()
            else:
                raise ValueError(self.problem_mode)
        else:
            x_background = X.detach().clone()
            y_background = y.detach().clone()

        model = NwKernelModel(
            x_background=x_background, y_background=y_background,
            fit_background=self.kernel_fit_background,
            dist_mode=self.dist_mode,
            problem_mode=self.problem_mode,
            n_layers=self.n_layers, n_neurons=self.n_neurons, batch_norm=self.batch_norm
        )

        history = []
        if self.optimizer == 'Adam':
            opt_claz = torch.optim.Adam
        elif self.optimizer == 'SGD':
            opt_claz = torch.optim.SGD
        else:
            raise ValueError(self.optimizer)

        optimizer = opt_claz(
            [
                dict(params=model.dist_model.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum = self.momentum),
                dict(params=model.y_background, lr=self.background_lr, weight_decay=self.background_weight_decay),
            ]
        )

        it = tqdm(range(self.epoch_n)) if self.verbose_tqdm else range(self.epoch_n)

        best_loss = None
        best_state = None

        for epoch_i in it:
            for x_batch, y_batch in dataloader:
                optimizer.zero_grad()
                y_pred = model(x_batch)
                loss = criterion(y_batch, y_pred)
                loss.backward()
                optimizer.step()

            self._model = model

        #model.load_state_dict(best_state)

        self._model = model
        self._fit_history = pd.DataFrame(history)

    def predict(self, X: Union[pd.DataFrame, np.ndarray, torch.Tensor]) -> np.ndarray:
        assert self._model is not None, 'Calling predict(..) before fit(..)'
        X = self._torch_cast(X)
        predloader = DataLoader(
            dataset=TensorDataset(X),
            batch_size=self.pred_batch_size,
            shuffle=False
        )

        preds: List[np.ndarray] = []
        with torch.no_grad():
            for (x_batch,) in predloader:
                x_batch = x_batch
                output = self._model(x_batch)
                preds.append(output.detach().cpu().numpy())

        return np.vstack(preds)

    def __sklearn_is_fitted__(self):
        return self._model is not None
