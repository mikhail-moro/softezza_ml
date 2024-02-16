import typing
import abc

import numpy as np
import tqdm.notebook as tqdm
import pandas as pd

from ..data import Data
from .base import Heuristic


class _ScalarDistance(Heuristic, abc.ABC):

    _user_id2scalar: typing.Dict[typing.Hashable, typing.Union[float, int]]
    _item_id2scalar: typing.Dict[typing.Hashable, typing.Union[float, int]]
    scalar_column: str

    def __init__(self, scalar_column: str, verbose: bool = False):
        super().__init__(verbose)
        self.scalar_column = scalar_column

    @abc.abstractmethod
    def _idx2distance(self, uid: typing.Any, iid: typing.Any) -> float:
        raise NotImplementedError()

    def fit(self, x: pd.DataFrame, y: pd.DataFrame = None, data: Data = None):
        items_features = data.item_features

        self._item_id2scalar = items_features.drop_duplicates(subset=['item_id']).set_index('item_id')[self.scalar_column].to_dict()
        self._user_id2scalar = x.groupby('user_id')['item_id'].apply(lambda items: items.apply(self._item_id2scalar.__getitem__).mean()).to_dict()
        self.__fitted = True

    def _predict(self, user_idx: typing.Iterable, item_idx: typing.Iterable) -> np.ndarray:
        cos_gen = tqdm.tqdm(
            (self._idx2distance(v1, v2) for v1, v2 in zip(user_idx, item_idx)),
            total=len(user_idx)
        )
        return np.fromiter(cos_gen, dtype=np.float32)

    def rerank(self, candidates: pd.DataFrame, data: Data = None) -> pd.Series:
        assert self.__fitted, 'Call Heuristic.fit(*args, **kwargs) before make predicts'
        assert len(set(candidates['user_id'].values) - set(self._user_id2scalar.keys())) == 0, 'Not all users were in train data'
        assert len(set(candidates['item_id'].values) - set(self._item_id2scalar.keys())) == 0, 'Not all items were in train data'

        return pd.Series(
            self._predict(candidates['user_id'].values, candidates['user_id'].values),
            index=candidates.index
        )


class ScalarScaleDistance(_ScalarDistance):
    def _idx2distance(self, uid: typing.Any, iid: typing.Any) -> float:
        v1 = self._user_id2scalar[uid]
        v2 = self._item_id2scalar[iid]
        
        return min(v1, v2) / max(v1, v2)
