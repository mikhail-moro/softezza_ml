import typing
import abc

import numpy as np
import pandas as pd
import tqdm.notebook as tqdm

from .base import Heuristic
from ..data import Data


class _Cosine(Heuristic, abc.ABC):
    _user_id2vec: typing.Dict[typing.Hashable, np.ndarray]
    _item_id2vec: typing.Dict[typing.Hashable, np.ndarray]
    features_columns: typing.List[str]

    def __init__(self, features_columns: typing.Iterable[str], verbose: bool = False):
        super().__init__(verbose)
        self.features_columns = list(features_columns)

    @abc.abstractmethod
    def _get_user_vec(self, items: pd.Series) -> np.ndarray:
        raise NotImplementedError()

    def fit(self, x: pd.DataFrame, y: pd.DataFrame = None, data: Data = None):
        items_features = data.item_features[['item_id'] + self.features_columns]

        item_id2vec = pd.DataFrame({
            'item_id': items_features['item_id'],
            'vectors': items_features.drop(columns='item_id').apply(lambda row: row.values, axis=1)
        })
        self._item_id2vec = item_id2vec.drop_duplicates(subset=['item_id']).set_index('item_id')['vectors'].to_dict()
        self._user_id2vec = x.groupby('user_id')['item_id'].apply(self._get_user_vec).to_dict()
        self.__fitted = True

    def _idx2cosine(self, uid: typing.Any, iid: typing.Any) -> np.float32:
        v1 = self._user_id2vec[uid]
        v2 = self._item_id2vec[iid]
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def _predict(self, user_idx: typing.Iterable, item_idx: typing.Iterable) -> np.ndarray:
        cos_gen = tqdm.tqdm(
            (self._idx2cosine(uid=v1, iid=v2) for v1, v2 in zip(user_idx, item_idx)),
            total=len(user_idx),
            disable=not self._verbose
        )
        return np.fromiter(cos_gen, dtype=np.float32)

    def rerank(self, candidates: pd.DataFrame, data: Data = None) -> pd.Series:
        assert self.__fitted, 'Call Heuristic.fit(*args, **kwargs) before make predicts'
        assert len(set(candidates['user_id']) - set(self._user_id2vec.keys())) == 0, 'Not all users were in train data'
        assert len(set(candidates['item_id']) - set(self._item_id2vec.keys())) == 0, 'Not all items were in train data'

        return pd.Series(
            self._predict(candidates['user_id'].values, candidates['item_id']),
            index=candidates.index
        )


class Cosine(_Cosine):

    def _get_user_vec(self, items: pd.Series) -> np.ndarray:
        return items.apply(self._item_id2vec.__getitem__).mean()

    def __init__(self, verbose: bool = False):
        Heuristic.__init__(self, verbose)


# TODO Weighted time-delayed cosine distanse - user vector with bias on latest item in user history
class TimeDelayedWeightedCosine(_Cosine):
    def _get_user_vec(self, items: pd.Series) -> np.ndarray:
        raise NotImplementedError()

    def __init__(self, verbose: bool = False):
        Heuristic.__init__(self, verbose)
        raise NotImplementedError()
