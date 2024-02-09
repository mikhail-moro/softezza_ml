import typing

import numpy as np
import pandas as pd

from .base import Heuristic


class Cosine(Heuristic):
    _user_id2vec: typing.Dict[typing.Hashable, np.ndarray]
    _item_id2vec: typing.Dict[typing.Hashable, np.ndarray]

    def __init__(self):
        self.name = 'cosine'

    def fit(self, interactions: pd.DataFrame, items_features: pd.DataFrame):
        item_id2vec = pd.DataFrame({
            'item_id': items_features['item_id'],
            'vectors': items_features.drop(columns='item_id').apply(lambda row: row.values, axis=1)
        })
        self._item_id2vec = item_id2vec.drop_duplicates(subset=['item_id']).set_index('item_id')['vectors'].to_dict()
        self._user_id2vec = interactions.groupby('user_id')['item_id'].apply(lambda items: items.apply(self._item_id2vec.__getitem__).mean()).to_dict()        
        self.fitted = True

    def predict_user(self, user_id: typing.Any, item_idx: typing.Iterable) -> pd.Series:
        assert self.fitted, 'Call Hueristic.fit(*args, **kwargs) before make predicts'
        assert user_id in set(self._user_id2vec.keys()), 'User was not in train data'
        assert len(set(item_idx) - set(self._item_id2vec.keys())) == 0, 'Not all items were in train data'

        return pd.Series(
            [self._vecs2cosine(self._user_id2vec[user_id], self._item_id2vec[item_id]) for item_id in item_idx],
            index=item_idx.index if isinstance(item_idx, pd.Series) else None
        )

    def predict(self, user_idx: typing.Iterable, item_idx: typing.Iterable) -> pd.Series:
        assert self.fitted, 'Call Heuristic.fit(*args, **kwargs) before make predicts'
        assert len(set(user_idx) - set(self._user_id2vec.keys())) == 0, 'Not all users were in train data'
        assert len(set(item_idx) - set(self._item_id2vec.keys())) == 0, 'Not all items were in train data'

        return pd.Series(
            [self._vecs2cosine(self._user_id2vec[user_id], self._item_id2vec[item_id]) for user_id, item_id in zip(user_idx, item_idx)],
            index=user_idx.index if isinstance(user_idx, pd.Series) else None
        )

    def _vecs2cosine(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
