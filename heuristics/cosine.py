import typing

import numpy as np
import pandas as pd
import tqdm.notebook as tqdm

from .base import Heuristic


class Cosine(Heuristic):

    _user_id2vec: typing.Dict[typing.Hashable, np.ndarray]
    _item_id2vec: typing.Dict[typing.Hashable, np.ndarray]

    def __init__(self):
        self.name = 'cosine'

    def fit(self, interactions: pd.DataFrame, items_features: pd.DataFrame):
        item_id2vec = pd.DataFrame({
            'item_id': items_features['item_id'],
            'vectors': items_features.drop(columns='item_id').parallel_apply(lambda row: row.values, axis=1)
        })
        self._item_id2vec = item_id2vec.drop_duplicates(subset=['item_id']).set_index('item_id')['vectors'].to_dict()
        self._user_id2vec = interactions.groupby('user_id')['item_id'].apply(lambda items: items.apply(self._item_id2vec.__getitem__).mean()).to_dict()        
        self.__fitted = True

    def _idx2cosine(self, uid: typing.Any, iid: typing.Any) -> np.float32:
        v1 = self._user_id2vec[uid]
        v2 = self._item_id2vec[iid]
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def _predict(self, user_idx: typing.Iterable, item_idx: typing.Iterable) -> np.ndarray:
        cos_gen = tqdm.tqdm(
            (self._idx2cosine(uid=v1, iid=v2) for v1, v2 in zip(user_idx, item_idx)),
            total=len(user_idx)
        )
        return np.fromiter(cos_gen, dtype=np.float32)

    def predict_user(self, user_id: typing.Any, item_idx: typing.Iterable) -> pd.Series:
        assert self.__fitted, 'Call Hueristic.fit(*args, **kwargs) before make predicts'
        assert user_id in set(self._user_id2vec.keys()), 'User was not in train data'
        assert len(set(item_idx) - set(self._item_id2vec.keys())) == 0, 'Not all items were in train data'

        return pd.Series(
            self._predict(np.repeat(user_id, len(item_idx)), item_idx),
            index=item_idx.index if isinstance(item_idx, pd.Series) else None
        )

    def predict(self, user_idx: typing.Iterable, item_idx: typing.Iterable) -> pd.Series:
        assert self.__fitted, 'Call Hueristic.fit(*args, **kwargs) before make predicts'
        assert len(set(user_idx) - set(self._user_id2vec.keys())) == 0, 'Not all users were in train data'
        assert len(set(item_idx) - set(self._item_id2vec.keys())) == 0, 'Not all items were in train data'

        return pd.Series(
            self._predict(user_idx, item_idx),
            index=user_idx.index if isinstance(user_idx, pd.Series) else None
        )

