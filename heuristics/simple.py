import typing

import numpy as np
import tqdm.notebook as tqdm
import pandas as pd

from .base import Heuristic


class ScalarScaleDistanse(Heuristic):

    _user_id2scalar: typing.Dict[typing.Hashable, typing.Union[float, int]]
    _item_id2scalar: typing.Dict[typing.Hashable, typing.Union[float, int]]

    def __init__(self):
        self.name = 'scalar_scale_distanse'

    def fit(self, interactions: pd.DataFrame, items_features: pd.DataFrame):
        assert len(items_features.columns) == 2, 'Dataframe must have 2 columns - [id, feature]'

        self._item_id2scalar = items_features.drop_duplicates(subset=['item_id']).set_index('item_id')['scalars'].to_dict()
        self._user_id2scalar = interactions.groupby('user_id')['item_id'].apply(lambda items: items.apply(self._item_id2scalar.__getitem__).mean()).to_dict()        
        self.__fitted = True

    def _idx2distanse(self, uid: typing.Any, iid: typing.Any) -> np.float32:
        v1 = self._user_id2scalar[uid]
        v2 = self._item_id2scalar[iid]
        
        return min(v1, v2) / max(v1, v2)

    def _predict(self, user_idx: typing.Iterable, item_idx: typing.Iterable) -> np.ndarray:
        cos_gen = tqdm.tqdm(
            (self._idx2distanse(v1, v2) for v1, v2 in zip(user_idx, item_idx)),
            total=len(user_idx)
        )
        return np.fromiter(cos_gen, dtype=np.float32)

    def predict_user(self, user_id: typing.Any, item_idx: typing.Iterable) -> pd.Series:
        assert self.__fitted, 'Call Hueristic.fit(*args, **kwargs) before make predicts'
        assert user_id in set(self._user_id2scalar.keys()), 'User was not in train data'
        assert len(set(item_idx) - set(self._item_id2scalar.keys())) == 0, 'Not all items were in train data'

        return pd.Series(
            self._predict(np.repeat(user_id, len(item_idx)), item_idx),
            index=item_idx.index if isinstance(item_idx, pd.Series) else None
        )

    def predict(self, user_idx: typing.Iterable, item_idx: typing.Iterable) -> pd.Series:
        assert self.__fitted, 'Call Hueristic.fit(*args, **kwargs) before make predicts'
        assert len(set(user_idx) - set(self._user_id2scalar.keys())) == 0, 'Not all users were in train data'
        assert len(set(item_idx) - set(self._item_id2scalar.keys())) == 0, 'Not all items were in train data'

        return pd.Series(
            self._predict(user_idx, item_idx),
            index=user_idx.index if isinstance(user_idx, pd.Series) else None
        )


class ScalarModuleDistanse(Heuristic):

    _user_id2scalar: typing.Dict[typing.Hashable, float | int]
    _item_id2scalar: typing.Dict[typing.Hashable, float | int]

    def __init__(self):
        self.name = 'scalar_module_distanse'

    def fit(self, interactions: pd.DataFrame, items_features: pd.DataFrame):
        assert len(items_features.columns) == 2, 'Dataframe must have 2 columns - [id, feature]'

        self._item_id2scalar = items_features.drop_duplicates(subset=['item_id']).set_index('item_id')['scalars'].to_dict()
        self._user_id2scalar = interactions.groupby('user_id')['item_id'].apply(lambda items: items.apply(self._item_id2scalar.__getitem__).mean()).to_dict()        
        self.__fitted = True

    def _idx2distanse(self, uid: typing.Any, iid: typing.Any) -> np.float32:
        v1 = self._user_id2scalar[uid]
        v2 = self._item_id2scalar[iid]
        
        return abs(v1-v2)

    def _predict(self, user_idx: typing.Iterable, item_idx: typing.Iterable) -> np.ndarray:
        cos_gen = tqdm.tqdm(
            (self._idx2distanse(uid=v1, iid=v2) for v1, v2 in zip(user_idx, item_idx)),
            total=len(user_idx)
        )
        return np.fromiter(cos_gen, dtype=np.float32)

    def predict_user(self, user_id: typing.Any, item_idx: typing.Iterable) -> pd.Series:
        assert self.__fitted, 'Call Hueristic.fit(*args, **kwargs) before make predicts'
        assert user_id in set(self._user_id2scalar.keys()), 'User was not in train data'
        assert len(set(item_idx) - set(self._item_id2scalar.keys())) == 0, 'Not all items were in train data'

        return pd.Series(
            self._predict(np.repeat(user_id, len(item_idx)), item_idx),
            index=item_idx.index if isinstance(item_idx, pd.Series) else None
        )

    def predict(self, user_idx: typing.Iterable, item_idx: typing.Iterable) -> pd.Series:
        assert self.__fitted, 'Call Hueristic.fit(*args, **kwargs) before make predicts'
        assert len(set(user_idx) - set(self._user_id2scalar.keys())) == 0, 'Not all users were in train data'
        assert len(set(item_idx) - set(self._item_id2scalar.keys())) == 0, 'Not all items were in train data'

        return pd.Series(
            self._predict(user_idx, item_idx),
            index=user_idx.index if isinstance(user_idx, pd.Series) else None
        )