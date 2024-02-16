import abc
import typing
import pandas as pd

from ..data import Data
from xgboost import XGBRanker


class Ranker(abc.ABC):
    _verbose: bool

    def __init__(self, verbose: bool = False) -> None:
        self._verbose = verbose

    @abc.abstractmethod
    def fit(self, x: pd.DataFrame, y: pd.DataFrame = None, data: Data = None) -> typing.Union[pd.Series, pd.DataFrame]:
        raise NotImplementedError()

    @abc.abstractmethod
    def rerank(self, candidates: pd.DataFrame, data: Data = None) -> typing.Union[pd.Series, pd.DataFrame]:
        raise NotImplementedError()


class Heuristic(Ranker, abc.ABC):
    __fitted: bool = False


class XGBoostRanker(XGBRanker, Ranker):
    user_features_columns: typing.Iterable[str]
    item_features_columns: typing.Iterable[str]

    def __init__(
        self,
        verbose: bool,
        user_features_columns: typing.Iterable[str] = None,
        item_features_columns: typing.Iterable[str] = None,
        **xgbranker_kwargs
    ):
        XGBRanker.__init__(self, **xgbranker_kwargs)
        Ranker.__init__(self, verbose)

        self.user_features_columns = user_features_columns
        self.item_features_columns = item_features_columns

    def _prepare_df(self, df, data):
        df = pd.merge(df, data.item_features[['item_id'] + list(self.item_features_columns)], on='item_id')
        df = pd.merge(df, data.user_features[['user_id'] + list(self.user_features_columns)], on='user_id')

        return df.rename(columns={'user_id': 'qid'}).sort_values('qid')

    def fit(self, x: pd.DataFrame, y: pd.DataFrame = None, data: Data = None):
        XGBRanker.fit(self, X=self._prepare_df(x, data), y=self._prepare_df(y, data))

    def rerank(self, candidates: pd.DataFrame, data: Data = None) -> typing.Union[pd.Series, pd.DataFrame]:
        return XGBRanker.predict(self._prepare_df(candidates, data))
