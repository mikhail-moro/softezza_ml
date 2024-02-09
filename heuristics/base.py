import abc
import typing
import pandas as pd


class Heuristic(abc.ABC):
    fitted: bool = False
    name: str = '__base__'

    @abc.abstractmethod
    def fit(self, interactions: pd.DataFrame, **kwargs) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def predict_user(self, user_id: typing.Any, item_idx: typing.Iterable, **kwargs) -> typing.Union[pd.Series, pd.DataFrame]:
        raise NotImplementedError()
    
    @abc.abstractmethod
    def predict(self, user_idx: typing.Iterable, item_idx: typing.Iterable, **kwargs) -> typing.Union[pd.Series, pd.DataFrame]:
        raise NotImplementedError()
