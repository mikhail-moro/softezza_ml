from .base import Heuristic
from .cosine import Cosine


import pandas as pd
import typing


class HeuristicsWrapper:
    _heuristic_name2data: typing.Dict[str, typing.Tuple[Heuristic, float]]
    
    def __init__(self, **heuristic_weight: typing.Tuple[Heuristic, float]):
        assert len(heuristic_weight) > 0

        for h in heuristic_weight:
            assert heuristic_weight[h][0].fitted, 'Fit model before make predictions'

        self._heuristic_name2data = heuristic_weight

    def rerank(self, k:int,  reco: pd.DataFrame):
        reco['norm_score'] = self._norm_coll(reco['score'])

        for hueristic_name, hueristic_data in self._heuristic_name2data.items():
            reco[f"{hueristic_name}_norm_score"] = self._norm_coll(hueristic_data[0].predict(reco['user_id'], reco['item_id'])) * hueristic_data[1]

        norm_score_columns = [c for c in reco.columns if c.endswith('norm_score')]
        reco['norm_score'] = reco[norm_score_columns].sum(axis=1)
        reco = reco.sort_values('norm_score', ascending=False)

        reco['rank'] = reco.groupby('user_id', sort=False).cumcount() + 1
        reco = reco.drop(columns=norm_score_columns)

        return reco[reco['rank'] <= k]

    def _norm_coll(self, coll: pd.Series):
        return (coll-coll.min())/(coll.max()-coll.min())
    
    def __repr__(self) -> str:
        return self._heuristic_name2data.__repr__()