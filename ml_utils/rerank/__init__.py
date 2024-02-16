from base import (
    Ranker,
    Heuristic,
    XGBoostRanker
)

from cosine import (
    Cosine,
    # TimeDelayedWeightedCosine
)

from simple import (
    ScalarScaleDistance
)

from ..data import Data
import pandas as pd
import typing


class Reranker:
    first_stage_score_weight: float
    _ranker_name2data: typing.Dict[str, typing.Tuple[Heuristic, float]]
    
    def __init__(self, first_stage_weight: float = 1., **heuristics: typing.Tuple[Heuristic, float]):
        assert len(heuristics) > 0

        self.first_stage_weight = first_stage_weight
        self._ranker_name2data = heuristics

    def rerank(
        self,
        reco: pd.DataFrame,
        data: Data,
        k: int = 10,
        keep_first_stage_score: bool = False,
        keep_first_stage_rank: bool = False,
        keep_rankers_scores: bool = False,
        keep_rankers_ranks: bool = False
    ):
        
        for ranker_name, (ranker, weight) in self._ranker_name2data.items():
            ranker_df = reco[['user_id', 'item_id']].copy()
            ranker_df[f"{ranker_name}_score"] = ranker.rerank(ranker_df['user_id'], data)
            ranker_df = ranker_df.sort_values(f"{ranker_name}_score", ascending=False)
            ranker_df[f"{ranker_name}_rank"] = ranker_df.groupby('user_id').cumcount() + 1

            if not keep_rankers_scores:
                ranker_df = ranker_df.drop(columns=f"{ranker_name}_score")

            reco = pd.merge(reco, ranker_df, on=['user_id', 'item_id'], how='left')
            reco[f"{ranker_name}_rank"] = reco[f"{ranker_name}_rank"].fillna(-999)
            reco[f"weighted_{ranker_name}_rank"] = reco[f"{ranker_name}_rank"] * weight

            if not keep_rankers_ranks:
                reco = reco.drop(columns=f"{ranker_name}_rank")

        reco['weighted_rank'] = reco['rank'] * self.first_stage_weight
        rank_columns = ['weighted_rank'] + [f"weighted_{hueristic_name}_rank" for hueristic_name in self._ranker_name2data.keys()]
        
        if keep_first_stage_score:
            reco = reco.rename(columns={'score': 'first_stage_rank'})
        else:
            reco = reco.drop(columns='score')

        if keep_first_stage_rank:
            reco = reco.rename(columns={'rank': 'first_stage_rank'})
        
        reco['mid_rank'] = reco[rank_columns].sum(axis=1)
        reco = reco.sort_values('mid_rank').groupby('user_id').head(k).drop(columns='mid_rank')
        reco["rank"] = reco.groupby("user_id").cumcount() + 1
        
        return reco

    def __repr__(self) -> str:
        return self._ranker_name2data.__repr__()
