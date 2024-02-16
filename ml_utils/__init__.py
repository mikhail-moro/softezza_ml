from .data import (
    Data,
    DataConfig,
    
    NoFilter,
    MinNumInteractionsFilter,
    OnlyLastInteractionsFilter,
    
    NoSplit,
    RandomSplit,
    TimeSortSplit,
    
    NoWeight,
    NumViewsBasedWeight,
    ViewTimeBasedWeight,
    ViewRatioBasedWeight,

    FeaturesConfig,
    
    load_data
)

from rerank import (
    Reranker,

    Ranker,
    XGBoostRanker,
    Heuristic,

    Cosine,

    ScalarScaleDistance
)

from .utils import (    
    PopularIntersect,
    RecallNoPop,

    genres_report,
    users_report,
    get_db_engine,
    get_users_for_test,
    light_fm_predict_user,
    xgboost_predict_user
)


LightFM_Config = DataConfig(
    split_strategy=TimeSortSplit(num_interactions='all', splits=(.8, .2)),
    filter_strategy=[
        MinNumInteractionsFilter(20, 500),
        OnlyLastInteractionsFilter('user_id', 20)
    ],
    features_config=FeaturesConfig(use_labels=True)
)

RecTools_Config = DataConfig(
    split_strategy=TimeSortSplit(num_interactions='all', splits=(.8, .2)),
    filter_strategy=[
        MinNumInteractionsFilter(20, 500),
        OnlyLastInteractionsFilter('user_id', 20)
    ],
    features_config=FeaturesConfig(use_labels=False)
)

XGBoost_Config = DataConfig(
    split_strategy=TimeSortSplit(num_interactions='all', splits=(.6, .2, .2)),
    filter_strategy=[
        MinNumInteractionsFilter(20, 500),
        OnlyLastInteractionsFilter('user_id', 20)
    ],
    features_config=FeaturesConfig(use_labels=False)
)
