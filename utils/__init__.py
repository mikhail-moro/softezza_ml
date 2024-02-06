from .inners._sklearn import (
    SklearnEstimatorLightFM,
    dataframe2sparse_matrix,
    sparse_matrix2dataframe,
    build_estimator_params,
    build_scorer
)

from .inners._data import (
    Data,
    DataConfig,
    load_data,
    NoFilter,
    MinNumInteractionsFilter,
    OnlyLastInteractionsFilter,
    NoSplit,
    RandomSplit,
    TimeSortSplit,
    Experiment
)

from .inners._rectools import compute_rectools_metrics

from .utils import (
    genres_report,
    users_report,
    get_db_engine,
    get_users_for_test,
    light_fm_predict_user,
    xgboost_predict_user,
    PopIntersect
)


LightFM_Config = DataConfig(
    experiment=Experiment.LIGHT_FM,
    split_strategy=TimeSortSplit('all', .6, .2, .2),
    filter_strategy=[
        MinNumInteractionsFilter(20, 500),
        OnlyLastInteractionsFilter('user_id', 20)
    ],
    concat_stages=True,
    use_popular_penalty=False
)

XGBoost_Config = DataConfig(
    experiment=Experiment.XGBOOST,
    split_strategy=TimeSortSplit('all', .6, .2, .2),
    filter_strategy=[
        MinNumInteractionsFilter(20, 500),
        OnlyLastInteractionsFilter('user_id', 20)
    ],
    concat_stages=False,
    use_popular_penalty=False
)