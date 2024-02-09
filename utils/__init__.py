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
    
    PopularIntersect,
    RecallNoPop
)



import typing
import sklearn.model_selection
import rectools.metrics


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


class AnOptimizedParamGridSearch:
    k: int
    grid: dict
    verbose: bool
    num_threads: int

    def __init__(self, grid: dict, num_threads: int = 12, k: int = 10, verbose: bool = True):
        self.k = k
        self.grid = grid
        self.verbose = verbose
        self.num_threads = num_threads

    def fit(self, X, y, data: Data):
        results = []

        for label, params in self.grid.items():
            
            for p in sklearn.model_selection.ParameterGrid(params['grid']):
            
                if 'wrapper' in params.keys():
                    model = params['wrapper'](params['model'](num_threads=self.num_threads, **p))
                else:
                    model = params['model'](num_threads=self.num_threads, **p)

                model.fit(X)
        
                recos = model.recommend(
                    k=self.k,
                    users=data.all_users,
                    dataset=X,
                    filter_viewed=True,
                    add_rank_col=True,
                )

                metrics = rectools.metrics.calc_metrics(
                    {
                        'MAP@10': rectools.metrics.MAP(self.k),
                        'Recall@10': rectools.metrics.Recall(self.k),
                        'Siren@10': rectools.metrics.Serendipity(self.k),
                        'MIUF@10': rectools.metrics.MeanInvUserFreq(self.k)
                    },
                    reco=recos,
                    interactions=data.test_interactions,
                    prev_interactions=data.train_interactions,
                    catalog=data.all_items
                )
                metrics['PopInt@10'] = PopularIntersect(self.k).calc(reco=recos, prev_interactions=data.train_interactions)
                metrics['model'] = label
                metrics = {**metrics, **p}
                results.append(metrics)
                
                if self.verbose:
                    print(metrics)