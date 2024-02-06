import pandas as pd

from sklearn.base import BaseEstimator
from rectools import metrics
from scipy.sparse import coo_matrix, spmatrix as _sparse_matrix_base

from ._rectools import compute_rectools_metrics
from ._data import Data

import typing
import lightfm.data

if typing.TYPE_CHECKING:
    from rectools.metrics.base import MetricAtK


def build_estimator_params(data: Data, dataset: Data.LightFM_Dataset) -> typing.Dict[str, typing.Any]:
    return {
        'lightfm_dataset': dataset.lightfm_dataset,
        'user_features': dataset.user_features,
        'item_features': dataset.item_features,
        'weights': dataset.weights,
        'train_interactions': dataset.train_interactions,
        'test_interactions': dataset.test_interactions,
        'train_interactions_df': data.train_interactions,
        'test_interactions_df': data.test_interactions,
        'catalog': data.all_items,
        'mapping': dataset.mapping
    }


def build_scorer(
    data: Data,
    dataset: Data.LightFM_Dataset,
    eval_metrics: typing.Iterable[typing.Type['MetricAtK']]
) -> typing.Callable[['SklearnEstimatorLightFM', pd.DataFrame, typing.Any], typing.Dict[str, float]]:

    def _(estimator: SklearnEstimatorLightFM, X: pd.DataFrame, y=None):
        return estimator.score(X, y, eval_metrics=eval_metrics, **build_estimator_params(data, dataset))

    return _


class SklearnEstimatorLightFM(lightfm.LightFM, BaseEstimator):
    _fit = lightfm.LightFM.fit
    _predict = lightfm.LightFM.predict

    num_epochs: int

    def __init__(
            self,
            no_components=10,
            learning_schedule="adagrad",
            loss="warp",
            learning_rate=0.05,
            item_alpha=0.0,
            user_alpha=0.0,
            max_sampled=10,
            num_epochs=10,
            epsilon=0.000001,
            rho=0.95,
            k=5,
            n=10,
            random_state=None,
            **kwargs,
    ):
        BaseEstimator.__init__(self)
        lightfm.LightFM.__init__(
            self,
            no_components=no_components,
            k=k,
            n=n,
            learning_schedule=learning_schedule,
            loss=loss,
            learning_rate=learning_rate,
            rho=rho,
            epsilon=epsilon,
            item_alpha=item_alpha,
            user_alpha=user_alpha,
            max_sampled=max_sampled,
            random_state=random_state
        )

        self.num_epochs = num_epochs

    def fit(self, X: typing.Union[pd.DataFrame, coo_matrix], y=None, **params):
        if not isinstance(X, coo_matrix):
            X, w = dataframe2sparse_matrix(params['lightfm_dataset'], X)
        else:
            w = params['weights']

        lightfm.LightFM.fit(
            self,
            X,
            user_features=params['user_features'].copy() if params['user_features'] else None,
            item_features=params['item_features'].copy() if params['item_features'] else None,
            sample_weight=w,
            epochs=self.num_epochs,
            num_threads=12
        )

    def score(
        self,
        X,
        y=None,
        eval_metrics: typing.Iterable[typing.Type['MetricAtK']] = None,
        **params
    ) -> typing.Dict[str, float]:
        if not issubclass(X, _sparse_matrix_base):
            X, _ = dataframe2sparse_matrix(params['lightfm_dataset'], X)

        if not eval_metrics:
            eval_metrics = (metrics.MAP, metrics.Recall, metrics.MeanInvUserFreq)

        ranks = lightfm.LightFM.predict_rank(
            self,
            test_interactions=params['test_interactions'].copy(),
            train_interactions=X,
            user_features=params['user_features'].copy(),
            item_features=params['item_features'].copy(),
            num_threads=12
        )

        return compute_rectools_metrics(
            ranks,
            params['mapping'].int_uid2ext_uid.copy(),
            params['mapping'].int_iid2ext_iid.copy(),
            eval_metrics=eval_metrics,
            interactions=params['test_interactions_df'].copy(),
            prev_interactions=params['train_interactions_df'].copy(),
            catalog=params['catalog'].copy()
        )

    def get_params(self, deep=True) -> typing.Dict[str, typing.Any]:
        params = super().get_params(deep)
        params['num_epochs'] = self.num_epochs

        return params


def sparse_matrix2dataframe(
    lightfm_dataset: Data.LightFM_Dataset,
    sparse_matrix: _sparse_matrix_base
) -> pd.DataFrame:
    sparse_matrix = sparse_matrix.tocsr()
    csr = sparse_matrix.tocsr()
    nnz = csr.nonzero()

    reco = pd.DataFrame({
        'user_id': nnz[0],
        'item_id': nnz[1],
        'rank': map(csr.__getitem__, zip(*nnz))
    })

    reco['user_id'] = reco['user_id'].apply(lightfm_dataset.mapping.int_uid2ext_uid.__getitem__)
    reco['item_id'] = reco['item_id'].apply(lightfm_dataset.mapping.int_iid2ext_iid.__getitem__)
    reco['rank'] = reco['rank'].astype(int)

    return reco


def dataframe2sparse_matrix(
    lightfm_dataset: Data.LightFM_Dataset,
    dataframe: pd.DataFrame
) -> typing.Tuple[coo_matrix, coo_matrix]:
    if 'weight' in dataframe.columns:
        return lightfm_dataset.lightfm_dataset.build_interactions(tuple(dataframe[['user_id', 'item_id', 'weight']].itertuples(False, None)))
    else:
        return lightfm_dataset.lightfm_dataset.build_interactions(tuple(dataframe[['user_id', 'item_id']].itertuples(False, None)))
