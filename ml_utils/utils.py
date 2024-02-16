import pandas as pd
import numpy as np
import xgboost as xgb

import os
import pymysql
import typing
import lightfm
import lightfm.data
import dataclasses

from .data import Data
from rectools.metrics import Recall
from rectools.columns import Columns
from rectools.models.base import ModelBase
from rectools.metrics.base import MetricAtK

if typing.TYPE_CHECKING:
    from rectools.dataset import Dataset


conn: typing.Union[pymysql.Connection, None] = None

def get_db_engine():
    global conn

    if conn is None:
        conn = pymysql.connect(
            host=os.environ['DB_ENDPOINT'],
            user=os.environ['DB_USER'],
            passwd=os.environ['DB_PASSWORD'],
            port=int(os.environ['DB_PORT']),
            database=os.environ['DB_NAME']
        )
    
    return conn


def rectools_predict_user(
    dataset: 'Dataset',
    model: ModelBase,
    user: int,
    n: int = 10
):
    return model.recommend((user,), dataset, k=n, filter_viewed=False, add_rank_col=True)[Columns.Item].values


def light_fm_predict_user(
    dataset: Data.LightFM_Dataset,
    model: lightfm.LightFM,
    user: int,
    n: int = 10
):
    int_user = dataset.mapping.ext_uid2int_uid[user]
    pred_method = model.predict if isinstance(model, lightfm.LightFM) else model._predict  

    items = dataset.test_interactions['item_id'].drop_duplicates().apply(dataset.mapping.ext_iid2int_iid.__getitem__).values
    users = np.repeat(int_user, len(items))

    score = pred_method(
        users,
        items,
        item_features=dataset.item_features,
        user_features=dataset.user_features,
        num_threads=12
    )

    out_items = sorted(items, key=dict(zip(items, score)).__getitem__, reverse=True)[:n]

    return [dataset.mapping.int_iid2ext_iid[i] for i in out_items]


def xgboost_predict_user(
    data: Data,
    mapper: Data.LightFM_DatasetMapping,
    first_stage_model: lightfm.LightFM,
    second_stage_model: xgb.XGBRanker,
    user: int,
    n: int = 10
):
    pred_method = first_stage_model.predict if isinstance(first_stage_model, lightfm.LightFM) else first_stage_model._predict
    
    items = data.test_interactions['item_id'].drop_duplicates().values
    users = np.repeat(user, len(items))
    score = pred_method(
        [mapper.ext_uid2int_uid[u] for u in users],
        [mapper.ext_iid2int_iid[i] for i in items],
        num_threads=12
    )

    df = pd.DataFrame({'user_id': users, 'item_id': items, 'score': score})
    df = data.set_xgboost_features(df).rename(columns={'user_id': 'qid'}).drop(columns=['item_id'])[
        second_stage_model.feature_names_in_
    ]
    second_stage_score = second_stage_model.predict(df)

    return sorted(items, key=dict(zip(items, second_stage_score)).__getitem__, reverse=True)[:n]


def genres_report(preds_for_genre_lambda):
    item_id2title = pd.read_csv(os.path.join(os.environ['DIR'], '/static_mappers/item_id2title.csv')).set_index('item_id', drop=True)['title']
    meta_data = pd.read_csv(os.path.join(os.environ['DIR'], '/static_mappers/item_id2meta.csv'))
    report = pd.DataFrame(columns=['genre', 'user_id', 'profile_number', 'n', 'title', 'genres'])

    for g in ['Drama', 'Comedy', 'Thriller', 'Romance', 'Action', 'Adventure', 'Fantasy', 'Sci-Fi', 'Horror', 'Family',
              'Animation']:
        data = pd.DataFrame()
        data['genre'] = g
        data['item_id'] = preds_for_genre_lambda(g)
        data.loc[data.index > 0, 'genre'] = ''
        data.loc[data.index > 0, 'user_id'] = ''
        data.loc[data.index > 0, 'profile_number'] = ''
        data['n'] = range(1, 11, 1)
        data['title'] = data['item_id'].apply(lambda x: item_id2title.get(x, 'None'))
        data['genres'] = data['item_id'].apply(lambda x: meta_data[meta_data['item_id'] == x]['genres'].values[0])

        report = pd.concat([report, data], ignore_index=True)

    report.to_csv(os.path.join(os.environ['DIR'], 'reports/report.csv'), index=False)

    return report


@dataclasses.dataclass
class _CachedUsers:
    users_idx: list
    users_histories: dict


def get_users_for_test(
    interactions_df: pd.DataFrame,
    n_users: int = 10,
    min_n_interactions: int = None,
    max_n_interactions: int = None,
    sort_histories: bool = True,
    top_n_hist: int = 10
) -> _CachedUsers:
    if 'timestamp' not in interactions_df.columns:
        sort_histories = False

    counts = interactions_df['user_id'].value_counts()

    if min_n_interactions:
        users = counts[counts.values >= min_n_interactions].index
    
    if max_n_interactions:
        users = counts[counts.values <= max_n_interactions].index
    
    users = np.random.choice(users, n_users)
    users_data = interactions_df.groupby('user_id')

    _users = []
    _hists = {}

    for u in users:
        u_data = users_data.get_group(u)

        if sort_histories:
            u_data = u_data.sort_values('timestamp', ascending=False)

        _users.append(u)
        _hists[u] = u_data['item_id'].head(top_n_hist).values

    return _CachedUsers(_users, _hists)


def users_report(
    dataset: typing.Union[Data.LightFM_Dataset, 'Dataset'],
    model: typing.Union[lightfm.LightFM, 'ModelBase'],
    users: _CachedUsers,
    n_items: int = 10,
    postfix: str = None, 
    report_dir: str = None
):
    if isinstance(model, lightfm.LightFM):
        pred_method = light_fm_predict_user
    elif issubclass(model.__class__, ModelBase):
        pred_method = rectools_predict_user
    else:
        raise ValueError()

    def _pad_items(_items):
        if len(_items) < n_items:
            _items = list(_items) + ['' for _ in range(n_items - len(_items))]
        return _items

    item_id2title = pd.read_csv(os.path.join(os.environ['DIR'], 'static_mappers/item_id2title.csv')).set_index('item_id', drop=True)['title']
    item_id2genres = pd.read_csv(os.path.join(os.environ['DIR'], 'static_mappers/item_id2meta.csv'))[['item_id', 'genres']].set_index('item_id', drop=True)['genres']
    report = pd.DataFrame(columns=['user_id', 'hist_item_id', 'hist_title', 'hist_genres', 'pred_item_id', 'pred_title', 'pred_genres'])

    for u in users.users_idx:
        user_data = pd.DataFrame()
        user_data['user_id'] = [u for _ in range(n_items)]
        user_data.loc[user_data.index > 0, 'user_id'] = ''

        user_data['hist_item_id'] = _pad_items(users.users_histories[u])
        user_data['hist_title'] = user_data['hist_item_id'].apply(lambda x: item_id2title.get(x, ''))
        user_data['hist_genres'] = user_data['hist_item_id'].apply(lambda x: item_id2genres.get(x, ''))

        user_data['pred_item_id'] = _pad_items(pred_method(dataset=dataset, model=model, user=u, n=n_items))
        user_data['pred_title'] = user_data['pred_item_id'].apply(lambda x: item_id2title.get(x, ''))
        user_data['pred_genres'] = user_data['pred_item_id'].apply(lambda x: item_id2genres.get(x, ''))

        report = pd.concat([report, user_data], ignore_index=True)

    p = f'report_{postfix}.csv' if postfix else f'report.csv'
    p = os.path.join(report_dir, p) if report_dir else p
    report.to_csv(p, index=False)

    return report


class PopularIntersect(MetricAtK):
    def calc(self, reco: pd.DataFrame, prev_interactions: pd.DataFrame):
        pop_recos = set(prev_interactions['item_id'].value_counts().head(self.k).index)
        user_values = reco.groupby('user_id', group_keys=False).apply(lambda g: 1 - (len(set(g['item_id'].values) - pop_recos) / self.k))
        
        return user_values.mean()


class RecallNoPop(Recall):
    def calc(self, reco: pd.DataFrame, interactions: pd.DataFrame, prev_interactions: pd.DataFrame):
        pop_recos = set(prev_interactions['item_id'].value_counts().head(self.k).index)

        no_pop_test_interactions = interactions[~interactions['item_id'].isin(pop_recos)]
        no_pop_reco = reco.merge(
            no_pop_test_interactions[["user_id"]].drop_duplicates(),
            on="user_id",
            how="inner",
        )

        return super().calc(
            reco=no_pop_reco,
            interactions=no_pop_test_interactions,
        )
