import pandas as pd
import polars as pl
import numpy as np
import xgboost as xgb

import pymysql
import lightfm.data
import dataclasses
import os

from .inners._data import Data
from .inners._sklearn import SklearnEstimatorLightFM


ENDPOINT = "apollo-api-staging-f82be878-d243-4113-8052-ef36565618e0.cpljy7lbflfq.eu-west-1.rds.amazonaws.com"
PORT = 3306
USER = "admin"
PASSWORD = 'zsfZMSpS0SGz8gp203QJ4r3bqpVNxwmG'
DBNAME = "vapor"

metadata_columns = ['title', 'year', 'rank', 'mppa', 'genres', 'company', 'director', 'writer', 'cast']
conn: pymysql.Connection


def get_db_engine():
    global conn
    if not conn:
        conn = pymysql.connect(host=ENDPOINT, user=USER, passwd=PASSWORD, port=PORT, database=DBNAME)
    return conn


def light_fm_predict_user(
        model: lightfm.LightFM,
        data: Data,
        lightfm_data: Data.LightFM_Dataset,
        user: int,
        n: int = 10
):
    int_user = lightfm_data.mapping.ext_uid2int_uid[user]
    items = data.test_interactions['item_id'].drop_duplicates().apply(
        lightfm_data.mapping.ext_iid2int_iid.__getitem__).values
    users = np.repeat(int_user, len(items))

    if isinstance(model, lightfm.LightFM):
        score = model.predict(
            users,
            items,
            item_features=lightfm_data.item_features,
            user_features=lightfm_data.user_features,
            num_threads=12
        )
    elif isinstance(model, SklearnEstimatorLightFM):
        score = model._predict(
            users,
            items,
            item_features=lightfm_data.item_features,
            user_features=lightfm_data.user_features,
            num_threads=12
        )
    else:
        raise ValueError()

    return [lightfm_data.mapping.int_iid2ext_iid[i] for i in
            sorted(items, key=dict(zip(items, score)).__getitem__, reverse=True)[:n]]


def xgboost_predict_user(
        fs_model: lightfm.LightFM,
        ss_model: xgb.XGBRanker,
        light_fm_data: 'Data.LightFM_Dataset',
        data: 'Data',
        user: int,
        n: int = 10
):
    items = data.test_interactions['item_id'].drop_duplicates().values
    users = np.repeat(user, len(items))
    score = fs_model.predict(
        [light_fm_data.mapping.ext_uid2int_uid[u] for u in users],
        [light_fm_data.mapping.ext_iid2int_iid[i] for i in items],
        num_threads=12
    )

    df = pd.DataFrame({'user_id': users, 'item_id': items, 'score': score})
    df = data.set_xgboost_features(df).rename(columns={'user_id': 'qid'}).drop(columns=['item_id'])[
        ss_model.feature_names_in_]
    ss_score = ss_model.predict(df)

    return sorted(items, key=dict(zip(items, ss_score)).__getitem__, reverse=True)[:n]


def genres_report(preds_for_genre_lambda):
    item_id2title = pd.read_csv('../static_mappers/item_id2title.csv').set_index('item_id', drop=True)['title']
    meta_data = pd.read_csv('../static_mappers/item_id2meta.csv')
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

    report.to_csv('data/report.csv', index=False)

    return report


@dataclasses.dataclass
class _CachedUsers:
    users_idx: list
    users_histories: dict


def get_users_for_test(
        interactions_df: pd.DataFrame,
        n_users: int = 10,
        min_n_interactions: int = 10,
        max_n_interactions: int = 50,
        sort_histories: bool = True,
        top_n_hist: int = 10
) -> _CachedUsers:
    if 'timestamp' not in interactions_df.columns:
        sort_histories = False

    counts = interactions_df['user_id'].value_counts()
    users = counts[(counts.values >= min_n_interactions) & (counts.values <= max_n_interactions)].index
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


def users_report(model: lightfm.LightFM, users: _CachedUsers, n_items: int, light_fm_data: 'Data.LightFM_Dataset',
                 _data: 'Data', postfix: str = '', _dir: str = ''):
    item_id2title = pd.read_csv('../static_mappers/item_id2title.csv').set_index('item_id', drop=True)['title']
    meta_data = pd.read_csv('../static_mappers/item_id2meta.csv')

    report = pd.DataFrame(
        columns=['user_id', 'hist_item_id', 'hist_title', 'hist_genres', 'pred_item_id', 'pred_title', 'pred_genres'])

    for u in users.users_idx:
        data = pd.DataFrame()
        data['user_id'] = [u for _ in range(n_items)]
        data.loc[data.index > 0, 'user_id'] = ''

        data['hist_item_id'] = users.users_histories[u]
        data['hist_title'] = data['hist_item_id'].apply(lambda x: item_id2title.get(x, 'None'))
        data['hist_genres'] = data['hist_item_id'].apply(
            lambda x: meta_data[meta_data['item_id'] == x]['genres'].values[0])

        data['pred_item_id'] = light_fm_predict_user(model, _data, light_fm_data, u, n_items)
        data['pred_title'] = data['pred_item_id'].apply(lambda x: item_id2title.get(x, 'None'))
        data['pred_genres'] = data['pred_item_id'].apply(
            lambda x: meta_data[meta_data['item_id'] == x]['genres'].values[0])

        report = pd.concat([report, data], ignore_index=True)

    _path = f'report{postfix}.csv' if _dir == '' else os.path.join(_dir, f'report{postfix}.csv')
    report.to_csv(_path, index=False)

    return report


class PopIntersect:
    def _get_pops_from_df(self, df: pd.DataFrame):
        counts = df['item_id'].value_counts()
        return counts.index[:self.k]

    def __init__(self, k: int):
        """
        Parameters
        ----------
            k :
                num of top popular and top predicted items between which will be computed intersection 
        """
        self.k = k

    def calc(self, reco: pd.DataFrame, train_data: pd.DataFrame):
        pop_items = self._get_pops_from_df(train_data)
        user_groups = reco.groupby('user_id')
        user_groups = user_groups.apply(lambda g: g[g['rank'] <= self.k])
        user_pop_inters = user_groups.apply(lambda g: sum([i in g['item_id'] for i in pop_items]) / len(g))
        return user_pop_inters.mean()
