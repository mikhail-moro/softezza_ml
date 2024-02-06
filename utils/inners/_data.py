import pandas as pd
import polars as pl
import numpy as np

from collections.abc import Iterable

import typing
import lightfm.data
import collections
import dataclasses
import enum
import abc


class Experiment(enum.Enum):
    LIGHT_FM = 0
    XGBOOST = 1


class Data:
    LightFM_Dataset = collections.namedtuple(
        'LightFM_Dataset',
        ['train_interactions', 'test_interactions', 'user_features', 'item_features', 'weights', 'lightfm_dataset',
         'mapping']
    )
    LightFM_DatasetMapping = collections.namedtuple(
        'LightFM_DatasetMapping',
        ['ext_uid2int_uid', 'int_uid2ext_uid', 'ext_iid2int_iid', 'int_iid2ext_iid']
    )

    train_interactions: pd.DataFrame or tuple
    test_interactions: pd.DataFrame
    user_features: pd.DataFrame
    item_features: pd.DataFrame

    all_users: np.ndarray
    all_items: np.ndarray

    def __init__(
            self,
            train_interactions: pd.DataFrame or tuple,
            test_interactions: pd.DataFrame,
            user_features: pd.DataFrame,
            item_features: pd.DataFrame
    ):
        self.train_interactions = train_interactions
        self.test_interactions = test_interactions
        self.user_features = user_features
        self.item_features = item_features

        if isinstance(train_interactions, tuple):
            self.all_users = np.unique(np.concatenate([
                train_interactions[0]['user_id'].drop_duplicates().values,
                train_interactions[1]['user_id'].drop_duplicates().values,
                test_interactions['user_id'].drop_duplicates().values
            ]))
            self.all_items = np.unique(np.concatenate([
                train_interactions[0]['item_id'].drop_duplicates().values,
                train_interactions[1]['item_id'].drop_duplicates().values,
                test_interactions['item_id'].drop_duplicates().values
            ]))
        else:
            self.all_users = np.unique(np.concatenate([
                train_interactions['user_id'].drop_duplicates().values,
                test_interactions['user_id'].drop_duplicates().values
            ]))
            self.all_items = np.unique(np.concatenate([
                train_interactions['item_id'].drop_duplicates().values,
                test_interactions['item_id'].drop_duplicates().values
            ]))

    def get_lightfm_features(
            self,
            drop_features: list,
            list_values_columns: list,
            scalar_values_columns: list
    ):
        def _parse_feature_names(column: str, _data: pd.DataFrame):
            if column in drop_features:
                return []
            if column in scalar_values_columns:
                return [column]
            if column in list_values_columns:
                return [f"{column}_{_}" for _ in np.unique(np.concatenate(_data[column].values))]

            return [f"{column}_{_}" for _ in np.unique(_data[column].values)]

        ufeatures = self.user_features.copy()
        ifeatures = self.item_features.copy()

        user_features = ufeatures[ufeatures['user_id'].isin(self.all_users)]
        user_labels = np.unique(np.concatenate(
            [_parse_feature_names(c, _data=user_features) for c in user_features.columns if c != 'user_id']))
        user_features = user_features.set_index('user_id')

        item_features = ifeatures[ifeatures['item_id'].isin(self.all_items)]
        item_labels = np.unique(np.concatenate(
            [_parse_feature_names(c, _data=item_features) for c in item_features.columns if c != 'item_id']))
        item_features = item_features.set_index('item_id')

        def _parse_feature_row(row: pd.Series):
            out = {}

            for i in row.index:
                if i in drop_features:
                    continue
                if i in list_values_columns:
                    for _ in row[i]: out[f"{i}_{_}"] = 1
                    continue
                if i in scalar_values_columns:
                    out[i] = row[i]
                    continue

                if isinstance(row[i], float):
                    out[f"{i}_{int(row[i])}"] = 1
                else:
                    out[f"{i}_{row[i]}"] = 1

            return out

        return collections.namedtuple('LightFM_Features',
                                      ['user_features', 'item_features', 'user_labels', 'item_labels'])(
            user_features=tuple([(i, _parse_feature_row(r)) for i, r in user_features.iterrows()]),
            item_features=tuple([(i, _parse_feature_row(r)) for i, r in item_features.iterrows()]),
            user_labels=user_labels,
            item_labels=item_labels
        )

    def set_xgboost_features(
            self,
            interactions: pd.DataFrame
    ):
        interactions = pd.merge(interactions, self.user_features, on='user_id', how='inner')
        interactions = pd.merge(interactions, self.item_features, on='item_id', how='inner')

        return interactions

    def get_lightfm_dataset(
            self,
            with_features: bool = True,
            drop_features: list = None,
            list_values_columns: list = None,
            scalar_values_columns: list = None
    ):
        if with_features:
            features = self.get_lightfm_features(drop_features, list_values_columns, scalar_values_columns)
        else:
            features = None

        lightfm_dataset = lightfm.data.Dataset()
        lightfm_dataset.fit(
            self.all_users,
            self.all_items,
            user_features=features.user_labels if features else None,
            item_features=features.item_labels if features else None,
        )

        if isinstance(self.train_interactions, tuple):
            train, weights = lightfm_dataset.build_interactions(
                tuple(self.train_interactions[0][['user_id', 'item_id', 'weight']].itertuples(False, None)))
            test, _ = lightfm_dataset.build_interactions(
                tuple(self.test_interactions[['user_id', 'item_id', 'weight']].itertuples(False, None)))
        else:
            train, weights = lightfm_dataset.build_interactions(
                tuple(self.train_interactions[['user_id', 'item_id', 'weight']].itertuples(False, None)))
            test, _ = lightfm_dataset.build_interactions(
                tuple(self.test_interactions[['user_id', 'item_id', 'weight']].itertuples(False, None)))

        if with_features:
            user_features = lightfm_dataset.build_user_features(features.user_features)
            item_features = lightfm_dataset.build_item_features(features.item_features)
        else:
            user_features = None
            item_features = None

        ext_uid2int_uid, _, ext_iid2int_iid, _ = lightfm_dataset.mapping()
        int_uid2ext_uid = dict(zip(ext_uid2int_uid.values(), ext_uid2int_uid.keys()))
        int_iid2ext_iid = dict(zip(ext_iid2int_iid.values(), ext_iid2int_iid.keys()))

        return self.LightFM_Dataset(
            train_interactions=train,
            test_interactions=test,
            user_features=user_features if user_features else None,
            item_features=item_features if item_features else None,
            weights=weights,
            lightfm_dataset=lightfm_dataset,
            mapping=self.LightFM_DatasetMapping(
                ext_uid2int_uid=ext_uid2int_uid,
                int_uid2ext_uid=int_uid2ext_uid,
                ext_iid2int_iid=ext_iid2int_iid,
                int_iid2ext_iid=int_iid2ext_iid
            )
        )

    def __del__(self):
        del self.train_interactions
        del self.test_interactions
        del self.user_features
        del self.item_features
        del self.all_users
        del self.all_items


class FilterStrategy(abc.ABC):
    @abc.abstractmethod
    def filter(self, data: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError()


@dataclasses.dataclass
class MinNumInteractionsFilter(FilterStrategy):
    min_user_ints: int
    min_item_ints: int

    def filter(self, data: pl.DataFrame) -> pl.DataFrame:
        user_id2num_inters = data['user_id'].value_counts()
        user_id2num_inters = dict(zip(user_id2num_inters['user_id'], user_id2num_inters['count']))
        item_id2num_inters = data['item_id'].value_counts()
        item_id2num_inters = dict(zip(item_id2num_inters['item_id'], item_id2num_inters['count']))

        data = data.filter(data['user_id'].map_elements(lambda x: user_id2num_inters[x] > self.min_user_ints))
        data = data.filter(data['item_id'].map_elements(lambda x: item_id2num_inters[x] > self.min_item_ints))

        return data


@dataclasses.dataclass
class OnlyLastInteractionsFilter(FilterStrategy):
    filter_column: str
    n_last: int

    def filter(self, data: pl.DataFrame) -> pl.DataFrame:
        data = data.with_columns(index=np.arange(len(data)))
        data = data.group_by(self.filter_column)
        data = data.map_groups(lambda g: g.sort('timestamp', descending=True).head(self.n_last))
        return data


class NoFilter(FilterStrategy):
    def filter(self, data: pl.DataFrame) -> pl.DataFrame:
        return data


class SplitStrategy(abc.ABC):

    @abc.abstractmethod
    def split(self, data: pl.DataFrame) -> tuple:
        raise NotImplementedError()


@dataclasses.dataclass
class TimeSortSplit(SplitStrategy):
    num_interactions: int or str
    first_stage_train_split: float
    second_stage_train_split: float
    test_split: float

    def split(self, data: pl.DataFrame) -> tuple:
        self.num_interactions = len(data) if self.num_interactions == 'all' else self.num_interactions
        fs_interactions = self.num_interactions * (1 - self.first_stage_train_split)
        ss_interactions = self.num_interactions * (1 - self.first_stage_train_split - self.second_stage_train_split)
        data = data.sort('timestamp', descending=True)
        data = data.with_columns(index=np.arange(len(data)))

        train_1 = data.filter(
            (pl.col('index') < self.num_interactions)
            &
            (pl.col('index') > fs_interactions)
        )

        train_2 = data.filter(
            (pl.col('index') < fs_interactions)
            &
            (pl.col('index') > ss_interactions)
        )

        test = data.filter(
            (pl.col('index') < ss_interactions)
        )

        return train_1, train_2, test


@dataclasses.dataclass
class RandomSplit(SplitStrategy):
    num_interactions: int or str
    first_stage_train_split: float
    second_stage_train_split: float
    test_split: float

    def split(self, data: pl.DataFrame) -> tuple:
        self.num_interactions = len(data) if self.num_interactions == 'all' else self.num_interactions
        fs_interactions = self.num_interactions * (1 - self.first_stage_train_split)
        ss_interactions = self.num_interactions * (1 - self.first_stage_train_split - self.second_stage_train_split)
        data = data.sample(fraction=1., shuffle=True)
        data = data.with_columns(index=list(np.arange(len(data))))

        train_1 = data.filter(
            (pl.col('index') < self.num_interactions)
            &
            (pl.col('index') > fs_interactions)
        )

        train_2 = data.filter(
            (pl.col('index') < fs_interactions)
            &
            (pl.col('index') > ss_interactions)
        )

        test = data.filter(
            (pl.col('index') < ss_interactions)
        )

        return train_1, train_2, test


class NoSplit(SplitStrategy):
    def split(self, data: pl.DataFrame) -> tuple:
        return tuple([data])


@dataclasses.dataclass
class DataConfig:
    """
    Warning:
    ----------

    Length of output data will may be less than splits values because user/item filtering


    Parameters:
    ----------

    experiment :
        Must be `light_fm` or `xgboost`

    split_strategy :
        Data splitting config

    filter_strategy :
        Data filtering config

    concat_stages :
        If `True` - unless 2 data files for every period, returned data will
        be concat to one file

    use_popular_penalty :
        If `True` - combine weight matrix with popular penalty matrix
    """
    experiment: Experiment = Experiment.LIGHT_FM
    split_strategy: SplitStrategy = NoSplit
    filter_strategy: typing.Union[FilterStrategy, typing.Iterable[FilterStrategy]] = NoFilter

    concat_stages: bool = True
    use_popular_penalty: bool = False


# Default configs
LightFM_DataConfig = DataConfig(
    experiment=Experiment.LIGHT_FM,
    split_strategy=RandomSplit('all', 0.6, 0.2, 0.2),
    filter_strategy=MinNumInteractionsFilter(50, 100),
    concat_stages=True,
    use_popular_penalty=False
)
XGBoost_DataConfig = DataConfig(
    experiment=Experiment.XGBOOST,
    split_strategy=RandomSplit('all', 0.6, 0.2, 0.2),
    filter_strategy=MinNumInteractionsFilter(50, 100),
    concat_stages=True,
    use_popular_penalty=False
)


def load_data(config: DataConfig, verbose_lens: bool = True) -> Data:
    """
    Loads all data for given config settings and pack it in `Data` class
    """

    interactions = pl.read_csv(
        'data/history_with_dt.csv',
        try_parse_dates=True,
        schema={'user_id': pl.Int64, 'item_id': pl.String, 'timestamp': pl.Datetime, 'weight': pl.Float64}
    )
    interactions = interactions.drop_nulls()
    interactions = interactions.unique(subset=['user_id', 'item_id'], keep='first')

    if isinstance(config.filter_strategy, Iterable):
        for fs in config.filter_strategy:
            interactions = fs.filter(interactions)
    elif config.filter_strategy:
        interactions = config.filter_strategy.filter(interactions)

    if config.experiment == Experiment.LIGHT_FM:
        item_features = pl.read_csv(
            '/home/ml/ml_proj/features/item_features.csv',
            dtypes={
                'item_id': pl.String,
                'rank': pl.String,
                'year': pl.String,
                'mppa': pl.String,
                'genres': pl.String,
                'runtime': pl.String
            }
        )
        user_features = pl.read_csv(
            '/home/ml/ml_proj/features/user_features.csv',
            columns=['user_id', 'device', 'account_type', 'lifetime']
        )
        parse_genres = lambda genres: genres.replace('"', '').replace("'", '').replace(']', '').replace('[',
                                                                                                        '').replace(' ',
                                                                                                                    '').split(
            ',')
        item_features = item_features.with_columns(
            genres=item_features['genres'].map_elements(parse_genres)
        )
    else:
        item_features = pl.read_csv('/home/ml/ml_proj/features/item_features_bin.csv')
        user_features = pl.read_csv('/home/ml/ml_proj/features/user_features_bin.csv')

    item_features = item_features.drop_nulls()
    user_features = user_features.drop_nulls()
    interactions = interactions.filter(pl.col('user_id').is_in(set(user_features['user_id'].to_list())))
    interactions = interactions.filter(pl.col('item_id').is_in(set(item_features['item_id'].to_list())))

    train_1, train_2, test = config.split_strategy.split(interactions)
    all_users = train_1[['user_id']].unique()
    all_items = train_1[['item_id']].unique()

    train_1 = train_1.to_pandas()
    train_2 = train_2.join(all_users, how='inner', on='user_id')
    train_2 = train_2.join(all_items, how='inner', on='item_id').to_pandas()
    test = test.join(all_users, how='inner', on='user_id')
    test = test.join(all_items, how='inner', on='item_id').to_pandas()

    if config.concat_stages:
        train = pd.concat([train_1, train_2])

        item_id2num_inters = train['item_id'].value_counts()
        item_id2num_inters = dict(zip(item_id2num_inters.index, item_id2num_inters.values))

        if config.use_popular_penalty:
            max_views_q95 = train['item_id'].value_counts().quantile(0.95)
            train['penalty'] = np.clip((train['item_id'].apply(item_id2num_inters.__getitem__) / max_views_q95).values,
                                       0, 1)
            train['penalty'] = train['penalty'].apply(lambda x: 1 - x)

            train['weight'] = train['weight'] * train['penalty']
            train['weight'] = (train['weight'] - train['weight'].min()) / (
                        train['weight'].max() - train['weight'].min())

            train = train.drop(columns='penalty')

        train['weight'] = np.clip(train['weight'].values, a_min=0.01, a_max=0.99)

        if verbose_lens:
            print(
                "Data after filter:\n" +
                f"Len of train interactions with period [{train.iloc[-1]['timestamp']} / {train.iloc[0]['timestamp']}] - {len(train)}\n" +
                f"Len of test interactions with period [{test.iloc[-1]['timestamp']} / {test.iloc[0]['timestamp']}] - {len(test)}"
            )

        data = Data(
            train_interactions=train,
            test_interactions=test,
            user_features=user_features.to_pandas(),
            item_features=item_features.to_pandas()
        )
    else:
        item_id2num_inters = train_1['item_id'].value_counts()
        item_id2num_inters = dict(zip(item_id2num_inters.index, item_id2num_inters.values))

        if config.use_popular_penalty:
            max_views_q95 = train_1['item_id'].value_counts().quantile(0.95)
            pop_penalty = train_1['item_id'].apply(lambda x: 1 - np.clip(item_id2num_inters[x] / max_views_q95, 0, 1))
            train_1['weight'] = train_1['weight'] * pop_penalty
            train_1['weight'] = (train_1['weight'] - train_1['weight'].min()) / (
                        train_1['weight'].max() - train_1['weight'].min())

        train_1['weight'] = np.clip(train_1['weight'].values, a_min=0.01, a_max=0.99)
        train_2['weight'] = np.clip(train_2['weight'].values, a_min=0.01, a_max=0.99)

        if verbose_lens:
            print(
                "Data after filter:\n" +
                f"Len of first stage train interactions with period [{train_1.iloc[-1]['timestamp']} / {train_1.iloc[0]['timestamp']}] - {len(train_1)}\n" +
                f"Len of second stage train interactions with period [{train_2.iloc[-1]['timestamp']} / {train_2.iloc[0]['timestamp']}] - {len(train_2)}\n" +
                f"Len of test interactions with period [{test.iloc[-1]['timestamp']} / {test.iloc[0]['timestamp']}] - {len(test)}"
            )

        data = Data(
            train_interactions=(train_1, train_2),
            test_interactions=test,
            user_features=user_features.to_pandas(),
            item_features=item_features.to_pandas()
        )

    return data


def sampling(candidates: pd.DataFrame, train_data: pd.DataFrame, n_negatives: int = 3) -> pd.DataFrame:
    train_data['target'] = 1
    train_data['user_id'] = train_data['user_id'].astype('int32')

    train_data = pd.merge(candidates, train_data[['user_id', 'item_id', 'target']], how='left',
                          on=['user_id', 'item_id'])
    train_data['target'] = train_data['target'].fillna(0).astype('int32')
    train_data['id'] = train_data.index

    pos = train_data[train_data["target"] == 1]

    num_positives = pos.groupby("user_id")["item_id"].count().astype('int32')
    num_positives.name = "num_positives"

    neg = train_data[train_data["target"] == 0]
    num_negatives = neg.groupby("user_id")["item_id"].count().astype('int32')
    num_negatives.name = "num_negatives"

    neg_sampling = pd.DataFrame(neg.groupby("user_id")["id"].apply(list))
    neg_sampling = neg_sampling.join(
        num_positives, on="user_id", how="left"
    )
    neg_sampling = neg_sampling.join(num_negatives, on="user_id", how="left")
    neg_sampling['num_negatives'] = neg_sampling["num_negatives"].fillna(0).astype('int32')
    neg_sampling['num_positives'] = neg_sampling["num_positives"].fillna(0).astype('int32')
    neg_sampling['num_choices'] = np.clip(neg_sampling["num_positives"] * n_negatives, a_min=0, a_max=25, )

    rng = np.random.default_rng()
    neg_sampling['sampled_idx'] = neg_sampling.apply(
        lambda row: tuple(
            rng.choice(
                row['id'],
                size=min(row['num_choices'], row['num_negatives']),
                replace=False,
            )
        ),
        axis=1
    )

    idx_chosen = neg_sampling["sampled_idx"].explode().values
    neg = neg[neg["id"].isin(idx_chosen)].drop(columns="id")

    return pd.concat([neg, pos], ignore_index=True).sample(frac=1).drop(columns='id')