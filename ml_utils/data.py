import pandas as pd
import polars as pl
import numpy as np

import lightfm.data
import collections
import dataclasses
import rectools
import typing
import enum
import abc
import os

from collections.abc import Iterable
from rectools.dataset import Dataset, DenseFeatures
from rectools.dataset.identifiers import IdMap
from rectools.dataset.interactions import Interactions

if typing.TYPE_CHECKING:
    from scipy.sparse import coo_matrix


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

    train_interactions: typing.Union[pd.DataFrame, typing.Tuple[pd.DataFrame, pd.DataFrame]]
    test_interactions: pd.DataFrame
    user_features: pd.DataFrame
    item_features: pd.DataFrame

    all_users: np.ndarray
    all_items: np.ndarray
    all_features: np.ndarray

    def save_for_fast_load(self, save_dir: str):
        self.train_interactions.to_csv(os.path.join(save_dir, 'train_interactions.csv'), index=False)
        self.test_interactions.to_csv(os.path.join(save_dir, 'test_interactions.csv'), index=False)

        if self.user_features is not None:
            self.user_features.to_csv(os.path.join(save_dir, 'user_features.csv'), index=False)

        if self.item_features is not None:
            self.item_features.to_csv(os.path.join(save_dir, 'item_features.csv'), index=False)

    @classmethod
    def fast_load(cls, load_dir: str):
        """
        Fast load already preprocessed and saved with Data.save_for_fast_load() method data
        """

        train_interactions = pd.read_csv(os.path.join(load_dir, 'train_interactions.csv'))
        test_interactions = pd.read_csv(os.path.join(load_dir, 'test_interactions.csv'))

        if 'user_features.csv' in os.listdir(load_dir):
            user_features = pd.read_csv(os.path.join(load_dir, 'user_features.csv'))
        else:
            user_features = None

        if 'item_features.csv' in os.listdir(load_dir):
            item_features = pd.read_csv(os.path.join(load_dir, 'item_features.csv'))
        else:
            item_features = None

        return Data(
            train_interactions=train_interactions,
            test_interactions=test_interactions,
            user_features=user_features,
            item_features=item_features
        )

    def __init__(
            self,
            train_interactions: typing.Union[pd.DataFrame, typing.Tuple[pd.DataFrame, pd.DataFrame]],
            test_interactions: pd.DataFrame,
            user_features: pd.DataFrame,
            item_features: pd.DataFrame
    ):
        self.train_interactions = train_interactions
        self.test_interactions = test_interactions
        self.user_features = user_features
        self.item_features = item_features

        self.all_features = []

        if self.user_features is not None:
            self.all_features += [c for c in user_features.columns if c != 'user_id']

        if self.item_features is not None:
            self.all_features += [c for c in item_features.columns if c != 'item_id']

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

        if user_features is not None:
            self.all_users = np.array(list(set(self.all_users).intersection(set(user_features['user_id'].values))))

        if item_features is not None:
            self.all_items = np.array(list(set(self.all_items).intersection(set(item_features['item_id'].values))))

    def get_lightfm_features(
            self,
            drop_features: list = None,
            list_values_columns: list = None,
            scalar_values_columns: list = None
    ) -> 'coo_matrix':
        drop_features = drop_features if drop_features is not None else []
        list_values_columns = list_values_columns if list_values_columns is not None else []
        scalar_values_columns = scalar_values_columns if scalar_values_columns is not None else []

        def _parse_feature_names(column: str, _data: pd.DataFrame):

            if column in drop_features:
                return []
            if column in scalar_values_columns:
                return [column]
            if column in list_values_columns:
                return [f"{column}_{_}" for _ in set(np.concatenate(_data[column].values))]
            return [f"{column}_{_}" for _ in set(_data[column].values)]

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

        return collections.namedtuple(
            'LightFM_Features',
            ['user_features', 'item_features', 'user_labels', 'item_labels']
        )(
            user_features=tuple([(i, _parse_feature_row(r)) for i, r in user_features.iterrows()]),
            item_features=tuple([(i, _parse_feature_row(r)) for i, r in item_features.iterrows()]),
            user_labels=user_labels,
            item_labels=item_labels
        )

    def set_xgboost_features(
            self,
            interactions: pd.DataFrame
    ) -> pd.DataFrame:
        interactions = pd.merge(interactions, self.user_features, on='user_id', how='inner')
        interactions = pd.merge(interactions, self.item_features, on='item_id', how='inner')

        return interactions

    def get_rectools_dataset(
            self,
            item_features: pd.DataFrame = None,
            user_features: pd.DataFrame = None
    ) -> Dataset:
        interactions_cols = [rectools.Columns.User, rectools.Columns.Item, rectools.Columns.Weight,
                             rectools.Columns.Datetime]

        train = self.train_interactions.rename(columns={
            'user_id': rectools.Columns.User,
            'item_id': rectools.Columns.Item,
            'weight': rectools.Columns.Weight,
            'timestamp': rectools.Columns.Datetime
        })

        if user_features is not None:
            users = set(train['user_id'].values).intersection(set(user_features['user_id'].values))
            train = train[train['user_id'].isin(users)]
            train_user_features = user_features[user_features['user_id'].isin(users)]

        if item_features is not None:
            items = set(train['item_id'].values).intersection(set(item_features['item_id'].values))
            train = train[train['item_id'].isin(items)]
            train_item_features = item_features[item_features['item_id'].isin(items)]

        user_id_map = IdMap.from_values(train[rectools.Columns.User].values)
        item_id_map = IdMap.from_values(train[rectools.Columns.Item].values)
        train_interactions = Interactions.from_raw(train, user_id_map, item_id_map)

        train_dataset = Dataset(
            user_id_map, item_id_map, train_interactions,
            user_features=DenseFeatures.from_dataframe(train_user_features.drop_duplicates('user_id'), user_id_map,
                                                       'user_id') if user_features is not None else None,
            item_features=DenseFeatures.from_dataframe(train_item_features.drop_duplicates('item_id'), item_id_map,
                                                       'item_id') if item_features is not None else None
        )

        return train_dataset

    def get_lightfm_dataset(
            self,
            with_features: bool = True,
            drop_features: list = None,
            list_values_columns: list = None,
            scalar_values_columns: list = None
    ) -> lightfm.data.Dataset:
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


# Filters


class FilterStrategy(abc.ABC):
    @abc.abstractmethod
    def filter(self, data: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError()


class NoFilter(FilterStrategy):
    def filter(self, data: pl.DataFrame) -> pl.DataFrame:
        return data


@dataclasses.dataclass
class MinNumInteractionsFilter(FilterStrategy):
    min_user_ints: int = None
    min_item_ints: int = None

    def filter(self, data: pl.DataFrame) -> pl.DataFrame:
        user_id2num_inters = data['user_id'].value_counts()
        user_id2num_inters = dict(zip(user_id2num_inters['user_id'], user_id2num_inters['count']))
        item_id2num_inters = data['item_id'].value_counts()
        item_id2num_inters = dict(zip(item_id2num_inters['item_id'], item_id2num_inters['count']))

        if self.min_user_ints:
            data = data.filter(data['user_id'].map_elements(lambda x: user_id2num_inters[x] > self.min_user_ints))

        if self.min_item_ints:
            data = data.filter(data['item_id'].map_elements(lambda x: item_id2num_inters[x] > self.min_item_ints))

        return data


@dataclasses.dataclass
class OnlyLastInteractionsFilter(FilterStrategy):
    filter_column: str
    n_last: int

    def filter(self, data: pl.DataFrame) -> pl.DataFrame:
        data = data.with_columns(index=list(np.arange(len(data))))
        data = data.group_by(self.filter_column)
        data = data.map_groups(lambda g: g.sort('timestamp', descending=True).head(self.n_last))
        return data


# Splitters


class SplitStrategy(abc.ABC):

    @abc.abstractmethod
    def split(self, data: pl.DataFrame) -> typing.Tuple[pl.DataFrame, ...]:
        raise NotImplementedError()


class NoSplit(SplitStrategy):
    def split(self, data: pl.DataFrame) -> typing.Tuple[pl.DataFrame, ...]:
        return data, pl.DataFrame(schema=data.schema), pl.DataFrame(schema=data.schema)


@dataclasses.dataclass
class TimeSortSplit(SplitStrategy):
    num_interactions: typing.Union[int, typing.Literal['all']]
    splits: typing.Union[typing.Tuple[float, float], typing.Tuple[float, float, float]]

    def split(self, data: pl.DataFrame) -> tuple:
        self.num_interactions = len(data) if self.num_interactions == 'all' else self.num_interactions

        if len(self.splits) == 3:
            first_stage_train_split = self.splits[0]
            second_stage_train_split = self.splits[1]

            fs_interactions = self.num_interactions * (1 - first_stage_train_split)
            ss_interactions = self.num_interactions * (1 - first_stage_train_split - second_stage_train_split)
            data = data.sort('timestamp', descending=True)
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
        else:
            train_split = self.splits[0]

            train_interactions = self.num_interactions * (1 - train_split)
            data = data.sort('timestamp', descending=True)
            data = data.with_columns(index=list(np.arange(len(data))))

            train = data.filter(
                (pl.col('index') < self.num_interactions)
                &
                (pl.col('index') > train_interactions)
            )

            test = data.filter(
                (pl.col('index') < train_interactions)
            )

            return train, test


@dataclasses.dataclass
class RandomSplit(SplitStrategy):
    num_interactions: typing.Union[int, typing.Literal['all']]
    splits: typing.Union[typing.Tuple[float, float], typing.Tuple[float, float, float]]
    seed: int = 1337

    def split(self, data: pl.DataFrame) -> tuple:
        self.num_interactions = len(data) if self.num_interactions == 'all' else self.num_interactions

        if len(self.splits) == 3:
            fs_interactions = self.num_interactions * (1 - self.first_stage_train_split)
            ss_interactions = self.num_interactions * (1 - self.first_stage_train_split - self.second_stage_train_split)
            data = data.sample(fraction=1., shuffle=True, seed=self.seed)
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
        else:
            train_interactions = self.num_interactions * (1 - self.first_stage_train_split)
            data = data.sample(fraction=1., shuffle=True, seed=self.seed)
            data = data.with_columns(index=list(np.arange(len(data))))

            train = data.filter(
                (pl.col('index') < self.num_interactions)
                &
                (pl.col('index') > train_interactions)
            )

            test = data.filter(
                (pl.col('index') < train_interactions)
            )

            return train, test


# Weights

class WeightStrategy(abc.ABC):

    @abc.abstractmethod
    def get_weights(self, data: pl.DataFrame) -> pl.Series:
        raise NotImplementedError()


class NoWeight(WeightStrategy):

    def get_weights(self, data: pl.DataFrame) -> pl.Series:
        return pl.Series(name='weight', values=np.ones(len(data)))


@dataclasses.dataclass
class NumViewsBasedWeight(WeightStrategy):
    max_num_views: typing.Union[int, float]
    clip: typing.Tuple[int, int] = (0, 1)

    def get_weights(self, data: pl.DataFrame) -> pl.Series:
        num_views = data['item_id'].value_counts(parallel=True)

        if isinstance(self.max_num_views, float):
            self.max_num_views = int(num_views['count'].quantile(self.max_num_views))

        num_views = num_views.with_columns(
            count=num_views['count'].map_elements(lambda x: np.min(self.max_num_views, x)))
        max_num_views = num_views['count'].max()

        num_views = num_views.with_columns(
            weigth=np.clip(num_views['count'] / max_num_views, a_min=self.clip[0], a_max=self.clip[1])
        ).drop('count')

        return data.join(num_views, on='item_id', how='left')['weight']


@dataclasses.dataclass
class ViewTimeBasedWeight(WeightStrategy):
    n_last: int
    clip: typing.Tuple[int, int] = (0, 1)

    def get_weights(self, data: pl.DataFrame) -> pl.Series:
        timestamps = data[['user_id', 'item_id', 'timestamp']].sort('timestamp', descending=True)
        timestamps = timestamps.with_columns(
            weight=timestamps.group_by(['user_id', 'item_id']).agg(
                pl.col('timestamp').cum_count().map_elements(lambda x: np.max(self.n_last, x))
            )
        )
        timestamps = timestamps.with_columns(
            weight=timestamps.group_by(['user_id', 'item_id']).map_groups(lambda g: g['weight'] / g['weight'].max())
        )

        timestamps = timestamps.with_columns(
            weigth=np.clip(timestamps['weight'], a_min=self.clip[0], a_max=self.clip[1])
        ).drop('timestamp')

        return data.join(timestamps, on=['user_id', 'item_id'], how='left')['weight']


@dataclasses.dataclass
class ViewRatioBasedWeight(WeightStrategy):
    clip: typing.Tuple[int, int] = (0, 1)

    def get_weights(self, data: pl.DataFrame) -> pl.Series:
        return pl.Series(values=np.clip(data['weight'], a_min=self.clip[0], a_max=self.clip[1]), name='weight')


# Features

@dataclasses.dataclass
class FeaturesConfig:
    """
    Parameters:
    ----------


    features :
        List of features for use (rank, year, mppa, genres, runtime, device, account_type, lifetime) or `all`

    use_labels :
        Set `True` if using classic `lightfm.data.Dataset`
    """

    features: typing.Union[typing.Iterable[str], typing.Literal['all']] = 'all'
    use_labels: bool = False

    _dtypes = {
        'item_id': pl.String,
        'rank': pl.String,
        'year': pl.String,
        'mppa': pl.String,
        'genres': pl.String,
        'runtime': pl.String
    }

    def get_features(self) -> typing.Tuple[pl.DataFrame, pl.DataFrame]:
        if self.features == 'all':
            self.features = ['rank', 'year', 'mppa', 'genres', 'runtime', 'device', 'account_type', 'lifetime']

        if self.use_labels:
            item_features = pl.read_csv(
                os.path.join(os.environ['DIR'], 'data/item_features.csv'),
                dtypes={k: self._dtypes[k] for k in self._dtypes if k in self.features},
                columns=['item_id'] + [c for c in ['rank', 'year', 'mppa', 'genres', 'runtime'] if c in self.features]
            )
            user_features = pl.read_csv(
                os.path.join(os.environ['DIR'], 'data/user_features.csv'),
                columns=['user_id'] + [c for c in ['device', 'account_type', 'lifetime'] if c in self.features]
            )

            parse_genres = lambda genres: genres.replace('"', '').replace("'", '').replace(']', '').replace('[',
                                                                                                            '').replace(
                ' ', '').split(',')

            item_features = item_features.with_columns(
                genres=item_features['genres'].map_elements(parse_genres)
            )
        else:
            item_features = pl.read_csv(os.path.join(os.environ['DIR'], 'data/item_features_bin.csv'))
            user_features = pl.read_csv(os.path.join(os.environ['DIR'], 'data/user_features_bin.csv'))

        return user_features, item_features


# Config

@dataclasses.dataclass
class DataConfig:
    """
    Warning:
    ----------

    Length of output data may be less then original data length because user/item filtering


    Parameters:
    ----------

    split_strategy :
        Data splitting config.

    filter_strategy :
        Data filtering config or sequence of configs.

        If sequense was passed, filters will be compute sequentially - from first element of sequense to last.

    weight_strategy :
        Weight computing config or iterable of tuple pairs (`WeightStrategy`, `weight`), where `weight` is float value in range 0..1.

        If dict was passed, final weight will be computing with next formula:
            `final_weight` = `WeightStrategyResult[0]` * `weight[0]` + ... + `WeightStrategyResult[n]` * `weight[n]`.

            Where:
                `WeightStrategyResult` - is sequence of `pl.Series` of weights, that will be computing by current `WeightStrategy` instances,
                `n` - is number of pairs in input dict.
    """

    split_strategy: SplitStrategy = NoSplit()
    filter_strategy: typing.Union[FilterStrategy, typing.Iterable[FilterStrategy]] = NoFilter()
    weight_strategy: typing.Union[typing.Iterable[typing.Tuple[WeightStrategy, float]], WeightStrategy] = NoWeight()
    features_config: FeaturesConfig = None


def load_data(config: DataConfig, verbose_lens: bool = True) -> Data:
    """
    Loads all data for given config settings and pack it in `Data` class
    """

    interactions = pl.read_csv(
        os.path.join(os.environ['DIR'], 'data/interactions.csv'),
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

    if config.features_config is not None:
        user_features, item_features = config.features_config.get_features()

        item_features = item_features.drop_nulls()
        user_features = user_features.drop_nulls()
        interactions = interactions.filter(pl.col('user_id').is_in(set(user_features['user_id'].to_list())))
        interactions = interactions.filter(pl.col('item_id').is_in(set(item_features['item_id'].to_list())))

    splits = config.split_strategy.split(interactions)
    all_users = splits[0][['user_id']].unique()
    all_items = splits[0][['item_id']].unique()

    if len(splits) == 3:
        train_1 = splits[0]
        train_2 = splits[1]
        test = splits[2]

        if isinstance(config.weight_strategy, dict):
            weights = pl.Series(values=np.zeros(len(train_1)), name='weight')

            for weight_strategy, weight in config.weight_strategy.items():
                weights = weights + (weight_strategy.get_weights(train_1) * weight)
        else:
            weights = config.weight_strategy.get_weights(train_1)

        train_1 = train_1.to_pandas()
        train_2 = train_2.join(all_users, how='inner', on='user_id')
        train_2 = train_2.join(all_items, how='inner', on='item_id').to_pandas()
        test = test.join(all_users, how='inner', on='user_id')
        test = test.join(all_items, how='inner', on='item_id').to_pandas()

        train_1['weight'] = weights.to_numpy()
        train = (train_1, train_2)

        if verbose_lens:
            print(
                "Data after filter:\n" +
                f"Len of train_1 interactions with period [{train_1.tail(1)['timestamp'].values} / {train_1.head(1)['timestamp'].values}] - {len(train_1)}\n" +
                f"Len of train_2 interactions with period [{train_2.tail(1)['timestamp'].values} / {train_2.head(1)['timestamp'].values}] - {len(train_2)}\n" +
                f"Len of test interactions with period [{test.tail(1)['timestamp'].values} / {test.head(1)['timestamp'].values}] - {len(test)}"
            )
    else:
        train = splits[0]

        if isinstance(config.weight_strategy, dict):
            weights = pl.Series(values=np.zeros(len(train)), name='weight')

            for weight_strategy, weight in config.weight_strategy.items():
                weights = weights + (weight_strategy.get_weights(train) * weight)
        else:
            weights = config.weight_strategy.get_weights(train)

        train = train.to_pandas()
        test = splits[1].to_pandas()
        train['weight'] = weights.to_numpy()

        if verbose_lens:
            print(
                "Data after filter:\n" +
                f"Len of train interactions with period [{train.tail(1)['timestamp'].values} / {train.head(1)['timestamp'].values}] - {len(train)}\n" +
                f"Len of test interactions with period [{test.tail(1)['timestamp'].values} / {test.head(1)['timestamp'].values}] - {len(test)}\n" +
                f"Num of uniq users {len(all_users)}" +
                f"Num of uniq items {len(all_items)}"
            )

    return Data(
        train_interactions=train,
        test_interactions=test,
        user_features=user_features.to_pandas() if config.features_config is not None else None,
        item_features=item_features.to_pandas() if config.features_config is not None else None
    )


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
