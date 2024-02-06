from scipy.sparse import csr_matrix

import pandas as pd

import typing

if typing.TYPE_CHECKING:
    from rectools.metrics.base import Catalog, MetricAtK


def _validate_metric_kwargs(target: typing.Type['MetricAtK'], **kwargs) -> typing.Dict[str, typing.Any]:
    used_keys = target.calc.__annotations__.keys()
    for k in list(kwargs.keys()).copy():
        if k not in used_keys:
            kwargs.pop(k)
    return kwargs


def compute_rectools_metrics(
        rank_matrix: 'csr_matrix',
        uid_map: dict,
        iid_map: dict,
        eval_metrics: typing.Iterable[typing.Type['MetricAtK']],
        k=10,
        interactions: pd.DataFrame = None,
        prev_interactions: pd.DataFrame = None,
        catalog: 'Catalog' = None
) -> typing.Dict[str, float]:
    csr = rank_matrix.tocsr()
    nnz = csr.nonzero()
    reco = pd.DataFrame({
        'user_id': nnz[0],
        'item_id': nnz[1],
        'rank': map(csr.__getitem__, zip(*nnz))
    })

    reco['user_id'] = reco['user_id'].apply(uid_map.__getitem__)
    reco['item_id'] = reco['item_id'].apply(iid_map.__getitem__)
    reco['rank'] = reco['rank'].astype(int)

    out = {}

    for m in eval_metrics:
        mkwargs = _validate_metric_kwargs(
            m,
            reco=reco,
            interactions=interactions,
            prev_interactions=prev_interactions,
            catalog=catalog
        )
        out[m.__name__] = m(k).calc(**mkwargs)

    return out
