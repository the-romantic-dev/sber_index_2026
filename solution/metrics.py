"""ICVI и сравнение всех методов в общем потребительском пространстве."""
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from s_dbw import S_Dbw
from sklearn.metrics import adjusted_rand_score as ARI, calinski_harabasz_score, silhouette_score
from .graph import economic_graph


def evaluate(x, labels, graph, seed=42, sample_size=600):
    """Метрики для x формы (n, p), меток (n,) и симметричного графа (n, n)."""
    municipality_count = len(x)
    _, labels = np.unique(labels, return_inverse=True)
    cluster_count = labels.max() + 1

    membership = sparse.csr_matrix(
        (np.ones(municipality_count), (labels, np.arange(municipality_count))),
        shape=(cluster_count, municipality_count),
    )
    cluster_weights = (membership @ graph @ membership.T).toarray()
    internal_weight = cluster_weights.diagonal()
    volume = cluster_weights.sum(axis=1)
    outgoing_weight = volume - internal_weight
    incoming_weight = cluster_weights.sum(axis=0) - internal_weight

    isolation = np.divide(
        internal_weight, volume, out=np.zeros(cluster_count), where=volume > 0
    )
    union_weight = outgoing_weight[:, None] + incoming_weight[None, :] - cluster_weights
    unification = np.divide(
        cluster_weights,
        union_weight,
        out=np.zeros_like(cluster_weights),
        where=union_weight > 0,
    )
    np.fill_diagonal(unification, 0)
    total_weight = volume.sum()
    modularity = (
        internal_weight.sum() / total_weight - np.sum((volume / total_weight) ** 2)
        if total_weight else 0.
    )
    # shortcut: provisional directed-arc TurboMQ convention, upgrade when organizers define MQ.
    modularization_quality = np.divide(
        2 * internal_weight,
        2 * internal_weight + outgoing_weight + incoming_weight,
        out=np.zeros(cluster_count),
        where=internal_weight > 0,
    ).sum()
    result = dict(
        SW=np.nan,
        CH=np.nan,
        S_Dbw=np.nan,
        AVI=isolation.mean(),
        AVU=unification.sum() / cluster_count,
        MQ=modularization_quality,
        Q=modularity,
    )
    if not 1 < cluster_count < municipality_count:
        return result

    sample_rows = np.random.default_rng(seed).choice(
        municipality_count, min(municipality_count, sample_size), replace=False
    )
    sampled_labels = labels[sample_rows]
    if 1 < len(np.unique(sampled_labels)) < len(sample_rows):
        result['SW'] = silhouette_score(x[sample_rows], sampled_labels, metric='euclidean')
    result['CH'] = calinski_harabasz_score(x, labels)
    try:
        with np.errstate(divide='raise', invalid='raise', over='raise'):
            result['S_Dbw'] = S_Dbw(
                x, labels, method='Tong', alg_noise='bind', centr='mean',
                nearest_centr=True, metric='euclidean',
            )
    except (ValueError, ZeroDivisionError, FloatingPointError) as exc:
        warnings.warn(f'S_Dbw undefined: {exc}', RuntimeWarning, stacklevel=2)
    return result


def monthly_comparison(months, common, sequences, config):
    """Сводная таблица; common имеет форму (месяцы, n, p), метки метода — (месяцы, n)."""
    metric_rows = []
    for month_index, month in enumerate(months):
        features = common[month_index]
        reference_graph = economic_graph(features, config['graph']['neighbors'])

        for method_name, labels in sequences.items():
            group_sizes = np.unique(labels[month_index], return_counts=True)[1]
            metric_rows.append(dict(
                method=method_name,
                month=month,
                n=len(features),
                k=len(group_sizes),
                **evaluate(features, labels[month_index], reference_graph, config['random_seed'], config['metrics']['silhouette_sample']),
                SW_full=silhouette_score(features, labels[month_index]),
                min_size=group_sizes.min(),
                max_share=group_sizes.max() / len(features),
                switch_rate=(
                    np.mean(labels[month_index] != labels[month_index - 1])
                    if month_index else np.nan
                ),
                ARI_previous=(
                    ARI(labels[month_index - 1], labels[month_index])
                    if month_index else np.nan
                ),
            ))
    return pd.DataFrame(metric_rows)
