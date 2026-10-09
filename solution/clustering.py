"""Спектральные группы, согласование меток, переходы и сравнительные модели."""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.manifold import spectral_embedding
from sklearn.metrics import adjusted_rand_score as ARI, silhouette_score
from sklearn.preprocessing import RobustScaler
from . import METHOD
from .features import regionalize
from .graph import economic_graph, local_graph
from .metrics import evaluate


def spectral_coordinates(graph, k, seed):
    """Спектральные координаты формы (n, k) с единичной длиной каждой строки."""
    coordinates = spectral_embedding(
        graph, n_components=k, random_state=seed, drop_first=False
    )
    return coordinates / np.maximum(
        np.linalg.norm(coordinates, axis=1, keepdims=True), 1e-12
    )


def cluster_features(x, method, k, seed, graph=None, n_init=20):
    """Метки KMeans по исходным или спектральным координатам."""
    z = spectral_coordinates(graph, k, seed) if method == 'spectral' else x
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(z)


def cluster_graph_seeds(graph, k, seeds, embedding_seed, n_init):
    """Список массивов меток формы (n,): одни координаты, разные seed KMeans."""
    coordinates = spectral_coordinates(graph, k, embedding_seed)
    return [
        KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(coordinates)
        for seed in seeds
    ]


def align(previous, current):
    """Перенумеровывает current по пересечению с previous, оба массива формы (n,)."""
    overlap = pd.crosstab(current, previous)
    current_rows, previous_columns = linear_sum_assignment(-overlap.to_numpy())
    label_mapping = {
        overlap.index[row]: overlap.columns[column]
        for row, column in zip(current_rows, previous_columns)
        if overlap.iloc[row, column] > 0
    }
    next_label = previous.max() + 1
    for label in np.unique(current):
        if label not in label_mapping:
            label_mapping[label] = next_label
            next_label += 1
    return np.array([label_mapping[label] for label in current], int)


def reliability(reference, variations):
    """Jaccard каждой группы с наиболее похожей группой повторного запуска."""
    group_rows = []
    for variation_name, variation_labels in variations.items():
        for reference_group in np.unique(reference):
            reference_members = reference == reference_group
            matching_scores = [
                (
                    np.sum(reference_members & (variation_labels == group))
                    / np.sum(reference_members | (variation_labels == group)),
                    group,
                )
                for group in np.unique(variation_labels)
            ]
            best_match = max(matching_scores)
            group_rows.append(dict(
                variation=variation_name,
                cluster=reference_group,
                n=reference_members.sum(),
                jaccard=best_match[0],
                matched=best_match[1],
            ))
    return pd.DataFrame(group_rows)


def transition_flows(previous, current, month):
    """Все ненулевые потоки, включая муниципалитеты, сохранившие группу."""
    overlap = pd.crosstab(previous, current)
    return [
        dict(month=month, source=source, target=target, count=count)
        for source, row in overlap.iterrows()
        for target, count in row.items()
        if count
    ]


def fit_main_months(cube, months, k, config, initial_labels=None):
    """Кластеризовать cube (месяцы, МО, 7 признаков) и согласовать номера групп.

    Возвращает labels, месячные метрики, чувствительность и разреженные graphs.
    Проверки устойчивости являются частью исследования, а не проверками программы.
    """
    if len(cube) != len(months):
        raise ValueError('Каждому месяцу требуется одна матрица признаков')
    seeds = config['seeds']
    n_init = config['clustering']['n_init']
    neighbors = config['graph']['neighbors']
    sensitivity = config['sensitivity']
    monthly_labels, metric_rows, sensitivity_rows, graphs = [], [], [], []
    previous_labels = initial_labels
    for month_index, (month, features) in enumerate(zip(months, cube)):
        graph = local_graph(features, neighbors)
        seeded = cluster_graph_seeds(graph, k, seeds, config['random_seed'], n_init)
        labels = align(previous_labels, seeded[0]) if previous_labels is not None else seeded[0]
        variations = {f'seed_{seed}': variant for seed, variant in zip(seeds[1:], seeded[1:])}
        if month_index in set(sensitivity['months']) | {len(months) - 1}:
            for block, columns in [('basket', slice(0, 6)), ('spending', slice(6, 7))]:
                for factor in sensitivity['weight_factors']:
                    changed = features.copy()
                    changed[:, columns] *= np.sqrt(factor)
                    variations[f'weight_{block}_{factor}'] = cluster_features(
                        changed, 'spectral', k, config['random_seed'],
                        local_graph(changed, neighbors), n_init)
            changed = features + np.random.default_rng(sensitivity['noise_seed'] + month_index).normal(
                0, sensitivity['noise_std'], features.shape)
            variations['noise'] = cluster_features(changed, 'spectral', k, config['random_seed'],
                local_graph(changed, neighbors), n_init)
            for count in sensitivity['neighbors']:
                variations[f'neighbors_{count}'] = cluster_features(features, 'spectral', k,
                    config['random_seed'], local_graph(features, count), n_init)
        group_checks = reliability(labels, variations)
        sensitivity_rows.extend(dict(month=month, variation=name, ARI=ARI(labels, variant))
            for name, variant in variations.items())
        sizes = np.unique(labels, return_counts=True)[1]
        metric_rows.append(dict(method=METHOD, month=month, n=len(labels), k=k,
            **evaluate(features, labels, graph, config['random_seed'], config['metrics']['silhouette_sample']),
            SW_full=silhouette_score(features, labels), min_size=sizes.min(), max_share=sizes.max()/len(labels),
            seed_ARI_min=min(ARI(labels, variant) for variant in seeded[1:]),
            group_Jaccard_min=group_checks.jaccard.min(),
            switch_rate=np.mean(labels != previous_labels) if previous_labels is not None else None,
            evaluation_space='Smoothed consumer coordinates; own local graph'))
        monthly_labels.append(labels)
        previous_labels = labels
        graphs.append(graph)
    return dict(labels=np.array(monthly_labels), monthly=pd.DataFrame(metric_rows),
        sensitivity=pd.DataFrame(sensitivity_rows), graphs=graphs)


def fit_baselines(common, smoothed, months, static_labels, k, config):
    """Четыре последовательности меток (месяцы, n) и таблица проверок по seed."""
    seeds = config['seeds']
    n_init = config['clustering']['n_init']
    sequences = {}
    seed_rows = []
    method_names = [
        'kmeans', 'spectral', 'temporal_spectral',
        'adaptive_spectral',
    ]
    for method_name in method_names:
        previous_labels = static_labels if not method_name.startswith('adaptive_') else None
        monthly_labels = []

        for month_index, month in enumerate(months):
            features = smoothed[month_index] if 'temporal' in method_name else common[month_index]
            graph_builder = local_graph if method_name.startswith('adaptive_') else economic_graph
            graph = graph_builder(features, config['graph']['neighbors'])
            if method_name == 'kmeans':
                seeded_labels = [cluster_features(features, 'kmeans', k, seed, n_init=n_init) for seed in seeds]
            else:
                seeded_labels = cluster_graph_seeds(graph, k, seeds, config['random_seed'], n_init)

            labels = (
                align(previous_labels, seeded_labels[0])
                if previous_labels is not None else seeded_labels[0]
            )
            group_checks = reliability(labels, {
                str(seed): variant_labels
                for seed, variant_labels in zip(seeds[1:], seeded_labels[1:])
            })
            seed_rows.append(dict(
                method=method_name,
                month=month,
                seed_count=len(seeds),
                seed_ARI_min=min(ARI(labels, variant) for variant in seeded_labels[1:]),
                seed_group_Jaccard_min=group_checks.jaccard.min(),
            ))
            monthly_labels.append(labels)
            previous_labels = labels
        sequences[method_name] = np.array(monthly_labels)

    return sequences, pd.DataFrame(seed_rows)


def baseline_sequences(panel, consumer_z, smoothed, months, k, config):
    """Четыре метода при выбранном K и прототипы из среднего профиля 2023."""
    annual_relative = regionalize(
        panel["raw_cube"][:12].mean(axis=0), panel["regions"]
    )
    static_scaler = RobustScaler().fit(annual_relative)
    static_space = np.clip(static_scaler.transform(annual_relative), -config["features"]["static_clip"], config["features"]["static_clip"])
    static_space[:, :6] /= np.sqrt(6)
    static_space /= np.sqrt(2)
    static_labels = cluster_features(
        static_space, "spectral", k, config['random_seed'], economic_graph(static_space, config['graph']['neighbors']), n_init=config['clustering']['n_init']
    )

    baselines, baseline_seed_checks = fit_baselines(
        consumer_z[12:], smoothed[12:], months, static_labels, k=k, config=config
    )
    sequences = baselines


    for cluster_count in config['clustering']['anchored_counts']:
        training_consumer = consumer_z[:12]
        annual_mean = training_consumer.mean(axis=0)
        annual_labels = cluster_features(annual_mean, "kmeans", cluster_count, config['random_seed'], n_init=config['clustering']['n_init'])

        group_order = sorted(
            range(cluster_count),
            key=lambda group: -annual_mean[annual_labels == group, -1].mean(),
        )
        group_mapping = {
            old_group: new_group for new_group, old_group in enumerate(group_order)
        }
        annual_labels = np.array([group_mapping[label] for label in annual_labels])

        centers = np.stack([
            training_consumer[:, annual_labels == group].mean(axis=(0, 1))
            for group in range(cluster_count)
        ])
        distances_to_centers = np.linalg.norm(
            smoothed[:, :, None, :] - centers[None, None, :, :],
            axis=-1,
        )
        monthly_labels = distances_to_centers.argmin(axis=-1)
        method_name = f"anchored_k{cluster_count}"

        sequences[method_name] = monthly_labels[12:]


    return sequences, baseline_seed_checks
