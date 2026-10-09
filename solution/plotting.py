"""Рисунки исследования и фиксированные координаты графа для атласа."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy import sparse
from sklearn.decomposition import PCA
from matplotlib.collections import LineCollection
from . import METHOD


def save_figures(results, output_dir, config):
    """Сохранить четыре PNG; вернуть graph_xy (n, 2) и рёбра каждого месяца."""
    panel = results['panel']
    ids = results['ids']
    months = results['months']
    smoothed = results['smoothed']
    main = results['main']
    monthly_comparison_table = results['monthly_comparison_table']
    cluster_study = results['cluster_study']
    selected_k = results['selected_k']
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    for year in [2023, 2024]:
        rows = cluster_study.query('year == @year')
        for axis, column, title in zip(axes, ['SW_full', 'sensitivity_ARI_min', 'economic_separation'],
            ['Полная silhouette', 'Минимальная ARI чувствительности', 'Экономическое отличие профилей']):
            axis.plot(rows.k, rows[column], marker='o', label=str(year))
            axis.set(title=title, xlabel='Число групп K')
            axis.axvline(selected_k, color='gray', linestyle=':', alpha=.6)
            axis.set_xticks(rows.k)
            axis.legend()
    axes[1].axhline(config['screening']['min_sensitivity_ari'], color='firebrick', linestyle='--', linewidth=1)
    axes[2].axhline(config['screening']['min_economic_separation'], color='firebrick', linestyle='--', linewidth=1)
    axes[2].set_ylabel('Отличие медиан / общий IQR')
    figure.savefig(output_dir / 'cluster_count_comparison.png', dpi=config['visualization']['study_dpi'])
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")

    monthly_group_sizes = pd.DataFrame(
        [np.bincount(labels, minlength=selected_k) for labels in main["labels"]],
        index=months,
    )
    monthly_group_sizes.plot(
        ax=axes[0], marker="o", title="Размеры групп основного метода"
    )
    axes[0].set_ylabel("Муниципалитетов")
    axes[0].set_xlabel("Месяц 2024")

    for method_name in [METHOD, "adaptive_spectral", "kmeans"]:
        method_metrics = monthly_comparison_table.query("method == @method_name")
        axes[1].plot(
            method_metrics["month"],
            method_metrics["SW_full"],
            marker="o",
            label=method_name,
        )
    axes[1].set(title="Полная silhouette в общем пространстве", ylabel="SW")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend(fontsize=8)

    figure.savefig(output_dir / "comparison.png", dpi=config['visualization']['figure_dpi'])
    plt.close(figure)
    pca = PCA(2, random_state=config['random_seed']).fit(smoothed[:12].reshape(-1, 7))
    pca_coordinates = pca.transform(smoothed[-1])

    figure, axis = plt.subplots(figsize=(8, 5), layout="constrained")
    axis.scatter(
        pca_coordinates[:, 0],
        pca_coordinates[:, 1],
        c=main["labels"][-1],
        cmap="Set2",
        s=10,
        alpha=0.8,
    )
    axis.set(
        title="Декабрь 2024 · PCA основного 7-мерного пространства",
        xlabel="PC1",
        ylabel="PC2",
    )
    figure.savefig(output_dir / "consumer_pca.png", dpi=config['visualization']['figure_dpi'])
    plt.close(figure)
    average_graph = sum(main["graphs"]) / len(main["graphs"])
    graph_layout = nx.spring_layout(
        nx.from_scipy_sparse_array(average_graph),
        seed=config['random_seed'],
        method="energy",
        iterations=config['visualization']['layout_iterations'],
    )
    graph_xy = np.array([graph_layout[index] for index in range(len(ids))])

    edges_by_month = []
    for monthly_graph in main["graphs"]:
        upper_triangle = sparse.triu(monthly_graph, k=1).tocoo()
        edges_by_month.append(np.column_stack([
            upper_triangle.row,
            upper_triangle.col,
            upper_triangle.data,
        ]))

    december_edges = edges_by_month[-1]
    edge_segments = graph_xy[december_edges[:, :2].astype(int)]
    figure, axis = plt.subplots(figsize=(10, 6), layout="constrained")
    axis.add_collection(LineCollection(
        edge_segments,
        colors=(0.3, 0.4, 0.4, 0.1),
        linewidths=0.25,
    ))
    axis.scatter(
        graph_xy[:, 0], graph_xy[:, 1],
        c=main["labels"][-1], cmap="Set2", s=9,
    )
    axis.autoscale()
    axis.set_aspect("equal")
    axis.set_axis_off()
    axis.set_title(
        f"Граф сходства · {months[-1]} · {len(ids)} МО · {len(december_edges)} рёбер"
    )
    figure.savefig(output_dir / "consumer_network.png", dpi=config['visualization']['figure_dpi'])
    plt.close(figure)
    return graph_xy, edges_by_month
