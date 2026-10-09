"""Симметричные графы kNN с общим или адаптивным масштабом расстояния."""
import numpy as np
from sklearn.neighbors import kneighbors_graph


def economic_graph(x, neighbors):
    """Общий масштаб расстояний; x имеет форму (муниципалитеты, признаки)."""
    adjacency = kneighbors_graph(
        x, min(neighbors, len(x) - 1), mode='distance', include_self=False
    )
    positive_distances = adjacency.data[adjacency.data > 0]
    distance_scale = np.median(positive_distances) if len(positive_distances) else 1.
    adjacency.data = np.exp(-np.square(adjacency.data / distance_scale))
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    return adjacency


def local_graph(x, neighbors):
    """Локальный масштаб расстояний; возвращает разреженный граф размера n × n."""
    adjacency = kneighbors_graph(
        x, min(neighbors, len(x) - 1), mode='distance', include_self=False
    )
    neighbor_radii = adjacency.max(axis=1).toarray().ravel()
    positive_distances = adjacency.data[adjacency.data > 0]
    fallback_radius = np.median(positive_distances) if len(positive_distances) else 1.
    neighbor_radii = np.where(neighbor_radii > 0, neighbor_radii, fallback_radius)

    edge_sources = np.repeat(np.arange(len(x)), np.diff(adjacency.indptr))
    adjacency.data = np.exp(
        -np.square(adjacency.data)
        / (neighbor_radii[edge_sources] * neighbor_radii[adjacency.indices])
    )
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    return adjacency
