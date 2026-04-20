import importlib


ext_module = importlib.import_module('geotransformer.ext')


def radius_search(q_points, s_points, q_lengths, s_lengths, radius, neighbor_limit):

    neighbor_indices = ext_module.radius_neighbors(q_points, s_points, q_lengths, s_lengths, radius)
    if neighbor_limit > 0:
        neighbor_indices = neighbor_indices[:, :neighbor_limit]
    return neighbor_indices
