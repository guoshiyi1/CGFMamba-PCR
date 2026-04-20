import importlib


ext_module = importlib.import_module('geotransformer.ext')


def grid_subsample(points, lengths, voxel_size):

    s_points, s_lengths = ext_module.grid_subsampling(points, lengths, voxel_size)
    return s_points, s_lengths


def grid_subsample_dps(points, dps, lengths, voxel_size):

    s_points, s_dps, s_lengths = ext_module.grid_subsampling_dps(points, dps.float(), lengths, voxel_size)
    return s_points, s_dps, s_lengths