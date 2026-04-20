import os
import os.path as osp
import argparse

from easydict import EasyDict as edict

from geotransformer.utils.common import ensure_dir


_C = edict()

# common
_C.seed = 7351

# dirs
_C.working_dir = osp.dirname(osp.realpath(__file__))
_C.root_dir = osp.dirname(osp.dirname(_C.working_dir))
_C.exp_name = osp.basename(_C.working_dir)
_C.output_dir = osp.join(_C.root_dir, 'output', _C.exp_name)
_C.snapshot_dir = osp.join(_C.output_dir, 'snapshots')
_C.log_dir = osp.join(_C.output_dir, 'logs')
_C.event_dir = osp.join(_C.output_dir, 'events')
_C.feature_dir = osp.join(_C.output_dir, 'features')
_C.registration_dir = osp.join(_C.output_dir, 'registration')

ensure_dir(_C.output_dir)
ensure_dir(_C.snapshot_dir)
ensure_dir(_C.log_dir)
ensure_dir(_C.event_dir)
ensure_dir(_C.feature_dir)
ensure_dir(_C.registration_dir)

# data
_C.data = edict()
_C.data.dataset_root = ''

# test data
_C.test = edict()
_C.test.batch_size = 1
_C.test.num_workers = 8
_C.test.point_limit = None
_C.train = edict()

_C.train.point_limit = 25000
_C.train.use_augmentation = True
_C.train.augmentation_noise = 0.005
_C.train.augmentation_rotation = 1.0

# model - backbone
_C.backbone = edict()
_C.backbone.num_stages = 4
_C.backbone.init_voxel_size = 0.025
_C.backbone.kernel_size = 15
_C.backbone.base_radius = 2.5
_C.backbone.base_sigma = 2.0
_C.backbone.init_radius = _C.backbone.base_radius * _C.backbone.init_voxel_size
_C.backbone.init_sigma = _C.backbone.base_sigma * _C.backbone.init_voxel_size
_C.backbone.group_norm = 32
_C.backbone.input_dim = 1
_C.backbone.init_dim = 64
_C.backbone.output_dim = 256

# model - Global
_C.model = edict()
_C.model.ground_truth_matching_radius = 0.05
_C.model.num_points_in_patch = 64
_C.model.num_sinkhorn_iterations = 100

# model - Coarse Matching
_C.coarse_matching = edict()
_C.coarse_matching.num_targets = 128
_C.coarse_matching.overlap_threshold = 0.1
_C.coarse_matching.num_correspondences = 256
_C.coarse_matching.dual_normalization = True
# evaluation
_C.eval = edict()
_C.eval.acceptance_overlap = 0.0
_C.eval.acceptance_radius = 0.1
_C.eval.inlier_ratio_threshold = 0.05
_C.eval.rmse_threshold = 0.2
_C.eval.rre_threshold = 15.0
_C.eval.rte_threshold = 0.3
# model - GeoTransformer
_C.geotransformer = edict()
_C.geotransformer.hidden_dim = 128
_C.geotransformer.input_dim = 1024
_C.geotransformer.sigma_d = 0.2
_C.geotransformer.sigma_hd = 0.1
_C.geotransformer.sigma_a = 15
_C.geotransformer.angle_k = 3
_C.geotransformer.reduction_a = 'max'


# model - Fine Matching
_C.fine_matching = edict()
_C.fine_matching.topk = 3
_C.fine_matching.acceptance_radius = 0.1
_C.fine_matching.mutual = True
_C.fine_matching.confidence_threshold = 0.05
_C.fine_matching.use_dustbin = False
_C.fine_matching.use_global_score = False
_C.fine_matching.correspondence_threshold = 3
_C.fine_matching.correspondence_limit = None
_C.fine_matching.num_refinement_steps = 5




#model - mamba_backbone
_C.model1 = edict()
_C.model1.NAME = 'CGFMamba'
_C.model1.trans_dim = 128
_C.model1.depth = 6
_C.model1.cls_dim = 40
_C.model1.num_heads = 6
_C.model1.group_size = 32
_C.model1.num_group = 5000
_C.model1.encoder_dims = 128
_C.model1.rms_norm = False
_C.model1.usehsv = False
_C.model1.drop_path = 0.3
_C.model1.drop_out = 0.
_C.model1.D = 128
_C.model1.backbone = True
_C.model1.usehsv = False
_C.model1.geopos = False
_C.model1.inputdim = 128
_C.model1.flag = False

# model - mamba_geo
_C.model5 = edict()
_C.model5.NAME = 'CGFMamba'
_C.model5.trans_dim = 128
_C.model5.depth = 12
_C.model5.cls_dim = 40
_C.model5.num_heads = 6
_C.model5.group_size = 32
_C.model5.num_group = 250
_C.model5.encoder_dims = 128
_C.model5.rms_norm = False
_C.model5.drop_path = 0.3
_C.model5.drop_out = 0.
_C.model5.D = 128
_C.model5.backbone = False
_C.model5.usehsv = False
_C.model5.geopos = True
_C.model5.inputdim = 1024
_C.model5.flag = False




def make_cfg():
    return _C


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--link_output', dest='link_output', action='store_true', help='link output dir')
    args = parser.parse_args()
    return args


def main():
    cfg = make_cfg()
    args = parse_args()
    if args.link_output:
        os.symlink(cfg.output_dir, 'output')


if __name__ == '__main__':
    main()
