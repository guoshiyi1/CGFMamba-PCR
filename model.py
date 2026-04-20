import time
from sklearn.cluster import KMeans
import torch.nn.functional as F
import torch
from loss import Evaluator
from geotransformer.utils.torch import release_cuda
from geotransformer.modules.ops import point_to_node_partition, index_select
from geotransformer.modules.registration import get_node_correspondences
from backbone import KPConvFPN 
from geotransformer.engine import SingleTester
from geotransformer.modules.geotransformer import (
    SuperPointMatching,
    LocalGlobalRegistration,
    GeometricStructureEmbedding,
)
import torch.nn as nn
from tools import builder
import numpy as np
import os.path as osp
from dataset import test_data_loader
from geotransformer.utils.common import ensure_dir, get_log_string
import argparse
import torch.nn as nn
from geotransformer.modules.sinkhorn import LearnableLogOptimalTransport
import torch
from sklearn.cluster import KMeans

def kmeans_H_sort(color_c, n_clusters=10):
    H = color_c[:, 0].unsqueeze(1).cpu().numpy()  
    kmeans = KMeans(n_clusters=n_clusters).fit(H)
    labels = kmeans.labels_                     
    centers = kmeans.cluster_centers_.squeeze() 

    sorted_cluster_ids = torch.tensor(
        centers.argsort(), device=color_c.device
    )  


    cluster_labels = torch.tensor(labels, dtype=torch.long, device=color_c.device)

    cluster_order_map = torch.empty(n_clusters, device=color_c.device, dtype=torch.long)
    cluster_order_map[sorted_cluster_ids] = torch.arange(n_clusters, device=color_c.device)
    cluster_ranks = cluster_order_map[cluster_labels]  # [N]

    H_vals = color_c[:, 0]  # [N]
    sort_keys = cluster_ranks * 1e3 + H_vals  
    sorted_indices = torch.argsort(sort_keys)

    return sorted_indices

class mymodel(nn.Module):
    def  __init__(self, cfg):
        super(mymodel, self).__init__()
        self.base_model5 = builder.model_builder(cfg.model5)
        self.num_points_in_patch = cfg.model.num_points_in_patch
        self.matching_radius = cfg.model.ground_truth_matching_radius

        self.backbone = KPConvFPN(
            cfg.backbone.input_dim,
            cfg.backbone.output_dim,
            cfg.backbone.init_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.init_radius,
            cfg.backbone.init_sigma,
            cfg.backbone.group_norm,
            cfg,
        )
        self.hidden_dim = cfg.geotransformer.hidden_dim
        self.coarse_matching = SuperPointMatching(
            cfg.coarse_matching.num_correspondences, 0.98
        )

        self.fine_matching = LocalGlobalRegistration(
            cfg.fine_matching.topk,
            cfg.fine_matching.acceptance_radius,
            mutual=cfg.fine_matching.mutual,
            confidence_threshold=cfg.fine_matching.confidence_threshold,
            use_dustbin=cfg.fine_matching.use_dustbin,
            use_global_score=cfg.fine_matching.use_global_score,
            correspondence_threshold=cfg.fine_matching.correspondence_threshold,
            correspondence_limit=cfg.fine_matching.correspondence_limit,
            num_refinement_steps=cfg.fine_matching.num_refinement_steps,
        )
        self.linear = nn.Linear(cfg.geotransformer.hidden_dim, 4)
        self.optimal_transport = LearnableLogOptimalTransport(cfg.model.num_sinkhorn_iterations)
        self.embedding = GeometricStructureEmbedding(cfg.geotransformer.hidden_dim, cfg.geotransformer.sigma_d, cfg.geotransformer.sigma_a,
                                                     cfg.geotransformer.angle_k, reduction_a=cfg.geotransformer.reduction_a, sigma_hd=cfg.geotransformer.sigma_hd)
    def forward(self,data_dict):
        time0 = time.time()
        output_dict = {}

        feats = data_dict['features'].detach()
        transform = data_dict['transform'].detach()

        ref_length_c = data_dict['lengths'][-1][0].item()
        ref_length_f = data_dict['lengths'][1][0].item()
        ref_length = data_dict['lengths'][0][0].item()
        points_c = data_dict['points'][-1].detach()

        points_f = data_dict['points'][1].detach()#n,3
        points = data_dict['points'][0].detach()


        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]
        ref_points = points[:ref_length]
        src_points = points[ref_length:]

        output_dict['ref_points_c'] = ref_points_c
        output_dict['src_points_c'] = src_points_c
        output_dict['ref_points_f'] = ref_points_f
        output_dict['src_points_f'] = src_points_f
        output_dict['ref_points'] = ref_points
        output_dict['src_points'] = src_points
        #######
        colors_f = data_dict['hsv'][0].detach()
        ref_colors_f = colors_f[:ref_length_f]
        src_colors_f = colors_f[ref_length_f:]

        # 1. Generate ground truth node correspondences
        _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
            ref_points_f, ref_points_c,128
        )
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
            src_points_f, src_points_c,128
        )
        ref_padded_points_f = torch.cat([ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0)
        src_padded_points_f = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)
        ###########
        ref_padded_colors_f = torch.cat([ref_colors_f, torch.zeros_like(ref_colors_f[:1])], dim=0)
        src_padded_colors_f = torch.cat([src_colors_f, torch.zeros_like(src_colors_f[:1])], dim=0)
        ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices, dim=0)
        src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0)
        #############
        ref_node_knn_colors = index_select(ref_padded_colors_f, ref_node_knn_indices, dim=0)
        src_node_knn_colors = index_select(src_padded_colors_f, src_node_knn_indices, dim=0)

        gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
        ref_points_c,
        src_points_c,
        ref_node_knn_points,
        src_node_knn_points,
        transform,
        0.05,
        ref_masks=ref_node_masks,
        src_masks=src_node_masks,
        ref_knn_masks=ref_node_knn_masks,
        src_knn_masks=src_node_knn_masks,
            )
        output_dict['gt_node_corr_indices'] = gt_node_corr_indices
        output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps
        time1 = time.time()


        feats_list = self.backbone(feats, data_dict)
        time2 = time.time()
        
        color_c = data_dict['hsv'][-1].detach()
        ref_color = color_c[:ref_length_c]
        src_color = color_c[ref_length_c:]
        with torch.no_grad():
            sorted_indices = kmeans_H_sort(color_c, n_clusters=4)
            inverse_indices = torch.empty_like(sorted_indices)
            inverse_indices[sorted_indices] = torch.arange(len(sorted_indices), device=sorted_indices.device)
        ref_embeddings = self.embedding(ref_points_c.unsqueeze(0), hsv=ref_color.unsqueeze(0)).squeeze(0)
        src_embeddings = self.embedding(src_points_c.unsqueeze(0), hsv=src_color.unsqueeze(0)).squeeze(0)
        sum_weighted_features_r,_ = torch.max(ref_embeddings, dim=1)
        sum_weighted_features_s,_ = torch.max(src_embeddings, dim=1)
        sum_weighted_features = torch.cat((sum_weighted_features_r,sum_weighted_features_s),dim=0)
     
        feats_c = feats_list[-1]
        feats_f = feats_list[0]

        feats_ccolor = feats_c[sorted_indices]
        sum_weighted_features_color = sum_weighted_features[sorted_indices]
        output_dict['refk_feats_c'] = feats_c[:ref_length_c]
        output_dict['srck_feats_c'] = feats_c[ref_length_c:]
        sorted_x_indices = torch.argsort(points_c[:, 0]).cuda()
        sorted_y_indices = torch.argsort(points_c[:, 1]).cuda()
        sorted_z_indices = torch.argsort(points_c[:, 2]).cuda()
        inv_idx_x = torch.empty_like(sorted_x_indices).cuda()
        inv_idx_x[sorted_x_indices] = torch.arange(len(sorted_x_indices),device=sorted_x_indices.device)
        inv_idx_y = torch.empty_like(sorted_y_indices).cuda()
        inv_idx_y[sorted_y_indices] = torch.arange(len(sorted_y_indices),device=sorted_x_indices.device)
        inv_idx_z = torch.empty_like(sorted_z_indices).cuda()
        inv_idx_z[sorted_z_indices] = torch.arange(len(sorted_z_indices),device=sorted_x_indices.device)
        feats_cx = feats_c[sorted_x_indices]
        points_cx = points_c[sorted_x_indices]
        sum_weighted_features_x = sum_weighted_features[sorted_x_indices]
        feats_cy = feats_c[sorted_y_indices]
        points_cy = points_c[sorted_y_indices]
        sum_weighted_features_y = sum_weighted_features[sorted_y_indices]
        feats_cz = feats_c[sorted_z_indices]
        points_cz = points_c[sorted_z_indices]
        sum_weighted_features_z = sum_weighted_features[sorted_z_indices]
        time3 = time.time()

        pointsmamba_cx = self.base_model5(feats_cx.unsqueeze(0),sum_weighted_features_x.unsqueeze(0),points_cx.unsqueeze(0)).squeeze(0)                 
        pointsmamba_cy= self.base_model5(feats_cy.unsqueeze(0),sum_weighted_features_y.unsqueeze(0),points_cy.unsqueeze(0)).squeeze(0)    
        pointsmamba_cz = self.base_model5(feats_cz.unsqueeze(0),sum_weighted_features_z.unsqueeze(0),points_cz.unsqueeze(0)).squeeze(0)    
        pointsmamba_color = self.base_model5(feats_ccolor.unsqueeze(0),sum_weighted_features_color.unsqueeze(0),points_cz.unsqueeze(0)).squeeze(0)    
     
        pointsmamba_x= pointsmamba_cx[inv_idx_x]
        pointsmamba_y= pointsmamba_cy[inv_idx_y]
        pointsmamba_z= pointsmamba_cz[inv_idx_z]
        pointsmamba_color= pointsmamba_color[inverse_indices]
        concatenated_features = torch.cat((pointsmamba_x, pointsmamba_y, pointsmamba_z,pointsmamba_color), dim=0)
        f = torch.mean(concatenated_features, dim=0)
        weights = nn.functional.softmax(self.linear(f), dim=0)
        f_final = weights[0] * pointsmamba_x + weights[1] * pointsmamba_y + weights[2] * pointsmamba_z + weights[3] * pointsmamba_color
        ref_feats_c = f_final[:ref_length_c]
        src_feats_c = f_final[ref_length_c:]

        ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)
        output_dict['ref_feats_c'] = ref_feats_c_norm
        output_dict['src_feats_c'] = src_feats_c_norm
        # 5. Head for fine level matching
        ref_feats_f = feats_f[:ref_length_f]
        src_feats_f = feats_f[ref_length_f:]
        output_dict['ref_feats_f'] = ref_feats_f
        output_dict['src_feats_f'] = src_feats_f
        time4 = time.time()

        # 6. Select topk nearest node correspondences
        with torch.no_grad():
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                ref_feats_c_norm, src_feats_c_norm, ref_node_masks, src_node_masks
            )
            output_dict['ref_node_corr_indices'] = ref_node_corr_indices
            output_dict['src_node_corr_indices'] = src_node_corr_indices
            
        output_dict['node_corr_scores'] = node_corr_scores
        # 7.2 Generate batched node points & feats
        ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices]  # (P, K, 3)
        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]  # (P, K, 3)
        #############
        ref_node_corr_knn_colors = ref_node_knn_colors[ref_node_corr_indices]  # (P, K, 3)
        src_node_corr_knn_colors = src_node_knn_colors[src_node_corr_indices]  # (P, K, 3)

        ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
        src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)
        ref_node_corr_knn_feats = index_select(ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)  # (P, K, C)
        src_node_corr_knn_feats = index_select(src_padded_feats_f, src_node_corr_knn_indices, dim=0)  # (P, K, C)

        output_dict['ref_node_corr_knn_points'] = ref_node_corr_knn_points
        output_dict['src_node_corr_knn_points'] = src_node_corr_knn_points
        #######
        output_dict['ref_node_corr_knn_colors'] = ref_node_corr_knn_colors
        output_dict['src_node_corr_knn_colors'] = src_node_corr_knn_colors
        output_dict['ref_node_corr_knn_masks'] = ref_node_corr_knn_masks
        output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks
            # 8. Optimal transport
        matching_scores = torch.einsum('bnd,bmd->bnm', ref_node_corr_knn_feats, src_node_corr_knn_feats)  # (P, K, K)
        matching_scores = matching_scores / feats_f.shape[1] ** 0.5
        matching_scores = self.optimal_transport(matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks)

        output_dict['matching_scores'] = matching_scores

        # 9. Generate final correspondences during testing
        with torch.no_grad():
            #if not self.fine_matching.use_dustbin: 
            matching_scores = matching_scores[:, :-1, :-1]

            ref_corr_points, src_corr_points, corr_scores, estimated_transform, ref_corr_colors,src_corr_colors = self.fine_matching(
                ref_node_corr_knn_points,
                src_node_corr_knn_points,
                ref_node_corr_knn_masks,
                src_node_corr_knn_masks,
                matching_scores,
                node_corr_scores,
                ref_node_corr_knn_colors,
                src_node_corr_knn_colors
            )
        output_dict['ref_corr_points'] = ref_corr_points
        output_dict['src_corr_points'] = src_corr_points
        output_dict['ref_corr_colors'] = ref_corr_colors
        output_dict['src_corr_colors'] = src_corr_colors
        output_dict['corr_scores'] = corr_scores
        output_dict['estimated_transform'] = estimated_transform
 
        return output_dict

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark',default='3DMatch',choices=['3DMatch', '3DLoMatch', 'val'], help='test benchmark')
    return parser
class Tester(SingleTester):
    def __init__(self,cfg):
        super().__init__(cfg, parser=make_parser())

        start_time = time.time()
        data_loader, neighbor_limits = test_data_loader(cfg, self.args.benchmark)
        loading_time = time.time() - start_time
        message = f'Data loader created: {loading_time:.3f}s collapsed.'
        self.logger.info(message)
        message = f'Calibrate neighbors: {neighbor_limits}.'
        self.logger.info(message)
        self.register_loader(data_loader)

        # model
        model = mymodel(cfg).cuda()
        self.register_model(model)

        # evaluator
        self.evaluator = Evaluator(cfg).cuda()

        # preparation
        self.output_dir = osp.join(cfg.feature_dir, self.args.benchmark)
        ensure_dir(self.output_dir)

    def test_step(self, iteration, data_dict):
        output_dict = self.model(data_dict)
        return output_dict

    def eval_step(self, iteration, data_dict, output_dict):
        result_dict = self.evaluator(output_dict, data_dict)
        return result_dict

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        scene_name = data_dict['scene_name']
        ref_frame = data_dict['ref_frame']
        src_frame = data_dict['src_frame']
        message = f'{scene_name}, id0: {ref_frame}, id1: {src_frame}'
        message += ', ' + get_log_string(result_dict=result_dict)
        message += ', nCorr: {}'.format(output_dict['corr_scores'].shape[0])
        return message

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        scene_name = data_dict['scene_name']
        ref_id = data_dict['ref_frame']
        src_id = data_dict['src_frame']

        ensure_dir(osp.join(self.output_dir, scene_name))
        file_name = osp.join(self.output_dir, scene_name, f'{ref_id}_{src_id}.npz')
        np.savez_compressed(
            file_name,
            ref_points=release_cuda(output_dict['ref_points']),
            src_points=release_cuda(output_dict['src_points']),
            ref_points_f=release_cuda(output_dict['ref_points_f']),
            src_points_f=release_cuda(output_dict['src_points_f']),
            ref_points_c=release_cuda(output_dict['ref_points_c']),
            src_points_c=release_cuda(output_dict['src_points_c']),
            ref_feats_c=release_cuda(output_dict['ref_feats_c']),
            src_feats_c=release_cuda(output_dict['src_feats_c']),
            ref_node_corr_indices=release_cuda(output_dict['ref_node_corr_indices']),
            src_node_corr_indices=release_cuda(output_dict['src_node_corr_indices']),
            ref_corr_points=release_cuda(output_dict['ref_corr_points']),
            src_corr_points=release_cuda(output_dict['src_corr_points']),
            corr_scores=release_cuda(output_dict['corr_scores']),
            gt_node_corr_indices=release_cuda(output_dict['gt_node_corr_indices']),
            gt_node_corr_overlaps=release_cuda(output_dict['gt_node_corr_overlaps']),
            estimated_transform=release_cuda(output_dict['estimated_transform']),
            transform=release_cuda(data_dict['transform']),
            overlap=data_dict['overlap'],
            ref_node_corr_knn_points = release_cuda(output_dict['ref_node_corr_knn_points']),  # 
            src_node_corr_knn_points = release_cuda(output_dict['src_node_corr_knn_points']), 
            ref_corr_colors=release_cuda(output_dict['ref_corr_colors']),
            src_corr_colors=release_cuda(output_dict['src_corr_colors']),
            ref_node_corr_knn_masks = release_cuda(output_dict['ref_node_corr_knn_masks']),
            src_node_corr_knn_masks = release_cuda(output_dict['src_node_corr_knn_masks']) ,
            matching_scores = release_cuda(output_dict['matching_scores']) ,
            node_corr_scores = release_cuda(output_dict['node_corr_scores'])

        )
