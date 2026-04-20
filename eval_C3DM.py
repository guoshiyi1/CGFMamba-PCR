import argparse
import os.path as osp
import time
import glob
import sys
import json

import torch
import numpy as np
import torch.nn as nn
from geotransformer.engine import Logger
from geotransformer.modules.registration import weighted_procrustes
from geotransformer.utils.summary_board import SummaryBoard
from geotransformer.utils.open3d import registration_with_ransac_from_correspondences
from geotransformer.modules.ops import pairwise_distance
from geotransformer.utils.registration import (
    evaluate_sparse_correspondences,
    evaluate_correspondences,
    compute_registration_error,
)
from geotransformer.modules.geotransformer import LocalGlobalRegistration
from geotransformer.datasets.registration.threedmatch.utils import (
    get_num_fragments,
    get_scene_abbr,
    get_gt_logs_and_infos,
    compute_transform_error,
    write_log_file,
)
from geotransformer.modules.transformer import SinusoidalPositionalEmbedding
class GeometricStructureEmbedding(nn.Module):
    def __init__(self, hidden_dim, sigma_d, sigma_a, angle_k, reduction_a='max'):
        super(GeometricStructureEmbedding, self).__init__()
        self.sigma_d = sigma_d
        self.sigma_a = sigma_a
        self.factor_a = 180.0 / (self.sigma_a * np.pi)
        self.angle_k = angle_k

        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, hidden_dim)
        self.proj_a = nn.Linear(hidden_dim, hidden_dim)

        self.reduction_a = reduction_a
        if self.reduction_a not in ['max', 'mean']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction_a}.')

    @torch.no_grad()
    def get_embedding_indices(self, points):

        batch_size, num_point, _ = points.shape

        dist_map = torch.sqrt(pairwise_distance(points, points))  # (B, N, N)
        d_indices = dist_map / self.sigma_d

        k = self.angle_k
        knn_indices = dist_map.topk(k=k + 1, dim=2, largest=False)[1][:, :, 1:]  # (B, N, k)
        knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_point, k, 3)  # (B, N, k, 3)
        expanded_points = points.unsqueeze(1).expand(batch_size, num_point, num_point, 3)  # (B, N, N, 3)
        knn_points = torch.gather(expanded_points, dim=2, index=knn_indices)  # (B, N, k, 3)
        ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, k, 3)
        anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)  # (B, N, N, 3)
        ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, N, N, k)
        cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (B, N, N, k)
        angles = torch.atan2(sin_values, cos_values)  # (B, N, N, k)
        a_indices = angles * self.factor_a

        return d_indices, a_indices

    def forward(self, points):
        d_indices, a_indices = self.get_embedding_indices(points)

        d_embeddings = self.embedding(d_indices)
        d_embeddings = self.proj_d(d_embeddings)

        a_embeddings = self.embedding(a_indices)
        a_embeddings = self.proj_a(a_embeddings)
        if self.reduction_a == 'max':
            a_embeddings = a_embeddings.max(dim=3)[0]
        else:
            a_embeddings = a_embeddings.mean(dim=3)

        embeddings = d_embeddings + a_embeddings

        return embeddings
def create_transformation_matrix(R, T):

    transformation_matrix = np.eye(4)  
    transformation_matrix[:3, :3] = R  
    transformation_matrix[:3, 3] = T  
    return transformation_matrix
def compute_cost(points_src, points_tgt, R, T):
 
 
    transformed_src = (R @ points_src.T + T.reshape(-1, 1)).T 
    residuals = np.abs(points_tgt - transformed_src)  
    return np.sum(residuals)  

def get_upper_bound(a, lai):

    U = np.inf  
    for t1 in a:
        U_temp = np.sum(np.minimum(np.abs(a - t1), lai)) 
        U = min(U, U_temp) 
    return U

def get_lower_bound(a, lai):
  
    return np.sum(np.abs(a))  

def branch_and_bound_tears(points_src, points_tgt, max_iterations):
   
    best_R = None
    best_T = None
    best_cost = np.inf
    
    R = np.eye(3)  
    T = np.zeros(3) 

    for iteration in range(max_iterations):
        U_star = get_upper_bound(points_src, T) 
        L_B = get_lower_bound(points_src, T)  
        Q = [(L_B, [0, 2 * np.pi])] 

        while Q:
            L_B, branch = Q.pop(0)  

            if U_star - L_B < 1e-6:  
                break


            mid = (branch[0] + branch[1]) / 2
            new_branches = [(branch[0], mid), (mid, branch[1])]  
            for new_branch in new_branches:
                new_L_B = get_lower_bound(points_src, T)  
                if new_L_B >= U_star:
                    continue 
                U = get_upper_bound(points_src, T) 
                if U < U_star:
                    U_star = U  
                    best_R = R  
                    best_T = T 
                Q.append((new_L_B, new_branch))  

    return create_transformation_matrix(best_R, best_T)


def transform_(pts, trans):

    if len(pts.shape) == 3:
        trans_pts = trans[:, :3, :3] @ pts.permute(0,2,1) + trans[:, :3, 3:4]
        return trans_pts.permute(0,2,1)
    else:
        trans_pts = trans[:3, :3] @ pts.T + trans[:3, 3:4]
        return trans_pts.T
from config import make_cfg


class Matcher():
    def __init__(self,
                 inlier_threshold=0.10,
                 num_node='all',
                 use_mutual=True,
                 d_thre=0.1,
                 num_iterations=10,
                 ratio=0.2,
                 nms_radius=0.1,
                 max_points=8000,
                 k1=30,
                 k2=20,
                 select_scene=None,
                 ):
        self.inlier_threshold = inlier_threshold
        self.num_node = num_node
        self.use_mutual = use_mutual
        self.d_thre = d_thre
        self.num_iterations = num_iterations  
        self.ratio = ratio 
        self.max_points = max_points
        self.nms_radius = nms_radius
        self.k1 = k1
        self.k2 = k2

    def pick_seeds(self, dists, scores, R, max_num):

        assert scores.shape[0] == 1

        # parallel Non Maximum Suppression (more efficient)
        score_relation = scores.T >= scores  # [num_corr, num_corr], save the relation of leading_eig
        # score_relation[dists[0] >= R] = 1  # mask out the non-neighborhood node
        score_relation = score_relation.bool() | (dists[0] >= R).bool()
        is_local_max = score_relation.min(-1)[0].float()

        score_local_max = scores * is_local_max
        sorted_score = torch.argsort(score_local_max, dim=1, descending=True)

        # max_num = scores.shape[1]

        return_idx = sorted_score[:, 0: max_num].detach()

        return return_idx

    def cal_seed_trans(self, seeds, SC2_measure, src_keypts, tgt_keypts):

        bs, num_corr, num_channels = SC2_measure.shape[0], SC2_measure.shape[1], SC2_measure.shape[2]
        k1 = self.k1
        k2 = self.k2
        
        if k1 > num_channels:
            k1 = 4
            k2 = 4

        #################################
        # The first stage consensus set sampling
        # Finding the k1 nearest neighbors around each seed
        #################################
        sorted_score = torch.argsort(SC2_measure, dim=2, descending=True)
        knn_idx = sorted_score[:, :, 0: k1]
        sorted_value, _ = torch.sort(SC2_measure, dim=2, descending=True)
        idx_tmp = knn_idx.contiguous().view([bs, -1])
        idx_tmp = idx_tmp[:, :, None]
        idx_tmp = idx_tmp.expand(-1, -1, 3)

        #################################
        # construct the local SC2 measure of each consensus subset obtained in the first stage.
        #################################
        src_knn = src_keypts.gather(dim=1, index=idx_tmp).view([bs, -1, k1, 3]) 
        tgt_knn = tgt_keypts.gather(dim=1, index=idx_tmp).view([bs, -1, k1, 3])
        src_dist = ((src_knn[:, :, :, None, :] - src_knn[:, :, None, :, :]) ** 2).sum(-1) ** 0.5
        tgt_dist = ((tgt_knn[:, :, :, None, :] - tgt_knn[:, :, None, :, :]) ** 2).sum(-1) ** 0.5
        cross_dist = torch.abs(src_dist - tgt_dist)
        local_hard_SC_measure = (cross_dist < self.d_thre).float()
        local_SC2_measure = torch.matmul(local_hard_SC_measure[:, :, :1, :], local_hard_SC_measure)

        #################################
        # perform second stage consensus set sampling
        #################################
        sorted_score = torch.argsort(local_SC2_measure, dim=3, descending=True)
        knn_idx_fine = sorted_score[:, :, :, 0: k2]

        #################################
        # construct the soft SC2 matrix of the consensus set
        #################################
        num = knn_idx_fine.shape[1]
        knn_idx_fine = knn_idx_fine.contiguous().view([bs, num, -1])[:, :, :, None]
        knn_idx_fine = knn_idx_fine.expand(-1, -1, -1, 3)
        src_knn_fine = src_knn.gather(dim=2, index=knn_idx_fine).view([bs, -1, k2, 3])  
        tgt_knn_fine = tgt_knn.gather(dim=2, index=knn_idx_fine).view([bs, -1, k2, 3])

        src_dist = ((src_knn_fine[:, :, :, None, :] - src_knn_fine[:, :, None, :, :]) ** 2).sum(-1) ** 0.5
        tgt_dist = ((tgt_knn_fine[:, :, :, None, :] - tgt_knn_fine[:, :, None, :, :]) ** 2).sum(-1) ** 0.5
        cross_dist = torch.abs(src_dist - tgt_dist)
        local_hard_measure = (cross_dist < self.d_thre * 2).float()
        local_SC2_measure = torch.matmul(local_hard_measure, local_hard_measure) / k2
        local_SC_measure = torch.clamp(1 - cross_dist ** 2 / self.d_thre ** 2, min=0)
        # local_SC2_measure = local_SC_measure * local_SC2_measure
        local_SC2_measure = local_SC_measure
        local_SC2_measure = local_SC2_measure.view([-1, k2, k2])


        #################################
        # Power iteratation to get the inlier probability
        #################################
        local_SC2_measure[:, torch.arange(local_SC2_measure.shape[1]), torch.arange(local_SC2_measure.shape[1])] = 0
        total_weight = self.cal_leading_eigenvector(local_SC2_measure, method='power')
        total_weight = total_weight.view([bs, -1, k2])
        total_weight = total_weight / (torch.sum(total_weight, dim=-1, keepdim=True) + 1e-6)

        #################################
        # calculate the transformation by weighted least-squares for each subsets in parallel
        #################################
        total_weight = total_weight.view([-1, k2])
        src_knn = src_knn_fine
        tgt_knn = tgt_knn_fine
        src_knn, tgt_knn = src_knn.view([-1, k2, 3]), tgt_knn.view([-1, k2, 3])

        #################################
        # compute the rigid transformation for each seed by the weighted SVD
        #################################
        seedwise_trans = rigid_transform_3d(src_knn, tgt_knn, total_weight)
        seedwise_trans = seedwise_trans.view([bs, -1, 4, 4])

        #################################
        # calculate the inlier number for each hypothesis, and find the best transformation for each point cloud pair
        #################################
        pred_position = torch.einsum('bsnm,bmk->bsnk', seedwise_trans[:, :, :3, :3],
                                     src_keypts.permute(0, 2, 1)) + seedwise_trans[:, :, :3,
                                                                    3:4]  # [bs, num_seeds, num_corr, 3]
        #################################
        # calculate the inlier number for each hypothesis, and find the best transformation for each point cloud pair
        #################################
        pred_position = pred_position.permute(0, 1, 3, 2)
        L2_dis = torch.norm(pred_position - tgt_keypts[:, None, :, :], dim=-1) 
        seedwise_fitness = torch.sum((L2_dis < self.inlier_threshold).float(), dim=-1) 
        batch_best_guess = seedwise_fitness.argmax(dim=1)
        best_guess_ratio = seedwise_fitness[0, batch_best_guess]
        final_trans = seedwise_trans.gather(dim=1,index=batch_best_guess[:, None, None, None].expand(-1, -1, 4, 4)).squeeze(1)

        return final_trans

    def cal_leading_eigenvector(self, M, method='power'):

        if method == 'power':
            # power iteration algorithm
            leading_eig = torch.ones_like(M[:, :, 0:1])
            leading_eig_last = leading_eig
            for i in range(self.num_iterations):
                leading_eig = torch.bmm(M, leading_eig)
                leading_eig = leading_eig / (torch.norm(leading_eig, dim=1, keepdim=True) + 1e-6)
                if torch.allclose(leading_eig, leading_eig_last):
                    break
                leading_eig_last = leading_eig
            leading_eig = leading_eig.squeeze(-1)
            return leading_eig
        elif method == 'eig':  # cause NaN during back-prop
            e, v = torch.symeig(M, eigenvectors=True)
            leading_eig = v[:, :, -1]
            return leading_eig
        else:
            exit(-1)

    def cal_confidence(self, M, leading_eig, method='eig_value'):

        if method == 'eig_value':
            # max eigenvalue as the confidence (Rayleigh quotient)
            max_eig_value = (leading_eig[:, None, :] @ M @ leading_eig[:, :, None]) / (
                        leading_eig[:, None, :] @ leading_eig[:, :, None])
            confidence = max_eig_value.squeeze(-1)
            return confidence
        elif method == 'eig_value_ratio':
            # max eigenvalue / second max eigenvalue as the confidence
            max_eig_value = (leading_eig[:, None, :] @ M @ leading_eig[:, :, None]) / (
                        leading_eig[:, None, :] @ leading_eig[:, :, None])
            # compute the second largest eigen-value
            B = M - max_eig_value * leading_eig[:, :, None] @ leading_eig[:, None, :]
            solution = torch.ones_like(B[:, :, 0:1])
            for i in range(self.num_iterations):
                solution = torch.bmm(B, solution)
                solution = solution / (torch.norm(solution, dim=1, keepdim=True) + 1e-6)
            solution = solution.squeeze(-1)
            second_eig = solution
            second_eig_value = (second_eig[:, None, :] @ B @ second_eig[:, :, None]) / (
                        second_eig[:, None, :] @ second_eig[:, :, None])
            confidence = max_eig_value / second_eig_value
            return confidence
        elif method == 'xMx':
            # max xMx as the confidence (x is the binary solution)
            # rank = torch.argsort(leading_eig, dim=1, descending=True)[:, 0:int(M.shape[1]*self.ratio)]
            # binary_sol = torch.zeros_like(leading_eig)
            # binary_sol[0, rank[0]] = 1
            confidence = leading_eig[:, None, :] @ M @ leading_eig[:, :, None]
            confidence = confidence.squeeze(-1) / M.shape[1]
            return confidence
    
    def post_refinement(self, initial_trans, src_keypts, tgt_keypts, it_num, weights=None):

        assert initial_trans.shape[0] == 1
        inlier_threshold = 1.2

        # inlier_threshold_list = [self.inlier_threshold] * it_num

        if self.inlier_threshold == 0.10:  # for 3DMatch
            inlier_threshold_list = [0.10] * it_num
        else:  # for KITTI
            inlier_threshold_list = [1.2] * it_num

        previous_inlier_num = 0
        for inlier_threshold in inlier_threshold_list:
            warped_src_keypts = transform_(src_keypts, initial_trans)

            L2_dis = torch.norm(warped_src_keypts - tgt_keypts, dim=-1)
            pred_inlier = (L2_dis < inlier_threshold)[0]  # assume bs = 1
            inlier_num = torch.sum(pred_inlier)
            if abs(int(inlier_num - previous_inlier_num)) < 1:
                break
            else:
                previous_inlier_num = inlier_num
            initial_trans = rigid_transform_3d(
                A=src_keypts[:, pred_inlier, :],
                B=tgt_keypts[:, pred_inlier, :],
                
                weights=1 / (1 + (L2_dis / inlier_threshold) ** 2)[:, pred_inlier],
              
            )
        return initial_trans

    def match_pair(self, src_keypts, tgt_keypts, src_features, tgt_features):
        N_src = src_features.shape[1]
        N_tgt = tgt_features.shape[1]
        # use all point or sample points.
        if self.num_node == 'all':
            src_sel_ind = np.arange(N_src)
            tgt_sel_ind = np.arange(N_tgt)
        else:
            src_sel_ind = np.random.choice(N_src, self.num_node)
            tgt_sel_ind = np.random.choice(N_tgt, self.num_node)
        src_desc = src_features[:, src_sel_ind, :]
        tgt_desc = tgt_features[:, tgt_sel_ind, :]
        src_keypts = src_keypts[:, src_sel_ind, :]
        tgt_keypts = tgt_keypts[:, tgt_sel_ind, :]

        # match points in feature space.
        distance = torch.sqrt(2 - 2 * (src_desc[0] @ tgt_desc[0].T) + 1e-6)
        distance = distance.unsqueeze(0)
        source_idx = torch.argmin(distance[0], dim=1)
        if torch.cuda.is_available():
            device = torch.device("cuda")
            #print("CUDA device is available. Using GPU.")
        else:
            device = torch.device("cpu")
            #print("CUDA device not available. Using CPU.")

        corr = torch.cat([torch.arange(source_idx.shape[0])[:, None].to(device), source_idx[:, None]], dim=-1)

        # generate correspondences
        src_keypts_corr = src_keypts[:, corr[:, 0]]
        tgt_keypts_corr = tgt_keypts[:, corr[:, 1]]

        return src_keypts_corr, tgt_keypts_corr


    def estimator(self, src_keypts, tgt_keypts, src_features, tgt_features):
    
        #################################
        # generate coarse correspondences
        #################################
        src_keypts_corr, tgt_keypts_corr = self.match_pair(src_keypts, tgt_keypts, src_features, tgt_features)

        #################################
        # use the proposed SC2-PCR to estimate the rigid transformation
        #################################
        pred_trans = self.SC2_PCR(src_keypts_corr, tgt_keypts_corr,None,False)

        frag1_warp = transform_(src_keypts_corr, pred_trans)
        distance = torch.sum((frag1_warp - tgt_keypts_corr) ** 2, dim=-1) ** 0.5
        pred_labels = (distance < self.inlier_threshold).float()

        return pred_trans, pred_labels, src_keypts_corr, tgt_keypts_corr

    def SC2_PCR_color(self, src_keypts, tgt_keypts, src_hsv=None, tgt_hsv=None, corr_score=None, use_corrscore=False):

        bs, num_corr = src_keypts.shape[0], tgt_keypts.shape[1]
        has_color = (src_hsv is not None and tgt_hsv is not None)

     
        if num_corr > self.max_points:
            indices = torch.arange(self.max_points)
            src_keypts = src_keypts[:, indices, :]
            tgt_keypts = tgt_keypts[:, indices, :]
            if has_color:
                src_hsv = src_hsv[:, indices, :]
                tgt_hsv = tgt_hsv[:, indices, :]
            if use_corrscore and corr_score is not None:
                corr_score = corr_score[:self.max_points]
            num_corr = self.max_points

        src_dist = torch.norm((src_keypts[:, :, None, :] - src_keypts[:, None, :, :]), dim=-1)
        target_dist = torch.norm((tgt_keypts[:, :, None, :] - tgt_keypts[:, None, :, :]), dim=-1)
        cross_dist = torch.abs(src_dist - target_dist)

     
        if has_color:
            h_diff_matrix = torch.abs(src_hsv[:, :, None, 0] - src_hsv[:, None, :, 0])
            h_diff_matrix = torch.min(h_diff_matrix, 1.0 - h_diff_matrix)  
            
            h_diff_tgt = torch.abs(tgt_hsv[:, :, None, 0] - tgt_hsv[:, None, :, 0])
            h_diff_tgt = torch.min(h_diff_tgt, 1.0 - h_diff_tgt)
            
            s_diff_matrix = torch.abs(src_hsv[:, :, None, 1] - src_hsv[:, None, :, 1])
            s_diff_tgt = torch.abs(tgt_hsv[:, :, None, 1] - tgt_hsv[:, None, :, 1])
            
            v_diff_matrix = torch.abs(src_hsv[:, :, None, 2] - src_hsv[:, None, :, 2])
            v_diff_tgt = torch.abs(tgt_hsv[:, :, None, 2] - tgt_hsv[:, None, :, 2])
            

            h_diff_cross = torch.abs(h_diff_matrix - h_diff_tgt)
            h_diff_cross = torch.min(h_diff_cross, 1.0 - h_diff_cross)
        
            h_consistency = torch.clamp(1.0 - h_diff_cross ** 2 / 0.1 ** 2, min=0)
            color_consistency = h_consistency 

            color_mask = (h_diff_cross < 0.5) 

        SC_dist_thre = self.d_thre
        SC_measure = torch.clamp(1.0 - cross_dist ** 2 / SC_dist_thre ** 2, min=0)
        hard_SC_measure = (cross_dist < SC_dist_thre).float()

        if has_color:
            SC_measure = SC_measure * (0.9 + 0.1 * color_consistency) 
            hard_SC_measure = hard_SC_measure * color_mask.float()


        confidence = self.cal_leading_eigenvector(SC_measure, method='power')
        if use_corrscore and corr_score is not None:
            if confidence.shape[1] == corr_score.shape[0]:
                confidence = confidence * corr_score
        seeds = self.pick_seeds(src_dist, confidence, R=self.nms_radius, max_num=int(num_corr * self.ratio))

        SC2_dist_thre = self.d_thre / 2
        hard_SC_measure_tight = (cross_dist < SC2_dist_thre).float()
        
        if has_color:
            tight_color_mask = (h_diff_cross < 0.6)
            hard_SC_measure_tight = hard_SC_measure_tight * tight_color_mask.float()
        
        seed_hard_SC_measure = hard_SC_measure.gather(dim=1,
                                index=seeds[:, :, None].expand(-1, -1, num_corr))
        seed_hard_SC_measure_tight = hard_SC_measure_tight.gather(dim=1,
                                index=seeds[:, :, None].expand(-1, -1, num_corr))
        SC2_measure = torch.matmul(seed_hard_SC_measure_tight, hard_SC_measure_tight) * seed_hard_SC_measure

   
        final_trans = self.cal_seed_trans(seeds, SC2_measure, src_keypts, tgt_keypts)                       
        final_trans = self.post_refinement(final_trans, src_keypts, tgt_keypts, 20)                               
        return final_trans




def rigid_transform_3d(A, B, weights=None, weight_threshold=0):

    bs = A.shape[0]
    if weights is None:
        weights = torch.ones_like(A[:, :, 0])
    weights[weights < weight_threshold] = 0
    # weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-6)

    # find mean of point cloud
    centroid_A = torch.sum(A * weights[:, :, None], dim=1, keepdim=True) / (torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)
    centroid_B = torch.sum(B * weights[:, :, None], dim=1, keepdim=True) / (torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    # construct weight covariance matrix
    Weight = torch.diag_embed(weights)
    H = Am.permute(0, 2, 1) @ Weight @ Bm

    # find rotation
    U, S, Vt = torch.svd(H.cpu())
    U, S, Vt = U.to(weights.device), S.to(weights.device), Vt.to(weights.device)
    delta_UV = torch.det(Vt @ U.permute(0, 2, 1))
    eye = torch.eye(3)[None, :, :].repeat(bs, 1, 1).to(A.device)
    eye[:, -1, -1] = delta_UV
    R = Vt @ eye @ U.permute(0, 2, 1)
    t = centroid_B.permute(0,2,1) - R @ centroid_A.permute(0,2,1)
    # warp_A = transform(A, integrate_trans(R,t))
    # RMSE = torch.sum( (warp_A - B) ** 2, dim=-1).mean()
    return integrate_trans(R, t)

def integrate_trans(R, t):

    if len(R.shape) == 3:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4)[None].repeat(R.shape[0], 1, 1).to(R.device)
        else:
            trans = np.eye(4)[None]
        trans[:, :3, :3] = R
        trans[:, :3, 3:4] = t.view([-1, 3, 1])
    else:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4).to(R.device)
        else:
            trans = np.eye(4)
        trans[:3, :3] = R
        trans[:3, 3:4] = t
    return trans

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_epoch', default=None, type=int, help='test epoch')
    parser.add_argument('--benchmark', choices=['3DMatch', '3DLoMatch'], required=True, help='test benchmark')
    parser.add_argument('--method', choices=['lgr', 'ransac', 'svd','csc2'], required=True, help='registration method')
    parser.add_argument('--num_corr', type=int, default=None, help='number of correspondences for registration')
    parser.add_argument('--verbose', action='store_true', help='verbose mode')
    return parser


def eval_one_epoch(args, cfg, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    usestimator = False
    use_corrscore = False
    use_batchscore = False 
    use_localsc = True 

    lgr = LocalGlobalRegistration(k=3,acceptance_radius=0.1,
                                  mutual=True,
                                  confidence_threshold=0.05,
                                  use_dustbin=False,
                                  use_global_score=True,
                                #   use_batchscore=use_batchscore,
                                #   use_localsc = use_localsc,
                                  correspondence_threshold=3,
                                  correspondence_limit=args.num_corr,
                                  num_refinement_steps=10
                                  )
    matcher = Matcher(inlier_threshold=0.1,
                 num_node=1000,
                 use_mutual=False,
                 d_thre=0.1,
                 num_iterations=10,
                 ratio=0.2,
                 nms_radius=0.1,
                 max_points=8000,
                 k1=400,
                 k2=10,       
                )

    features_root = osp.join(cfg.feature_dir, args.benchmark)
    benchmark = args.benchmark

    coarse_matching_meter = SummaryBoard()
    coarse_matching_meter.register_meter('precision')
    coarse_matching_meter.register_meter('PMR>0')
    coarse_matching_meter.register_meter('PMR>=0.1')
    coarse_matching_meter.register_meter('PMR>=0.3')
    coarse_matching_meter.register_meter('PMR>=0.5')
    coarse_matching_meter.register_meter('scene_precision')
    coarse_matching_meter.register_meter('scene_PMR>0')
    coarse_matching_meter.register_meter('scene_PMR>=0.1')
    coarse_matching_meter.register_meter('scene_PMR>=0.3')
    coarse_matching_meter.register_meter('scene_PMR>=0.5')

    fine_matching_meter = SummaryBoard()
    fine_matching_meter.register_meter('rmse_recall')
    fine_matching_meter.register_meter('to_recall')
    fine_matching_meter.register_meter('inlier_ratio')
    fine_matching_meter.register_meter('overlap')
    fine_matching_meter.register_meter('scene_rmse_recall')
    fine_matching_meter.register_meter('scene_to_recall')
    fine_matching_meter.register_meter('scene_inlier_ratio')
    fine_matching_meter.register_meter('scene_overlap')

    registration_meter = SummaryBoard()
    registration_meter.register_meter('rmse_recall')
    registration_meter.register_meter('to_recall')
    registration_meter.register_meter('mean_rre')
    registration_meter.register_meter('mean_rte')
    registration_meter.register_meter('median_rre')
    registration_meter.register_meter('median_rte')
    registration_meter.register_meter('scene_rmse_recall')
    registration_meter.register_meter('scene_to_recall')
    registration_meter.register_meter('scene_rre')
    registration_meter.register_meter('scene_rte')

    scene_coarse_matching_result_dict = {}
    scene_fine_matching_result_dict = {}
    scene_registration_result_dict = {}

    scene_roots = sorted(glob.glob(osp.join(features_root, '*')))
    for scene_root in scene_roots:
        coarse_matching_meter.reset_meter('scene_precision')
        coarse_matching_meter.reset_meter('scene_PMR>0')
        coarse_matching_meter.reset_meter('scene_PMR>=0.1')
        coarse_matching_meter.reset_meter('scene_PMR>=0.3')
        coarse_matching_meter.reset_meter('scene_PMR>=0.5')

        fine_matching_meter.reset_meter('scene_rmse_recall')
        fine_matching_meter.reset_meter('scene_to_recall')
        fine_matching_meter.reset_meter('scene_inlier_ratio')
        fine_matching_meter.reset_meter('scene_overlap')

        registration_meter.reset_meter('scene_rmse_recall')
        registration_meter.reset_meter('scene_to_recall')
        registration_meter.reset_meter('scene_rre')
        registration_meter.reset_meter('scene_rte')

        scene_name = osp.basename(scene_root)
        scene_abbr = get_scene_abbr(scene_name)
        num_fragments = get_num_fragments(scene_name)
        gt_root = osp.join(cfg.data.dataset_root, 'metadata', 'benchmarks', benchmark, scene_name)
        gt_indices, gt_logs, gt_infos = get_gt_logs_and_infos(gt_root, num_fragments)

        estimated_transforms = []

        file_names = sorted(
            glob.glob(osp.join(scene_root, '*.npz')),
            key=lambda x: [int(i) for i in osp.basename(x).split('.')[0].split('_')],
        )
        for file_name in file_names:
            ref_frame, src_frame = [int(x) for x in osp.basename(file_name).split('.')[0].split('_')]

            data_dict = np.load(file_name)

            ref_points_c = data_dict['ref_points_c']
            src_points_c = data_dict['src_points_c']
            src_feats_c = data_dict['src_feats_c']
            ref_feats_c = data_dict['ref_feats_c']

            ref_node_corr_indices = data_dict['ref_node_corr_indices']
            src_node_corr_indices = data_dict['src_node_corr_indices']

            ref_corr_points = data_dict['ref_corr_points']
            src_corr_points = data_dict['src_corr_points']
            ref_corr_colors = data_dict['ref_corr_colors']
            src_corr_colors = data_dict['src_corr_colors']
            corr_scores = data_dict['corr_scores']

            gt_node_corr_indices = data_dict['gt_node_corr_indices']
            transform = data_dict['transform']
            pcd_overlap = data_dict['overlap']
            
            #add
            ref_node_corr_knn_points = torch.from_numpy(data_dict['ref_node_corr_knn_points']) # 
            src_node_corr_knn_points = torch.from_numpy(data_dict['src_node_corr_knn_points']) 
            ref_node_corr_knn_masks = torch.from_numpy(data_dict['ref_node_corr_knn_masks']) 
            src_node_corr_knn_masks = torch.from_numpy(data_dict['src_node_corr_knn_masks']) 
            matching_scores = torch.from_numpy(data_dict['matching_scores']) 
            matching_scores = matching_scores[:, :-1, :-1]
            node_corr_scores = torch.from_numpy(data_dict['node_corr_scores']) 


            if args.num_corr is not None and corr_scores.shape[0] > args.num_corr:
                sel_indices = np.argsort(-corr_scores)[: args.num_corr]
                ref_corr_points = ref_corr_points[sel_indices]
                src_corr_points = src_corr_points[sel_indices]
                corr_scores = corr_scores[sel_indices]
                src_corr_colors = src_corr_colors[sel_indices]
                ref_corr_colors = ref_corr_colors[sel_indices]

            message = '{}, id0: {}, id1: {}, OV: {:.3f}'.format(scene_abbr, ref_frame, src_frame, pcd_overlap)

            # 1. evaluate correspondences
            # 1.1 evaluate coarse correspondences
            coarse_matching_result_dict = evaluate_sparse_correspondences(
                ref_points_c, src_points_c, ref_node_corr_indices, src_node_corr_indices, gt_node_corr_indices
            )

            coarse_precision = coarse_matching_result_dict['precision']

            coarse_matching_meter.update('scene_precision', coarse_precision)
            coarse_matching_meter.update('scene_PMR>0', float(coarse_precision > 0))
            coarse_matching_meter.update('scene_PMR>=0.1', float(coarse_precision >= 0.1))
            coarse_matching_meter.update('scene_PMR>=0.3', float(coarse_precision >= 0.3))
            coarse_matching_meter.update('scene_PMR>=0.5', float(coarse_precision >= 0.5))

            # 1.2 evaluate fine correspondences
            fine_matching_result_dict = evaluate_correspondences(
                ref_corr_points, src_corr_points, transform, positive_radius=cfg.eval.acceptance_radius
            )

            inlier_ratio = fine_matching_result_dict['inlier_ratio']
            overlap = fine_matching_result_dict['overlap']

            fine_matching_meter.update('scene_inlier_ratio', inlier_ratio)
            fine_matching_meter.update('scene_overlap', overlap)
            fine_matching_meter.update('scene_rmse_recall', float(inlier_ratio >= cfg.eval.inlier_ratio_threshold))

            message += ', c_PIR: {:.3f}'.format(coarse_precision)
            message += ', f_IR: {:.3f}'.format(inlier_ratio)
            message += ', f_OV: {:.3f}'.format(overlap)
            message += ', f_RS: {:.3f}'.format(fine_matching_result_dict['residual'])
            message += ', f_NU: {}'.format(fine_matching_result_dict['num_corr'])

             # 2. evaluate registration
            if args.method == 'lgr' :
                if args.num_corr is None:
                    estimated_transform = data_dict['estimated_transform']
                else:
                    _,_,_,estimated_transform = lgr( ref_node_corr_knn_points.to('cuda:0'),
                src_node_corr_knn_points.to('cuda:0'),
                ref_node_corr_knn_masks.to('cuda:0'),
                src_node_corr_knn_masks.to('cuda:0'),
                matching_scores.to('cuda:0'),  
                node_corr_scores.to('cuda:0'), 
            )
                   
                if torch.is_tensor(estimated_transform):
                    if estimated_transform.is_cuda:
                        estimated_transform = estimated_transform.cpu()
                        estimated_transform = estimated_transform.numpy()
            
            elif args.method == 'ransac':
                estimated_transform = registration_with_ransac_from_correspondences(
                    src_corr_points,
                    ref_corr_points,
                    distance_threshold=cfg.ransac.distance_threshold,
                    ransac_n=cfg.ransac.num_points,
                    num_iterations=cfg.ransac.num_iterations,
                )
            elif args.method == 'svd':
                with torch.no_grad():
                    ref_corr_points = torch.from_numpy(ref_corr_points).cuda()
                    src_corr_points = torch.from_numpy(src_corr_points).cuda()
                    corr_scores = torch.from_numpy(corr_scores).cuda()
                    estimated_transform = weighted_procrustes(
                        src_corr_points, ref_corr_points, corr_scores, return_transform=True
                    )
                    estimated_transform = estimated_transform.detach().cpu().numpy()

            elif args.method == 'csc2':
                if usestimator==True:   
                    estimated_transform, pred_labels, src_keypts_corr, tgt_keypts_corr = matcher.estimator(torch.from_numpy(src_points_c).unsqueeze(0).cuda(), 
                                                                                                           torch.from_numpy(ref_points_c).unsqueeze(0).cuda(), 
                                                                                                           torch.from_numpy(src_feats_c).unsqueeze(0).cuda(),)
                                                                                                           
                    if torch.is_tensor(estimated_transform):
                        if estimated_transform.is_cuda:
                            estimated_transform = estimated_transform.cpu()
                            estimated_transform = estimated_transform.numpy()
                            estimated_transform = estimated_transform.squeeze()
                            estimated_transform = estimated_transform.astype(np.float32)
              
                elif usestimator==False:

                    estimated_transform = matcher.SC2_PCR_color(src_keypts=torch.from_numpy(src_corr_points).unsqueeze(0).to(device), 
                                                          tgt_keypts=torch.from_numpy(ref_corr_points).unsqueeze(0).to(device),
                                                          src_hsv=torch.from_numpy(src_corr_colors).unsqueeze(0).to(device),
                                                          tgt_hsv=torch.from_numpy(ref_corr_colors).unsqueeze(0).to(device),
                                                          corr_score=torch.from_numpy(corr_scores).to(device),use_corrscore=use_corrscore)
                    
                    if estimated_transform is None:
                        continue

                    estimated_transform = estimated_transform.cpu().numpy()
                    estimated_transform = estimated_transform.squeeze()
                    estimated_transform = estimated_transform.astype(np.float32)
            else:
                raise ValueError(f'Unsupported registration method: {args.method}.')

            estimated_transforms.append(
                dict(
                    test_pair=[ref_frame, src_frame],
                    num_fragments=num_fragments,
                    transform=estimated_transform,
                )
            )

            if gt_indices[ref_frame, src_frame] != -1:
                # evaluate transform (realignment error)
                gt_index = gt_indices[ref_frame, src_frame]
                transform = gt_logs[gt_index]['transform']
                covariance = gt_infos[gt_index]['covariance']
                error = compute_transform_error(transform, covariance, estimated_transform)
                message += ', r_RMSE: {:.3f}'.format(np.sqrt(error))
                accepted_rmse = error < cfg.eval.rmse_threshold ** 2
                accepted = error < cfg.eval.rmse_threshold ** 2
                rre, rte = compute_registration_error(transform, estimated_transform)
                rre_threshold = 15.0  
                rte_threshold = 0.3   
                accepted_rre_rte = (rre < rre_threshold) and (rte < rte_threshold)
                registration_meter.update('scene_rmse_recall', float(accepted_rmse))
                registration_meter.update('scene_to_recall', float(accepted_rre_rte))
                message += ', r_RRE: {:.3f}, r_RTE: {:.3f}'.format(rre, rte)
              
                if accepted_rre_rte:
                    message += ', Passed RRE/RTE threshold'

                registration_meter.update('scene_rmse_recall', float(accepted_rmse))
                registration_meter.update('scene_to_recall', float(accepted_rre_rte))

                if accepted_rmse:
                    rre, rte = compute_registration_error(transform, estimated_transform)
                    registration_meter.update('scene_rre', rre)
                    registration_meter.update('scene_rte', rte)
                    message += ', r_RRE: {:.3f}'.format(rre)
                    message += ', r_RTE: {:.3f}'.format(rte)

            if args.verbose:
                logger.info(message)

        est_log = osp.join(cfg.registration_dir, benchmark, scene_name, 'est.log')
        write_log_file(est_log, estimated_transforms)

        logger.info(f'Scene_name: {scene_name}')

        # 1. print correspondence evaluation results (one scene)
        # 1.1 coarse level statistics
        coarse_precision = coarse_matching_meter.mean('scene_precision')
        coarse_matching_recall_0 = coarse_matching_meter.mean('scene_PMR>0')
        coarse_matching_recall_1 = coarse_matching_meter.mean('scene_PMR>=0.1')
        coarse_matching_recall_3 = coarse_matching_meter.mean('scene_PMR>=0.3')
        coarse_matching_recall_5 = coarse_matching_meter.mean('scene_PMR>=0.5')
        coarse_matching_meter.update('precision', coarse_precision)
        coarse_matching_meter.update('PMR>0', coarse_matching_recall_0)
        coarse_matching_meter.update('PMR>=0.1', coarse_matching_recall_1)
        coarse_matching_meter.update('PMR>=0.3', coarse_matching_recall_3)
        coarse_matching_meter.update('PMR>=0.5', coarse_matching_recall_5)
        scene_coarse_matching_result_dict[scene_abbr] = {
            'precision': coarse_precision,
            'PMR>0': coarse_matching_recall_0,
            'PMR>=0.1': coarse_matching_recall_1,
            'PMR>=0.3': coarse_matching_recall_3,
            'PMR>=0.5': coarse_matching_recall_5,
        }

        # 1.2 fine level statistics
        recall = fine_matching_meter.mean('scene_rmse_recall')
        inlier_ratio = fine_matching_meter.mean('scene_inlier_ratio')
        overlap = fine_matching_meter.mean('scene_overlap')
        fine_matching_meter.update('rmse_recall', recall)
        fine_matching_meter.update('inlier_ratio', inlier_ratio)
        fine_matching_meter.update('overlap', overlap)
        scene_fine_matching_result_dict[scene_abbr] = {'recall': recall, 'inlier_ratio': inlier_ratio}

        message = '  Correspondence, '
        message += ', c_PIR: {:.3f}'.format(coarse_precision)
        message += ', c_PMR>0: {:.3f}'.format(coarse_matching_recall_0)
        message += ', c_PMR>=0.1: {:.3f}'.format(coarse_matching_recall_1)
        message += ', c_PMR>=0.3: {:.3f}'.format(coarse_matching_recall_3)
        message += ', c_PMR>=0.5: {:.3f}'.format(coarse_matching_recall_5)
        message += ', f_FMR: {:.3f}'.format(recall)
        message += ', f_IR: {:.3f}'.format(inlier_ratio)
        message += ', f_OV: {:.3f}'.format(overlap)
        logger.info(message)

        # 2. print registration evaluation results (one scene)
        recall_rmse = registration_meter.mean('scene_rmse_recall')
        recall_to = registration_meter.mean('scene_to_recall')

        mean_rre = registration_meter.mean('scene_rre')
        mean_rte = registration_meter.mean('scene_rte')
        median_rre = registration_meter.median('scene_rre')
        median_rte = registration_meter.median('scene_rte')
        registration_meter.update('rmse_recall', recall_rmse)
        registration_meter.update('to_recall', recall_to)
        registration_meter.update('mean_rre', mean_rre)
        registration_meter.update('mean_rte', mean_rte)
        registration_meter.update('median_rre', median_rre)
        registration_meter.update('median_rte', median_rte)

        scene_registration_result_dict[scene_abbr] = {
            'rmse_recall': recall_rmse,
            'to_recall': recall_rmse,
            'mean_rre': mean_rre,
            'mean_rte': mean_rte,
            'median_rre': median_rre,
            'median_rte': median_rte,
        }

        message = '  Registration'
        message += ', RR_rmse: {:.5f}'.format(recall_rmse)
        message += ', RR_to: {:.5f}'.format(recall_to)

        message += ', mean_RRE: {:.5f}'.format(mean_rre)
        message += ', mean_RTE: {:.5f}'.format(mean_rte)
        message += ', median_RRE: {:.5f}'.format(median_rre)
        message += ', median_RTE: {:.5f}'.format(median_rte)
        logger.info(message)

    if args.test_epoch is not None:
        logger.critical('Epoch {}'.format(args.test_epoch))

    # 1. print correspondence evaluation results
    message = '  Coarse Matching'
    message += ', PIR: {:.5f}'.format(coarse_matching_meter.mean('precision'))
    message += ', PMR>0: {:.3f}'.format(coarse_matching_meter.mean('PMR>0'))
    message += ', PMR>=0.1: {:.3f}'.format(coarse_matching_meter.mean('PMR>=0.1'))
    message += ', PMR>=0.3: {:.3f}'.format(coarse_matching_meter.mean('PMR>=0.3'))
    message += ', PMR>=0.5: {:.3f}'.format(coarse_matching_meter.mean('PMR>=0.5'))
    logger.critical(message)
    for scene_abbr, result_dict in scene_coarse_matching_result_dict.items():
        message = '    {}'.format(scene_abbr)
        message += ', PIR: {:.3f}'.format(result_dict['precision'])
        message += ', PMR>0: {:.3f}'.format(result_dict['PMR>0'])
        message += ', PMR>=0.1: {:.3f}'.format(result_dict['PMR>=0.1'])
        message += ', PMR>=0.3: {:.3f}'.format(result_dict['PMR>=0.3'])
        message += ', PMR>=0.5: {:.3f}'.format(result_dict['PMR>=0.5'])
        logger.critical(message)

    message = '  Fine Matching'
    message += ', FMR: {:.5f}'.format(fine_matching_meter.mean('rmse_recall'))
    message += ', FMR: {:.5f}'.format(fine_matching_meter.mean('to_recall'))
    message += ', IR: {:.5f}'.format(fine_matching_meter.mean('inlier_ratio'))
    message += ', OV: {:.5f}'.format(fine_matching_meter.mean('overlap'))
    message += ', std: {:.5f}'.format(fine_matching_meter.std('rmse_recall'))
    logger.critical(message)
    for scene_abbr, result_dict in scene_fine_matching_result_dict.items():
        message = '    {}'.format(scene_abbr)
        message += ', FMR: {:.5f}'.format(result_dict['recall'])
        message += ', IR: {:.5f}'.format(result_dict['inlier_ratio'])
        logger.critical(message)

    # 2. print registration evaluation results
    message = '  Registration'
    message += ', RR: {:.5f}'.format(registration_meter.mean('rmse_recall'))
    message += ', RR: {:.5f}'.format(registration_meter.mean('to_recall'))
    message += ', mean_RRE: {:.5f}'.format(registration_meter.mean('mean_rre'))
    message += ', mean_RTE: {:.5f}'.format(registration_meter.mean('mean_rte'))
    message += ', median_RRE: {:.3f}'.format(registration_meter.mean('median_rre'))
    message += ', median_RTE: {:.3f}'.format(registration_meter.mean('median_rte'))
    logger.critical(message)
    for scene_abbr, result_dict in scene_registration_result_dict.items():
        message = '    {}'.format(scene_abbr)
        message += ', RR_rmse: {:.5f}'.format(result_dict['rmse_recall'])
        message += ', RR_to: {:.5f}'.format(result_dict['to_recall'])
        message += ', mean_RRE: {:.5f}'.format(result_dict['mean_rre'])
        message += ', mean_RTE: {:.5f}'.format(result_dict['mean_rte'])
        message += ', median_RRE: {:.3f}'.format(result_dict['median_rre'])
        message += ', median_RTE: {:.3f}'.format(result_dict['median_rte'])
        logger.critical(message)


def main():
    parser = make_parser()
    args = parser.parse_args()

    cfg = make_cfg()
    log_file = osp.join(cfg.log_dir, 'eval-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
    logger = Logger(log_file=log_file)

    message = 'Command executed: ' + ' '.join(sys.argv)
    logger.info(message)
    message = 'Configs:\n' + json.dumps(cfg, indent=4)
    logger.info(message)

    eval_one_epoch(args, cfg, logger)


if __name__ == '__main__':
    main()
