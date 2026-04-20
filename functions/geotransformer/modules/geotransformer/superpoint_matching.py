import torch
import torch.nn as nn

from geotransformer.modules.ops import pairwise_distance

##############################################################1
class SuperPointMatching(nn.Module):
    def __init__(self, num_correspondences, ratio_threshold=0.8):
 
        super(SuperPointMatching, self).__init__()
        self.num_correspondences = num_correspondences
        self.ratio_threshold = ratio_threshold

    def forward(self, ref_feats, src_feats, ref_masks=None, src_masks=None):
        
        device = ref_feats.device
        
        if ref_masks is None:
            ref_masks = torch.ones(ref_feats.shape[0], dtype=torch.bool, device=device)
        if src_masks is None:
            src_masks = torch.ones(src_feats.shape[0], dtype=torch.bool, device=device)

        ref_valid_idx = torch.nonzero(ref_masks, as_tuple=True)[0]
        src_valid_idx = torch.nonzero(src_masks, as_tuple=True)[0]
        ref_feats_valid = ref_feats[ref_valid_idx]
        src_feats_valid = src_feats[src_valid_idx]

        dists = pairwise_distance(ref_feats_valid, src_feats_valid, normalized=True)  
        vals, indices = torch.topk(dists, k=2, largest=False)  
        
        ratio = vals[:, 0] / (vals[:, 1] + 1e-8)
        valid_ratio = ratio < self.ratio_threshold  


        best_src_idx = indices[:, 0]

        dists_trans = dists.transpose(0, 1)  
        _, best_ref_idx = torch.min(dists_trans, dim=1)  

      
        mutual_nn = best_ref_idx[best_src_idx] == torch.arange(dists.shape[0], device=device)

        valid_matches = valid_ratio & mutual_nn


        ref_matches = torch.arange(dists.shape[0], device=device)[valid_matches]
        src_matches = best_src_idx[valid_matches]
   
        match_scores = 1.0 / (vals[valid_matches, 0] + 1e-8)

        num_corr = min(self.num_correspondences, match_scores.numel())
        if num_corr > 0:
            match_scores, topk_idx = torch.topk(match_scores, k=num_corr)
            ref_corr_indices = ref_valid_idx[ref_matches[topk_idx]]
            src_corr_indices = src_valid_idx[src_matches[topk_idx]]
        else:

            ref_corr_indices = torch.empty((0,), dtype=torch.long, device=device)
            src_corr_indices = torch.empty((0,), dtype=torch.long, device=device)
            match_scores = torch.empty((0,), device=device)

        return ref_corr_indices, src_corr_indices, match_scores


