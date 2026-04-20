import sys
from tools import builder
from sklearn.cluster import KMeans

def kmeans_H_sort(color_c, n_clusters=10):
    H = color_c[:, 0].unsqueeze(1).cpu().numpy()  # shape: [N, 1]
    kmeans = KMeans(n_clusters=n_clusters).fit(H)
    labels = kmeans.labels_                     # numpy [N]
    centers = kmeans.cluster_centers_.squeeze() # numpy [n_clusters]

    sorted_cluster_ids = torch.tensor(
        centers.argsort(), device=color_c.device
    )  # [n_clusters]

    cluster_labels = torch.tensor(labels, dtype=torch.long, device=color_c.device)

    cluster_order_map = torch.empty(n_clusters, device=color_c.device, dtype=torch.long)
    cluster_order_map[sorted_cluster_ids] = torch.arange(n_clusters, device=color_c.device)
    cluster_ranks = cluster_order_map[cluster_labels]  # [N]

    H_vals = color_c[:, 0]  # [N]
    sort_keys = cluster_ranks * 1e3 + H_vals  
    sorted_indices = torch.argsort(sort_keys)

    return sorted_indices

import torch
import torch.nn as nn
from geotransformer.modules.kpconv import ConvBlock, ResidualBlock, UnaryBlock, LastUnaryBlock, nearest_upsample

# 
class KPConvFPN(nn.Module):
    def __init__(self, input_dim, output_dim, init_dim, kernel_size, init_radius, init_sigma, group_norm,cfg):
        super(KPConvFPN, self).__init__()
        self.base_model1 = builder.model_builder(cfg.model1)

        self.encoder1_1 = ConvBlock(input_dim+3, init_dim, kernel_size, init_radius, init_sigma, group_norm)
        self.encoder1_2 = ResidualBlock(init_dim, init_dim*2, kernel_size, init_radius, init_sigma, group_norm)

        self.encoder2_1 = ResidualBlock(
            init_dim * 2, init_dim * 2, kernel_size, init_radius, init_sigma, group_norm, strided=True
        )
        self.encoder2_2 = ResidualBlock(
            init_dim * 2 + 3, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, group_norm
        )
        self.encoder2_3 = ResidualBlock(
            init_dim * 4, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, group_norm
        )

        self.encoder3_1 = ResidualBlock(
            init_dim * 4, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, group_norm, strided=True
        )
        self.encoder3_2 = ResidualBlock(
            init_dim * 4+3, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, group_norm
        )
        self.encoder3_3 = ResidualBlock(
            init_dim * 8, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, group_norm
        )

        self.encoder4_1 = ResidualBlock(
            init_dim * 8, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, group_norm, strided=True
        )
        self.encoder4_2 = ResidualBlock(
            init_dim * 8, init_dim * 16, kernel_size, init_radius * 8, init_sigma * 8, group_norm
        )
        self.encoder4_3 = ResidualBlock(
            init_dim * 16, init_dim * 16, kernel_size, init_radius * 8, init_sigma * 8, group_norm
        )

        self.decoder3 = UnaryBlock(init_dim * 24, init_dim * 8, group_norm)
        self.decoder2 = LastUnaryBlock(init_dim * 12, output_dim)
        self.linear = nn.Linear(cfg.geotransformer.hidden_dim, 4)
    def enhance(self, feats, color):
        feats = torch.cat([feats,
                           color[:, 0].reshape(-1, 1),  # h
                           color[:, 2].reshape(-1, 1),  # v
                           color[:, 1].reshape(-1, 1),  # s
                           ], dim=1)
        return feats

    def forward(self, feats, data_dict):
        feats_list = []

        points_list = data_dict['points']
        neighbors_list = data_dict['neighbors']
        subsampling_list = data_dict['subsampling']
        upsampling_list = data_dict['upsampling']
        with torch.no_grad():
            sorted_indices = kmeans_H_sort(data_dict['hsv'][0], n_clusters=10)
            inverse_indices = torch.empty_like(sorted_indices)
            inverse_indices[sorted_indices] = torch.arange(len(sorted_indices), device=sorted_indices.device)
        sorted_x_indices = torch.argsort(points_list[0][:, 0]).cuda()
        sorted_y_indices = torch.argsort(points_list[0][:, 1]).cuda()
        sorted_z_indices = torch.argsort(points_list[0][:, 2]).cuda()
        inv_idx_x = torch.empty_like(sorted_x_indices).cuda()
        inv_idx_x[sorted_x_indices] = torch.arange(len(sorted_x_indices),device=sorted_x_indices.device)
        inv_idx_y = torch.empty_like(sorted_y_indices).cuda()
        inv_idx_y[sorted_y_indices] = torch.arange(len(sorted_y_indices),device=sorted_x_indices.device)
        inv_idx_z = torch.empty_like(sorted_z_indices).cuda()
        inv_idx_z[sorted_z_indices] = torch.arange(len(sorted_z_indices),device=sorted_x_indices.device)

        feats_s1 = feats
        feats_s1 = self.enhance(feats_s1, data_dict['hsv'][0])
        feats_s1 = self.encoder1_1(feats_s1, points_list[0], points_list[0], neighbors_list[0])
        feats_s1 = self.encoder1_2(feats_s1, points_list[0], points_list[0], neighbors_list[0])

        feats_s1c = self.base_model1(feats_s1[sorted_indices].unsqueeze(0),self.enhance(points_list[0][sorted_indices], data_dict['hsv'][0][sorted_indices]).unsqueeze(0)).squeeze(0)
        feats_s1x = self.base_model1(feats_s1[sorted_x_indices].unsqueeze(0),self.enhance(points_list[0][sorted_x_indices], data_dict['hsv'][0][sorted_x_indices]).unsqueeze(0)).squeeze(0)
        feats_s1y = self.base_model1(feats_s1[sorted_y_indices].unsqueeze(0),self.enhance(points_list[0][sorted_y_indices], data_dict['hsv'][0][sorted_y_indices]).unsqueeze(0)).squeeze(0)
        feats_s1z = self.base_model1(feats_s1[sorted_z_indices].unsqueeze(0),self.enhance(points_list[0][sorted_z_indices], data_dict['hsv'][0][sorted_z_indices]).unsqueeze(0)).squeeze(0)
        pointsmamba_x= feats_s1x[inv_idx_x]
        pointsmamba_y= feats_s1y[inv_idx_y]
        pointsmamba_z= feats_s1z[inv_idx_z]
        pointsmamba_color= feats_s1c[inverse_indices]
        concatenated_features = torch.cat((pointsmamba_x, pointsmamba_y, pointsmamba_z,pointsmamba_color), dim=0)
        f = torch.mean(concatenated_features, dim=0)
        weights = nn.functional.softmax(self.linear(f), dim=0)
        feats_s1 = weights[0] * pointsmamba_x + weights[1] * pointsmamba_y + weights[2] * pointsmamba_z + weights[3] * pointsmamba_color
        feats_s2 = self.encoder2_1(feats_s1, points_list[1], points_list[0], subsampling_list[0])
        feats_s2 = self.enhance(feats_s2, data_dict['hsv'][1])
        feats_s2 = self.encoder2_2(feats_s2, points_list[1], points_list[1], neighbors_list[1])
        feats_s2 = self.encoder2_3(feats_s2, points_list[1], points_list[1], neighbors_list[1])

        feats_s3 = self.encoder3_1(feats_s2, points_list[2], points_list[1], subsampling_list[1])
        feats_s3 = self.enhance(feats_s3, data_dict['hsv'][2])

        feats_s3 = self.encoder3_2(feats_s3, points_list[2], points_list[2], neighbors_list[2])
        feats_s3 = self.encoder3_3(feats_s3, points_list[2], points_list[2], neighbors_list[2])

        feats_s4 = self.encoder4_1(feats_s3, points_list[3], points_list[2], subsampling_list[2])
        feats_s4 = self.encoder4_2(feats_s4, points_list[3], points_list[3], neighbors_list[3])
        feats_s4 = self.encoder4_3(feats_s4, points_list[3], points_list[3], neighbors_list[3])

        latent_s4 = feats_s4
        feats_list.append(feats_s4)

        latent_s3 = nearest_upsample(latent_s4, upsampling_list[2])
        latent_s3 = torch.cat([latent_s3, feats_s3], dim=1)
        latent_s3 = self.decoder3(latent_s3)
        feats_list.append(latent_s3)

        latent_s2 = nearest_upsample(latent_s3, upsampling_list[1])
        latent_s2 = torch.cat([latent_s2, feats_s2], dim=1)
        latent_s2 = self.decoder2(latent_s2)
        feats_list.append(latent_s2)

        feats_list.reverse()

        return feats_list
