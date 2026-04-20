import torch
import numpy as np
import open3d as o3d

def calc_ppf_gpu(points, point_normals, patches, patch_normals):
    '''
    Calculate ppf gpu
    points: [n, 3]
    point_normals: [n, 3]
    patches: [n, nsamples, 3]
    patch_normals: [n, nsamples, 3]
    '''
    points = torch.unsqueeze(points, dim=1).expand(-1, patches.shape[1], -1)
    point_normals = torch.unsqueeze(point_normals, dim=1).expand(-1, patches.shape[1], -1)
    vec_d = patches - points #[n, n_samples, 3]
    d = torch.sqrt(torch.sum(vec_d ** 2, dim=-1, keepdim=True)) #[n, n_samples, 1]
    # angle(n1, vec_d)
    y = torch.sum(point_normals * vec_d, dim=-1, keepdim=True)
    x = torch.cross(point_normals, vec_d, dim=-1)
    x = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))
    angle1 = torch.atan2(x, y) / np.pi

    # angle(n2, vec_d)
    y = torch.sum(patch_normals * vec_d, dim=-1, keepdim=True)
    x = torch.cross(patch_normals, vec_d, dim=-1)
    x = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))
    angle2 = torch.atan2(x, y) / np.pi

    # angle(n1, n2)
    y = torch.sum(point_normals * patch_normals, dim=-1, keepdim=True)
    x = torch.cross(point_normals, patch_normals, dim=-1)
    x = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))
    angle3 = torch.atan2(x, y) / np.pi

    ppf = torch.cat([d, angle1, angle2, angle3], dim=-1) #[n, samples, 4]
    return ppf

def to_o3d_pcd(pcd):
    '''
    Transfer a point cloud of numpy.ndarray to open3d point cloud
    :param pcd: point cloud of numpy.ndarray in shape[N, 3]
    :return: open3d.geometry.PointCloud()
    '''
    pcd_ = o3d.geometry.PointCloud()
    pcd_.points = o3d.utility.Vector3dVector(pcd)
    return pcd_

def normal_redirect(points, normals, view_point):
    '''
    Make direction of normals towards the view point
    '''
    vec_dot = np.sum((view_point - points) * normals, axis=-1)
    mask = (vec_dot < 0.)
    redirected_normals = normals.copy()
    redirected_normals[mask] *= -1.
    return redirected_normals

def group_all(feats):
    '''
    all-to-all grouping
    feats: [n, c]
    out: grouped feat: [n, n, c]
    '''
    grouped_feat = torch.unsqueeze(feats, dim=0)
    grouped_feat = grouped_feat.expand(feats.shape[0], -1, -1) #[n, n, c]
    return grouped_feat
