import torch
import numpy as np


def rad2deg(rad: torch.Tensor) -> torch.Tensor:
    factor = 180.0 / np.pi
    deg = rad * factor
    return deg


def deg2rad(deg: torch.Tensor) -> torch.Tensor:
    factor = np.pi / 180.0
    rad = deg * factor
    return rad


def vector_angle(x: torch.Tensor, y: torch.Tensor, dim: int, use_degree: bool = False):

    cross = torch.linalg.norm(torch.cross(x, y, dim=dim), dim=dim)  # (*, 3 *) x (*, 3, *) -> (*, 3, *) -> (*)
    dot = torch.sum(x * y, dim=dim)  # (*, 3 *) x (*, 3, *) -> (*)
    angles = torch.atan2(cross, dot)  # (*)
    if use_degree:
        angles = rad2deg(angles)
    return angles
