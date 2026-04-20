#pragma once

#include <vector>
#include "../../common/torch_helper.h"

std::vector<at::Tensor> grid_subsampling_dps(
  at::Tensor points,
  at::Tensor dps,
  at::Tensor lengths,
  float voxel_size
);
