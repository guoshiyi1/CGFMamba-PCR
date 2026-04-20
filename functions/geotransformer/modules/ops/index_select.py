import torch


def index_select(data: torch.Tensor, index: torch.LongTensor, dim: int) -> torch.Tensor:

    output = data.index_select(dim, index.reshape(-1))

    if index.ndim > 1:
        output_shape = data.shape[:dim] + index.shape + data.shape[dim:][1:]
        output = output.view(*output_shape)

    return output
