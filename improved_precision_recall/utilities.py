import numpy as np
import torch


def as_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    elif isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    else:
        raise NotImplementedError


# brought from
# https://github.com/youngjung/improved-precision-and-recall-metric-pytorch/blob/5ad4629b07f3f3a51184c39d3dbe9085a60e264c/improved_precision_recall.py
# and modified
def compute_pairwise_distances(feat_x, feat_y=None, device='cuda') -> np.ndarray:
    with torch.no_grad():
        x = as_tensor(feat_x).to(device)
        xsq = x.square().sum(dim=1, keepdim=True)
        nx = x.shape[0]
        if feat_y is None:
            y = x
            ysq = xsq
            ny = nx
        else:
            y = as_tensor(feat_y).to(device)
            ysq = y.square().sum(dim=1, keepdim=True)
            ny = y.shape[0]
        xsq = xsq.repeat([1, ny])
        ysq = ysq.mT.repeat([nx, 1])
        dsq = xsq + ysq - 2*torch.matmul(x, y.mT)
        dsq = torch.maximum(dsq, torch.zeros_like(dsq))
        # sqrt on youngjung's implemention
        # d = d.sqrt()
    return dsq.detach().cpu().numpy()