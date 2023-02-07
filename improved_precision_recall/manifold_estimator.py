import numpy as np
import torch.nn as nn

from . import utilities
from . import Manifold


class ManifoldEstimator():
    def __init__(self, k=3, device='cuda'):
        self.k = k
        self.device = device

    def evaluate(self, feat):
        return Manifold(feat, self.distances2radii(utilities.compute_pairwise_distances(feat, device=self.device)))
    
    # brought from
    # https://github.com/youngjung/improved-precision-and-recall-metric-pytorch/blob/5ad4629b07f3f3a51184c39d3dbe9085a60e264c/improved_precision_recall.py
    # and modified
    def distances2radii(self, distances):
        num_features = distances.shape[0]
        radii = np.zeros(num_features)
        for i in range(num_features):
            radii[i] = self.get_kth_value(distances[i])
        return radii

    # brought from
    # https://github.com/youngjung/improved-precision-and-recall-metric-pytorch/blob/5ad4629b07f3f3a51184c39d3dbe9085a60e264c/improved_precision_recall.py
    # and modified
    def get_kth_value(self, np_array):
        kprime = self.k + 1  # kth NN should be (k+1)th because closest one is itself
        idx = np.argpartition(np_array, kprime)
        k_smallests = np_array[idx[:kprime]]
        kth_value = k_smallests.max()
        return kth_value
    
