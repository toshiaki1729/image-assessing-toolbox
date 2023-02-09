from . import utilities
from . import Manifold


class PrecisionRecall:
    def __init__(self, precision, recall):
        self.precision = precision
        self.recall = recall


# brought from
# https://github.com/youngjung/improved-precision-and-recall-metric-pytorch/blob/5ad4629b07f3f3a51184c39d3dbe9085a60e264c/improved_precision_recall.py
# and modified
def compute_metrics(manifold, features, device='cuda'):
    num_subjects = features.shape[0]
    count = 0
    dist = utilities.compute_pairwise_distances(manifold.features, features, device)
    for i in range(num_subjects):
        # modified because official implemention is less than "or equals"
        # https://github.com/kynkaat/improved-precision-and-recall-metric/blob/master/precision_recall.py
        count += (dist[:, i] <= manifold.radii).any()
    return count / num_subjects


def get_precision_and_recall(x:Manifold, y:Manifold, device='cuda'):
    precision = compute_metrics(x, y.features, device)
    recall = compute_metrics(y, x.features, device)
    return PrecisionRecall(precision, recall)