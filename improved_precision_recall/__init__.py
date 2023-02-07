from .manifold import Manifold
from .manifold_estimator import ManifoldEstimator
from .metrics import compute_metrics


class PrecisionRecall:
    def __init__(self, precision, recall):
        self.precision = precision
        self.recall = recall


def get_precision_and_recall(x:Manifold, y:Manifold, device='cuda'):
    precision = compute_metrics(x, y.features, device)
    recall = compute_metrics(y, x.features, device)
    return PrecisionRecall(precision, recall)