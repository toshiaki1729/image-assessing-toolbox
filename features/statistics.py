import numpy as np


class Statistics:
    def __init__(self, name:str, classifier:str, features:np.ndarray):
        self.name = name
        self.classifier = classifier
        self.values = features
        self.mu = np.mean(features, axis=0)
        self.sigma = np.cov(features, rowvar=False)
    