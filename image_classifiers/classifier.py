from typing import List

import numpy as np
from torch.utils.data import DataLoader

from .utilities import ImageDataset


class Classifier():
    def __init__(self, device):
        self.device = device

    def create_dataset(self, images:List[str]) -> ImageDataset:
        raise NotImplementedError
    
    def __enter__(self):
        self.load()
        print(f'Load classifier: {type(self).__name__}')
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.unload()
        print(f'Unload classifier: {type(self).__name__}')
        return self
    
    def load(self):
        pass

    def unload(self):
        pass

    def __call__(self, images:List[str], batch_size:int=50, num_workers:int=8):
        dataset = self.create_dataset(images)
        if batch_size > len(dataset):
            batch_size = len(dataset)
        dataloader = DataLoader(dataset, batch_size, num_workers=num_workers)
        return self.apply(dataloader)

    def apply(self, dataloader: DataLoader) -> np.ndarray:
        raise NotImplementedError