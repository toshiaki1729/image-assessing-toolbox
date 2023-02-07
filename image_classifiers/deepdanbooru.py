import gc

import torch
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from typing import List
from tqdm import tqdm
import numpy as np

from . import Classifier
from .utilities import ImageDataset


# brought from DaFID-512 by birdManIkoiShota and modified
# https://github.com/birdManIkioiShota/DaFID-512
class DeepDanbooru(Classifier):
    def __init__(self, dims, device):
        super().__init__(device)
        if dims not in {512, 4096, 6000}:
            raise NotImplementedError
        self.dims = dims

    def create_dataset(self, files:List[str]) -> ImageDataset:
        transformers = transforms.Compose([
            transforms.ToTensor()
        ])
        return ImageDataset(files, (512, 512), transformers)

    def load(self):
        self.model = torch.hub.load('RF5/danbooru-pretrained', 'resnet50')
        if self.dims == 512:
            identity_falyers = [7, 8]
        elif self.dims == 4096:
            identity_falyers = [3, 4, 5, 6, 7, 8]
        else:
            identity_falyers = []
        for i in identity_falyers:
            self.model[1][i] = torch.nn.Identity()
        self.model.eval().to(self.device)
    
    def unload(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

    def apply(self, dataloader:DataLoader):
        assert(hasattr(self, 'model'))

        pred_arr = np.empty((len(dataloader.dataset), self.dims))
        start_idx = 0

        for batch in tqdm(dataloader):
            batch:torch.Tensor = batch.to(self.device)

            with torch.no_grad():
                pred = self.model(batch).detach().cpu().numpy()

            pred_arr[start_idx:start_idx + pred.shape[0]] = pred
            start_idx = start_idx + pred.shape[0]

        return pred_arr



class DaFID512(DeepDanbooru):
    def __init__(self, device):
        super().__init__(512, device)

class DaFID4096(DeepDanbooru):
    def __init__(self, device):
        super().__init__(4096, device)

class DaFID6000(DeepDanbooru):
    def __init__(self, device):
        super().__init__(6000, device)