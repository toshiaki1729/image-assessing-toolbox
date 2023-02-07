import gc
from typing import List

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from tqdm import tqdm
import numpy as np

from . import Classifier
from .utilities import ImageDataset
from torch.utils.data import DataLoader

class VGG16(Classifier):
    def create_dataset(self, files:List[str]) -> ImageDataset:
        transformers = transforms.Compose([
            transforms.ToTensor()
        ])
        return ImageDataset(files, (224, 224), transformers)
    
    def load(self):
        self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.model.classifier = nn.Sequential(*self.model.classifier[:5]) # use the output of ReLU layer
        self.model.to(self.device).eval()

    def unload(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

    # brought from mseitzer's pytorch-fid and modified
    # https://github.com/mseitzer/pytorch-fid/blob/0a754fb8e66021700478fd365b79c2eaa316e31b/src/pytorch_fid/fid_score.py#L93-L149
    def apply(self, dataloader:DataLoader):
        assert(hasattr(self, 'model'))
        
        pred_arr = np.empty((len(dataloader.dataset), 4096))
        start_idx = 0

        for batch in tqdm(dataloader):
            batch = batch.to(self.device)

            with torch.no_grad():
                pred = self.model(batch).detach().cpu().numpy()
            
            pred_arr[start_idx:start_idx + pred.shape[0]] = pred
            start_idx = start_idx + pred.shape[0]

        return pred_arr