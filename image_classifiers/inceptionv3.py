import gc
from typing import List

import torch
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np

import pytorch_fid.inception as inception

from . import Classifier
from .utilities import ImageDataset


class InceptionV3(Classifier):
    def create_dataset(self, files:List[str]) -> ImageDataset:
        transformers = transforms.Compose([
            transforms.ToTensor()
        ])
        return ImageDataset(files, (299, 299), transformers)
    
    def load(self):
        block_idx = inception.InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = inception.InceptionV3([block_idx], resize_input=False).to(self.device).eval()
    
    def unload(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

    # brought from mseitzer's pytorch-fid and modified
    # https://github.com/mseitzer/pytorch-fid/blob/0a754fb8e66021700478fd365b79c2eaa316e31b/src/pytorch_fid/fid_score.py#L93-L149
    def apply(self, dataloader:DataLoader):
        assert(hasattr(self, 'model'))
        
        pred_arr = np.empty((len(dataloader.dataset), 2048))
        start_idx = 0

        for batch in tqdm(dataloader):
            batch = batch.to(self.device)

            with torch.no_grad():
                pred = self.model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2).detach().cpu().numpy()

            pred_arr[start_idx:start_idx + pred.shape[0]] = pred
            start_idx = start_idx + pred.shape[0]

        return pred_arr