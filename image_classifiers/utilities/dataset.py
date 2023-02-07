from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as tf

from . import resizer

class ImageDataset(Dataset):
    def __init__(self, files:List[str], size:Tuple[int, int], transforms:tf.Compose, repeat_edge_on_resize:bool = False):
        self.files = files
        self.size = size
        self.transforms = transforms
        self.repeat_edge = repeat_edge_on_resize

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert('RGB')
        if self.repeat_edge:
            # resize with PIL lanczos resampler if the aspect ratio of img is the same as self.size.
            # if not, img will be resized to fit the size, and the outside will be filled with edge color of img.
            img = resizer.resize_and_fill(img, self.size)
        else:
            # just resize with PIL lanczos resampler regardless of aspect ratio.
            img = resizer.resize(img, self.size)
        img = self.transforms(img)
        return img
