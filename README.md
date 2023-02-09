# image-assessing-toolbox
Toolbox for assessing generative models working with PyTorch



## Features
- **Generate feature vectors from image dataset**
  - With classifiers trained on ImageNet dataset
    - **Inception v3** ([Paper (arXiv)](https://arxiv.org/abs/1512.00567))
      - [pytorch-fid](https://github.com/mseitzer/pytorch-fid) by mseitzer is used internally and its model has the same weights as the original implemention of FID ([Paper (arXiv)](https://arxiv.org/abs/1706.08500), [GitHub](https://github.com/bioinf-jku/TTUR))
    - **VGG-16** ([Paper (arXiv)](https://arxiv.org/abs/1409.1556))
      - Use of this model is proposed in *Improved Precision and Recall Metric for Assessing Generative Models* by Kynkäänniemi et al. ([Paper (arXiv)](https://arxiv.org/abs/1904.06991), [GitHub](https://github.com/kynkaat/improved-precision-and-recall-metric))
  - With classifiers trained on danbooru dataset (may be better for anime-like images)
    - [**danbooru-pretrained**](https://github.com/RF5/danbooru-pretrained) (pretrained **ResNet50** ([Paper (arXiv)](https://arxiv.org/abs/1512.03385)))
      - Use of this model as a classifier is proposed as [DaFID](https://github.com/birdManIkioiShota/DaFID-512) by birdManIkioiShota
  - Save the vectors as `.npz` files
- **FID-like score**
  - As you know, FID is one of the most common method to evaluate similarity between two datasets which is proposed in *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium* by Heusel et al. ([Paper (arXiv)](https://arxiv.org/abs/1706.08500), [GitHub](https://github.com/bioinf-jku/TTUR))
  - Evaluate the similarity between two image datasets by using the method similar to FID
    - "similar" because it may not always use Inception v3
    - Approximate the distribution of feature vectors as multidimensional Gaussian and compute the Fréchet distance between two Gaussians.
  - Using [pytorch-fid](https://github.com/mseitzer/pytorch-fid)
- **Improved Precision and Recall Metrics**
  - Evaluate the similarity between two image datasets with [Improved Precision and Recall Metric](https://arxiv.org/abs/1904.06991)
    - Approximate the distribution of feature vectors as a set of hyperspheres with radius equals to distance of their kth nearest neighbor.
  - The code is based on [the official TensorFlow implemention](https://github.com/kynkaat/improved-precision-and-recall-metric) and [the PyTorch implemention by youngjung](https://github.com/youngjung/improved-precision-and-recall-metric-pytorch)



## Requirements
All required libraries are listed in `requirements.txt`  
[**Python 3**](https://www.python.org/) (developed with 3.10.7)  
[**PyTorch**](https://pytorch.org/) (developed with 1.13.1+cu117,  CUDA is needed if you want to use danbooru-pretrained)   

I recommend you to use venv to separate environment.  
```
python -m venv venv
./venv/Scripts/activate
```
or
```
python -m venv --system-site-packages venv
./venv/Scripts/activate
```
for minimal install.  
After setup above, run following command.
```
pip install -r requirements.txt
```


## Usage
1. Use `preprocess.py` to generate feature vectors from dataset
1. Use `frechet_distance.py` or  `precision_recall.py` to evaluate their similality
1. See the output csv
    - Use `simple_visualizer.py` if needed
  
Please note that **at least** 10,000 images are needed to get meaningful result.  
This is because distribution of feature vectors is extremely sparse since their dimensions are so high (like 512, 2048),
