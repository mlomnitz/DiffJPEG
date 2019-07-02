# DiffJPEG: A PyTorch implementation

This is a pytorch implementation of differentiable jpeg compression algorithm.  This work is based on the discussion in this [paper](https://machine-learning-and-security.github.io/papers/mlsec17_paper_54.pdf).  The work relies heavily on the tensorflow implementation in this [repository](https://github.com/rshin/differentiable-jpeg)

## Requirements
- Pytorch 1.0.0
- numpy 1.15.4

## Use

DiffJPEG functions as a standard pytorch module/layer.  To use, first import the layer and then initialize with the desired parameters:
- differentaible(bool): If true uses custom differentiable rounding function, if false uses standrard torch.round
- quality(float): Quality factor for jpeg compression scheme.

``` python
from DiffJPEG import DiffJPEG
jpeg = DiffJPEG(hieght=224, width=224, differentiable=True, quality=80)
```

![image](./diffjpeg.png)
