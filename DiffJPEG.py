# Pytorch
import torch
import torch.nn as nn
# Local
from compression import compress_jpeg
from decompression import decompress_jpeg
from utils import diff_round, quality_to_factor


class DiffJPEG(nn.Module):
    def __init__(self, differentiable=True, quality=80):
        ''' Initialize the DiffJPEG layer
        Inputs:
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme. 
        '''
        super(DiffJPEG, self).__init__()
        if differentiable:
            self.rounding = diff_round
        else:
            self.rounding = torch.round
        self.factor = quality_to_factor(quality)

    def forward(self, x):
        '''

        '''
        height, width = x.shape[2:]
        y, cb, cr = compress_jpeg(
            x, rounding=self.rounding, factor=self.factor)
        recovered = decompress_jpeg(y, cb, cr, height=height, width=width,
                                    rounding=self.rounding, factor=self.factor)
        return recovered
