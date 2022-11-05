# Pytorch
import torch
import torch.nn as nn
# Local
from .modules import compress_jpeg, decompress_jpeg
from .utils import diff_round, quality_to_factor
from models.differentiable_quantize import differentiable_quantize
import numpy as np


class DiffJPEG(nn.Module):
    def __init__(self, height, width, differentiable=True, quality=80, quality_range=3):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image hieght
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): max quality factor for jpeg compression scheme.
        '''
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = differentiable_quantize.apply
        else:
            rounding = torch.round
        # factor = quality_to_factor(quality)
        self.min_quality=quality-quality_range
        self.max_quality=quality+quality_range+1
        self.compress = compress_jpeg(rounding=rounding)
        self.decompress = decompress_jpeg(height, width)

    def forward(self, x):

        # quality  min: self.min_quality   max: self.max_quality
        quality=np.random.randint(self.min_quality,self.max_quality)
        factor = quality_to_factor(quality)
        comp_new, comp_before = self.compress(x, factor)
        #y, cb, cr = self.compress(x,factor)
        y, cb, cr = comp_new['y'], comp_new['cb'], comp_new['cr'], 
        recovered = self.decompress(y, cb, cr, factor)
        return recovered#, comp_new, comp_before
