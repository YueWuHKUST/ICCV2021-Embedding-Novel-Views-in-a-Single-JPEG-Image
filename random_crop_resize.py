import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RandomCropResize(nn.Module):
    def __init__(self,opt):
        super(RandomCropResize,self).__init__()
        self.opt=opt

    def forward(self,embedding_img,mpis):
        # embedding_img: predicted embedding image
        # mpis: the list of mpis

        # randomly set crop and resize factors
        crop_ratio_w=np.random.uniform(self.opt.min_crop_size,1)
        crop_ratio_h=np.random.uniform(self.opt.min_crop_size,1)
        resize_ratio_w=np.random.uniform(self.opt.min_resize_size,self.opt.max_resize_size)
        resize_ratio_h=np.random.uniform(self.opt.min_resize_size,self.opt.max_resize_size)

        crop_h=int(embedding_img.size(2)*crop_ratio_w)
        crop_w=int(embedding_img.size(3)*crop_ratio_h)
        resize_h=int(crop_h*resize_ratio_w)
        resize_w=int(crop_w*resize_ratio_h)

        crop_y=np.random.randint(embedding_img.size(2)-crop_h)
        crop_x=np.random.randint(embedding_img.size(3)-crop_w)


        # crop and resize the ref image
        embedding_img=F.interpolate(embedding_img[:,:,crop_y:crop_y+crop_h,crop_x:crop_x+crop_w],(resize_h,resize_w),mode='bilinear')

        # adjust the mpis
        resized_mpis=[]
        for mpi in mpis:
            mpi=F.interpolate(mpi[:,:,crop_y:crop_y+crop_h,crop_x:crop_x+crop_w],(resize_h,resize_w),mode='bilinear')
            resized_mpis.append(mpi)

        return embedding_img,resized_mpis