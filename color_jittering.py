import torch
import torch.nn as nn
import kornia
import numpy as np
import math
from kornia.constants import pi


def rgb_to_hsv(image):

    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))


    if image.is_cuda and torch.__version__ == '1.6.0':
        maxc, max_indices = image.cpu().max(-3)
        maxc, max_indices = maxc.to(image), max_indices.to(image.device)
    else:
        maxc, max_indices = image.max(-3)

    minc = image.min(-3)[0]

    v = maxc  # brightness

    deltac = maxc - minc
    s = deltac / (v + 1e-6)

    # avoid division by zero
    deltac = torch.where(
        deltac == 0, torch.ones_like(deltac), deltac)

    maxc_tmp = maxc.unsqueeze(-3) - image
    rc: torch.Tensor = maxc_tmp[..., 0, :, :]
    gc: torch.Tensor = maxc_tmp[..., 1, :, :]
    bc: torch.Tensor = maxc_tmp[..., 2, :, :]

    h = torch.stack([
        bc - gc,
        2.0 * deltac + rc - bc,
        4.0 * deltac + gc - rc,
    ], dim=-3)

    h = torch.gather(h, dim=-3, index=max_indices[..., None, :, :])
    h = h.squeeze(-3)
    h = h / deltac

    h = (h / 6.0) % 1.0

    h = 2 * math.pi * h

    return torch.stack([h, s, v], dim=-3)



def hsv_to_rgb(image):

    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    h = image[..., 0, :, :] / (2 * math.pi)
    s = image[..., 1, :, :]
    v = image[..., 2, :, :]

    hi = torch.floor(h * 6) % 6
    f = ((h * 6) % 6) - hi
    one = torch.tensor(1.).to(image.device)
    p = v * (one - s)
    q = v * (one - f * s)
    t = v * (one - (one - f) * s)

    hi = hi.long()
    indices= torch.stack([hi, hi + 6, hi + 12], dim=-3)
    out = torch.stack((
        v, q, p, p, t, v,
        t, v, v, q, p, p,
        p, p, t, v, v, q,
    ), dim=-3)
    out = torch.gather(out, -3, indices)

    return out





def adjust_saturation_raw(input, saturation_factor):

    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not isinstance(saturation_factor, (float, torch.Tensor,)):
        raise TypeError(f"The saturation_factor should be a float number or torch.Tensor."
                        f"Got {type(saturation_factor)}")

    if isinstance(saturation_factor, float):
        saturation_factor = torch.tensor([saturation_factor])

    saturation_factor = saturation_factor.to(input.device).to(input.dtype)

    if (saturation_factor < 0).any():
        raise ValueError(f"Saturation factor must be non-negative. Got {saturation_factor}")

    for _ in input.shape[1:]:
        saturation_factor = torch.unsqueeze(saturation_factor, dim=-1)

    # unpack the hsv values
    h, s, v = torch.chunk(input, chunks=3, dim=-3)

    # transform the hue value and appl module
    s_out = torch.clamp(s * saturation_factor, min=0, max=1)

    # pack back back the corrected hue
    out = torch.cat([h, s_out, v], dim=-3)

    return out


def adjust_saturation(input, saturation_factor):

    # convert the rgb image to hsv
    x_hsv= rgb_to_hsv(input)

    # perform the conversion
    x_adjusted= adjust_saturation_raw(x_hsv, saturation_factor)

    # convert back to rgb
    out= hsv_to_rgb(x_adjusted)
    return out






def adjust_hue_raw(input, hue_factor):

    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not isinstance(hue_factor, (float, torch.Tensor)):
        raise TypeError(f"The hue_factor should be a float number or torch.Tensor in the range between"
                        f" [-PI, PI]. Got {type(hue_factor)}")

    if isinstance(hue_factor, float):
        hue_factor = torch.tensor([hue_factor])

    hue_factor = hue_factor.to(input.device).to(input.dtype)

    if ((hue_factor < -pi) | (hue_factor > pi)).any():
        raise ValueError(f"Hue-factor must be in the range [-PI, PI]. Got {hue_factor}")

    for _ in input.shape[1:]:
        hue_factor = torch.unsqueeze(hue_factor, dim=-1)

    # unpack the hsv values
    h, s, v = torch.chunk(input, chunks=3, dim=-3)

    # transform the hue value and appl module
    divisor: float = 2 * pi.item()
    h_out = torch.fmod(h + hue_factor, divisor)

    # pack back back the corrected hue
    out = torch.cat([h_out, s, v], dim=-3)

    return out


def adjust_hue(input, hue_factor) :

    # convert the rgb image to hsv
    x_hsv = rgb_to_hsv(input)

    # perform the conversion
    x_adjusted = adjust_hue_raw(x_hsv, hue_factor)

    # convert back to rgb
    out = hsv_to_rgb(x_adjusted)
    return out







class ColorJitter(nn.Module):
    def __init__(self,opt):
        super(ColorJitter,self).__init__()
        self.opt=opt

    def forward(self,embedding_img,mpis):
        # embedding_img: predicted embedding image
        # mpis: the list of mpis

        # randomly set adjust factors
        self.brightness_factor=np.random.uniform(-self.opt.brightness,self.opt.brightness)
        self.hue_factor=np.random.uniform(-self.opt.hue,self.opt.hue)
        self.contrast_factor=np.random.uniform(-self.opt.contrast+1,self.opt.contrast+1)
        self.saturation_factor=np.random.uniform(-self.opt.saturation+1,self.opt.saturation+1)

        # adjust the ref image
        embedding_img = (embedding_img + 1.0) * 0.5
        embedding_img = self.adjust_img(embedding_img)
        embedding_img = (embedding_img - 0.5) * 2.0

        # adjust the mpis
        adjusted_mpis=[]
        for mpi in mpis:
            cnt_rgb = (mpi[:,0:3,:,:] + 1.0) * 0.5
            adjusted_rgb = self.adjust_img(cnt_rgb)
            adjusted_rgb = (adjusted_rgb - 0.5)*2.0
            adjusted_mpi=torch.cat([adjusted_rgb, mpi[:,3:4,:,:]],dim=1)
            adjusted_mpis.append(adjusted_mpi)

        return embedding_img,adjusted_mpis

    def adjust_img(self,img):
        img=kornia.adjust_brightness(img,self.brightness_factor)
        img=kornia.adjust_contrast(img,self.contrast_factor)
        img=adjust_hue(img,self.hue_factor)
        img=adjust_saturation(img,self.saturation_factor)
        return img

