from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections
from PIL import Image
import cv2
import scipy, scipy.misc
import sys

def save_network(network, save_path):
    network_cpu = network.cpu().state_dict()
    torch.save(network_cpu, save_path)
    if torch.cuda.is_available():
        network.cuda()

## Load pretrain model
def load_network(network, network_name, network_epoch, epoch_iter, save_dir):         
    save_filename = '%s_%03d_%08d.pth' % (network_name, network_epoch, epoch_iter)
    save_path = os.path.join(save_dir, save_filename)        
    if not os.path.isfile(save_path):
        print('%s not exists yet!' % save_path)
        #if 'G' in network_label:
        #    raise('Generator must exist!')
    else:
        #network.load_state_dict(torch.load(save_path))
        try:
            network.load_state_dict(torch.load(save_path))
            print("Load pretrain model %s success"%network_name)
        except:
            try:   
                # try loading multigpu model in single gpu mode
                pretrained_dict = torch.load(save_path)                
                model_dict = network.state_dict()
                # remove "module."
                pretrained_dict_new = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}                    
                network.load_state_dict(pretrained_dict_new)
                print('load multigpu model in single gpu model', network_name)
            except:
                ### printout layers in pretrained model
                initialized = set()                    
                for k, v in pretrained_dict.items():    
                    #print("k", k)                  
                    initialized.add(k.split('.')[0])     
                try:

                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}                    
                    network.load_state_dict(pretrained_dict)
                    print('Pretrained network %s has excessive layers; Only loading layers that are used' % network_name)
                except:
                    print('Pretrained network %s has fewer layers; The following are not initialized:' % network_name)
                    if sys.version_info >= (3,0):
                        not_initialized = set()
                    else:
                        from sets import Set
                        not_initialized = Set()
                    for k, v in pretrained_dict.items():                      
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v

                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])                            
                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)



# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if isinstance(image_tensor, torch.autograd.Variable):
        image_tensor = image_tensor.data
    if len(image_tensor.size()) == 4:
        image_tensor = image_tensor[0]
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        # -1 1 -> 0 - 2 -> 0 - 1
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    #image_numpy = (np.transpose(image_numpy, (1, 2, 0)) * std + mean)  * 255.0        
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)

def tensor2mask(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if isinstance(image_tensor, torch.autograd.Variable):
        image_tensor = image_tensor.data
    if len(image_tensor.size()) == 4:
        image_tensor = image_tensor[0]
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    #image_numpy = (np.transpose(image_numpy, (1, 2, 0)) * std + mean)  * 255.0        
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)

def tensor2composite(image_tensor, alpha_tensor, imtype=np.uint8):
    if isinstance(image_tensor, torch.autograd.Variable):
        image_tensor = image_tensor.data
    if isinstance(alpha_tensor, torch.autograd.Variable):
        alpha_tensor = alpha_tensor.data
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    alpha_numpy = alpha_tensor.cpu().float().numpy()
    alpha_numpy = np.transpose(alpha_numpy, (1, 2, 0)) * 255.0      
    image_numpy = np.clip(image_numpy, 0, 255)
    alpha_numpy = np.clip(alpha_numpy, 0, 255)
    composite = np.concatenate([image_numpy, alpha_numpy], axis=2)
    return composite.astype(imtype)


def tensor2image(image_tensor,normalize=True, imtype=np.uint8):
    if isinstance(image_tensor, torch.autograd.Variable):
        image_tensor = image_tensor.data
    image_dims=len(image_tensor.size())
    image_numpy = image_tensor.cpu().float().numpy()
    if image_dims==3:
        if normalize:
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1.0) / 2.0 * 255.0
        else:
            image_numpy = np.transpose(image_numpy, (1, 2, 0))
            image_numpy[:,:,:3] = (image_numpy[:,:,:3] + 1.0) / 2.0 * 255.0
            image_numpy[:,:,3:] = image_numpy[:,:,3:] * 255.0
    elif image_dims==2:
        if normalize:
            image_numpy = (image_numpy + 1) / 2.0 * 255.0
        else:
            image_numpy = image_numpy  * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    #scipy.misc.toimage(image_numpy).save(image_path)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def save_image_jpeg(image_numpy, image_path,quality):
    #scipy.misc.toimage(image_numpy).save(image_path)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path,quality=quality)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])