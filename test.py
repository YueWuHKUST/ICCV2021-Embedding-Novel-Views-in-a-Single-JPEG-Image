import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
import time
from PIL import Image
from collections import OrderedDict
import functools
import kornia
import copy
import lpips

import util.util as util
from util.visualizer import Visualizer
from models.networks import Encoder,Decoder, MultiscaleDiscriminator, EncoderAlphaWeight, DecoderDepthwise
from models.unet_model import UNetEncoder,UNetDecoder
from models.resunet_model import ResUnetEncoder, ResUnetDecoder
from opt_test import get_opts
from data.real_dataset import RealDataset
from models.losses import LossFunction
from DiffJPEG.DiffJPEG import DiffJPEG
from models.differentiable_quantize import DifferentiableQuantize
from color_jittering import ColorJitter
from random_crop_resize import RandomCropResize
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
torch.autograd.set_detect_anomaly(True)


def mpi_collect(batch):
    if None in batch:
        return None
    ref_list=[]
    mpi_num=len(batch[0][1])
    mpi_list=[]
    for _ in range(mpi_num):
        mpi_list.append([])
    for data in batch:
        ref_list.append(data[0].unsqueeze(0))
        for idx in range(mpi_num):
            mpi_list[idx].append(data[1][idx].unsqueeze(0))

    ref=torch.cat(ref_list,dim=0)
    mpis=[]
    for mpi in mpi_list:
        mpis.append(torch.cat(mpi,dim=0))

    return [ref,mpis]


def image_save_load_jpeg(embedding_img,ref,exp_name,quality=90):
    jpeg_temp_path='./temp_%s.jpg'%(exp_name)
    gt_jpeg_temp_path='./gt_temp_%s.jpg'%(exp_name)
    image_numpy=util.tensor2image(embedding_img[0])
    gt_image_numpy=util.tensor2image(ref[0])
    image_pil = Image.fromarray(image_numpy)
    gt_image_pil = Image.fromarray(gt_image_numpy)
    image_pil.save(jpeg_temp_path,quality=quality)
    gt_image_pil.save(gt_jpeg_temp_path,quality=quality)
    image_read_pil=Image.open(jpeg_temp_path)
    gt_image_read_pil=Image.open(gt_jpeg_temp_path)
    image_read_np = np.array(image_read_pil)
    gt_image_read_np = np.array(gt_image_read_pil)
    image_read_tensor=transforms_rgb(image_read_np)
    gt_image_read_tensor=transforms_rgb(gt_image_read_np)
    embedding_img_read=image_read_tensor.unsqueeze(0)
    gt_embedding_img_read=gt_image_read_tensor.unsqueeze(0)
    return embedding_img_read,gt_embedding_img_read



def image_save_load_png(embedding_img,ref,jpeg_compress,exp_name,quality=90):
    embedding_img=(embedding_img+1)/2
    embedding_img = jpeg_compress(embedding_img)
    embedding_img = embedding_img*255.0
    embedding_img = torch.round(embedding_img)
    embedding_img = embedding_img/255.0
    embedding_img=embedding_img*2-1
    # jpeg_temp_path='./temp_%s.png'%(exp_name)
    # gt_jpeg_temp_path='./gt_temp_%s.png'%(exp_name)
    # image_numpy=util.tensor2image(embedding_img[0])
    # gt_image_numpy=util.tensor2image(ref[0])
    # image_pil = Image.fromarray(image_numpy)
    # gt_image_pil = Image.fromarray(gt_image_numpy)
    # image_pil.save(jpeg_temp_path)
    # gt_image_pil.save(gt_jpeg_temp_path)
    # image_read_pil=Image.open(jpeg_temp_path)
    # gt_image_read_pil=Image.open(gt_jpeg_temp_path)
    # image_read_np = np.array(image_read_pil)
    # gt_image_read_np = np.array(gt_image_read_pil)
    # image_read_tensor=transforms_rgb(image_read_np)
    # gt_image_read_tensor=transforms_rgb(gt_image_read_np)
    # embedding_img_read=image_read_tensor.unsqueeze(0)
    # gt_embedding_img_read=gt_image_read_tensor.unsqueeze(0)
    embedding_img_read=embedding_img
    gt_embedding_img_read=ref

    return embedding_img_read,gt_embedding_img_read

def image_save_load_true_png(embedding_img, ref):
    embedding_img=(embedding_img+1)/2
    embedding_img = embedding_img*255.0
    embedding_img = torch.round(embedding_img)
    embedding_img = embedding_img/255.0
    embedding_img=embedding_img*2-1
    embedding_img_read=embedding_img
    gt_embedding_img_read=ref

    return embedding_img_read,gt_embedding_img_read





if __name__ == '__main__':
    FLAGS=get_opts()
    #testing only support single image, single gpu
    use_cuda=torch.cuda.is_available()
    device=torch.device("cuda" if use_cuda else "cpu")

    os.makedirs(FLAGS.checkpoints_dir, exist_ok=True)

    # data
    print("begin loading dataset")
    batch_size=FLAGS.batch_size_each_gpu*torch.cuda.device_count()
    dataset=RealDataset(FLAGS)
    dataloader=DataLoader(dataset,
                          shuffle=False,
                          num_workers=FLAGS.num_workers,
                          batch_size=batch_size,
                          pin_memory=True,
                          collate_fn=mpi_collect)

    # model
    if FLAGS.encoder_type==1:
        encoder = Encoder(input_num=FLAGS.mpi_num)
    elif FLAGS.encoder_type==2:
        encoder = EncoderAlphaWeight(input_num=FLAGS.mpi_num, base_feature_channels=FLAGS.feat_num)
    elif FLAGS.encoder_type==3:
        encoder = UNetEncoder()
    elif FLAGS.encoder_type==4:
        encoder = ResUnetEncoder()


    if FLAGS.decoder_type==1:
        decoder = Decoder(output_num=FLAGS.mpi_num)
    elif FLAGS.decoder_type==2:
        decoder = DecoderDepthwise(output_num=FLAGS.mpi_num)
    elif FLAGS.decoder_type==3:
        decoder = UNetDecoder()
    elif FLAGS.decoder_type==4:
        decoder = ResUnetDecoder()

    # jpeg
    jpeg_compress = DiffJPEG(height=FLAGS.image_height, width=FLAGS.image_width, differentiable=True, quality=FLAGS.jpeg_quality, quality_range=0)
    jpeg_compress = jpeg_compress.to(device)


    # color jitter
    if FLAGS.color_jitter:
        color_jitter=ColorJitter(FLAGS)

    # random crop and resize
    if FLAGS.random_crop_resize:
        random_crop_resize=RandomCropResize(FLAGS)

    # model
    util.load_network(encoder, 'encoder', FLAGS.which_epoch, FLAGS.which_iter, FLAGS.load_pretrain)
    util.load_network(decoder, 'decoder', FLAGS.which_epoch, FLAGS.which_iter, FLAGS.load_pretrain)
    encoder = nn.DataParallel(encoder)
    encoder = encoder.to(device)
    decoder = nn.DataParallel(decoder)
    decoder = decoder.to(device)
    print("encoder decoder initialized")

    transforms_rgb = transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5, 0.5)])

    #metrics lpips
    metric_fn_vgg=lpips.LPIPS(net='vgg')
    metric_fn_vgg=metric_fn_vgg.to(device)


    # visualizer
    visualizer = Visualizer(FLAGS)


    #scores
    cnt=0
    ref_lpips_accu=0.0
    ref_psnr_accu=0.0
    ref_ssim_accu=0.0

    with torch.no_grad():
        for idx , data in enumerate(dataloader):
            cnt+=1
            visualizer.vis_print('%04d'%(idx))
            # network processing
            ref, mpis = data
            ref = ref.to(device)
            mpis = [mpi.to(device) for mpi in mpis]

            embedding_img = encoder([ref, mpis])

            if FLAGS.jpeg_test:
                embedding_img_read,gt_embedding_img_read=image_save_load_jpeg(embedding_img,ref,FLAGS.name,quality=FLAGS.jpeg_quality)
            else:
                embedding_img_read,gt_embedding_img_read=image_save_load_png(embedding_img,ref,jpeg_compress,FLAGS.name,quality=FLAGS.jpeg_quality)

            embedding_img_read=embedding_img_read.to(device)
            embedding_img_read_for_loss=embedding_img_read.clone()
            gt_embedding_img_read=gt_embedding_img_read.to(device)

            # lpips metric
            ref_lpips_score=float(metric_fn_vgg(embedding_img_read,gt_embedding_img_read).detach()[0,0,0,0])
            ref_lpips_accu+=ref_lpips_score
            visualizer.vis_print('ref lpips: %f'%(ref_lpips_score))

            if FLAGS.color_jitter:
                embedding_img_read,mpis=color_jitter(embedding_img_read,mpis)
            if FLAGS.random_crop_resize:
                embedding_img_read,mpis=random_crop_resize(embedding_img_read,mpis)

            decoded_mpis = decoder(embedding_img_read)
            decoded_mpis = decoded_mpis.view(-1, FLAGS.mpi_num, 4, decoded_mpis.size(2), decoded_mpis.size(3))

            visual_list = []
            batch_idx = 0
            for mpi_idx in range(FLAGS.mpi_num):
                gt_mpi = util.tensor2image(mpis[mpi_idx][batch_idx], normalize=False)
                visual_list.append(('gt_mpi_%02d'%(mpi_idx), gt_mpi))

                pred_mpi = util.tensor2image(decoded_mpis[batch_idx, mpi_idx, ...])
                visual_list.append(('pred_mpi_%02d'%(mpi_idx), pred_mpi))

                compare_mpi = np.concatenate([gt_mpi, pred_mpi], axis=1)
                visual_list.append(('compare_mpi_%02d'%(mpi_idx), compare_mpi))

                gt_rgb = util.tensor2image(mpis[mpi_idx][batch_idx,0:3])
                visual_list.append(('gt_rgb_%02d'%(mpi_idx), gt_rgb))

                pred_rgb = util.tensor2image(decoded_mpis[batch_idx, mpi_idx, :3, ...])
                visual_list.append(('pred_rgb_%02d'%(mpi_idx), pred_rgb))

                compare_rgb = np.concatenate([gt_rgb, pred_rgb], axis=1)
                visual_list.append(('compare_rgb_%02d'%(mpi_idx), compare_rgb))

                gt_alpha = util.tensor2image(mpis[mpi_idx][batch_idx,3], normalize=False)
                visual_list.append(('gt_alpha_%02d'%(mpi_idx), gt_alpha))

                pred_alpha = util.tensor2image(decoded_mpis[batch_idx, mpi_idx, 3, ...])
                visual_list.append(('pred_alpha_%02d'%(mpi_idx), pred_alpha))

                compare_alpha = np.concatenate([gt_alpha, pred_alpha], axis=1)
                visual_list.append(('compare_alpha_%02d'%(mpi_idx), compare_alpha))
            gt_ref = util.tensor2image(gt_embedding_img_read[batch_idx])
            visual_list.append(('ref_gt', gt_ref))
            pred_ref = util.tensor2image(embedding_img_read_for_loss[batch_idx])
            visual_list.append(('ref_pred', pred_ref))
            compare_ref = np.concatenate([gt_ref, pred_ref], axis = 1)
            visual_list.append(('compare_ref', compare_ref))
            visuals = OrderedDict(visual_list)
            visualizer.save_images_test(idx, visuals)

            # metric

            ref_psnr_score=compare_psnr(pred_ref,gt_ref)
            ref_psnr_accu+=ref_psnr_score
            visualizer.vis_print('ref psnr:  %f'%(ref_psnr_score))

            ref_ssim_score=compare_ssim(pred_ref,gt_ref,multichannel=True)
            ref_ssim_accu+=ref_ssim_score
            visualizer.vis_print('ref ssim:  %f'%(ref_ssim_score))
            visualizer.vis_print('-'*50)

    ref_mean_lpips=ref_lpips_accu/cnt
    ref_mean_psnr=ref_psnr_accu/cnt
    ref_mean_ssim=ref_ssim_accu/cnt

    visualizer.vis_print('ref mean lpips: %f'%(ref_mean_lpips))
    visualizer.vis_print('ref mean psnr:  %f'%(ref_mean_psnr))
    visualizer.vis_print('ref mean ssim:  %f'%(ref_mean_ssim))
