import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import time
from PIL import Image
from collections import OrderedDict
import functools
import kornia
import copy

import util.util as util
from util.visualizer import Visualizer
from models.networks import Encoder, Decoder, MultiscaleDiscriminator, EncoderAlphaWeight, DecoderDepthwise
from models.unet_model import UNetEncoder,UNetDecoder
from models.resunet_model import ResUnetEncoder, ResUnetDecoder
from models.fft import image_fft
from opt import get_opts
from data.real_dataset import RealDataset
from models.losses import LossFunction
from DiffJPEG.DiffJPEG import DiffJPEG
from models.differentiable_quantize import DifferentiableQuantize
from color_jittering import ColorJitter
from random_crop_resize import RandomCropResize
from mpi_render import random_render
torch.autograd.set_detect_anomaly(True)

torch.manual_seed(123)

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


if __name__ == '__main__':
    FLAGS=get_opts()
    #print(FLAGS.isTrain)
    use_cuda=torch.cuda.is_available()
    device=torch.device("cuda" if use_cuda else "cpu")
    multi_gpu=False
    if torch.cuda.device_count() > 1:
        multi_gpu=True
        print('train on %d GPUs'%(torch.cuda.device_count()))


    os.makedirs(FLAGS.checkpoints_dir, exist_ok=True)

    # data
    print("begin loading dataset")
    batch_size=FLAGS.batch_size_each_gpu*torch.cuda.device_count()
    dataset=RealDataset(FLAGS)
    dataloader=DataLoader(dataset,
                          shuffle=True,
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

    ### for continue training
    if FLAGS.continue_train:
        util.load_network(encoder, 'encoder', FLAGS.which_epoch, FLAGS.which_iter, FLAGS.load_pretrain)
        util.load_network(decoder, 'decoder', FLAGS.which_epoch, FLAGS.which_iter, FLAGS.load_pretrain)
    if multi_gpu:
        encoder = nn.DataParallel(encoder)
    encoder = encoder.to(device)
    if multi_gpu:
        decoder = nn.DataParallel(decoder)
    decoder = decoder.to(device)
    print("encoder decoder initialized")



    # Discriminator
    if FLAGS.lambda_discriminator>0:

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
        D_hide_image = MultiscaleDiscriminator(input_nc=3, ndf=64, n_layers=3, norm_layer=norm_layer, num_D=3, getIntermFeat=False)


        print("discriminator initialized")
        if FLAGS.continue_train:
            util.load_network(D_hide_image, 'discriminator', FLAGS.which_epoch, FLAGS.which_iter, FLAGS.load_pretrain)
        if multi_gpu:
            D_hide_image = nn.DataParallel(D_hide_image)
        D_hide_image = D_hide_image.to(device)

    # quantization
    if FLAGS.quantize:
        quantize=DifferentiableQuantize()

    # jpeg compress
    if FLAGS.jpeg_compress:
        jpeg_compress = DiffJPEG(height=FLAGS.image_height, width=FLAGS.image_width, differentiable=True, quality=FLAGS.jpeg_quality, quality_range=FLAGS.jpeg_quality_range)
        jpeg_compress = jpeg_compress.to(device)

    # color jitter
    if FLAGS.color_jitter:
        color_jitter=ColorJitter(FLAGS)

    # random crop and resize
    if FLAGS.random_crop_resize:
        random_crop_resize=RandomCropResize(FLAGS)

    # loss
    loss_function = LossFunction(FLAGS)
    loss_function = loss_function.to(device)


    # optimizer
    optimizer_G = optim.Adam(list(encoder.parameters())+list(decoder.parameters()),lr=FLAGS.learning_rate, eps=1e-8)
    scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, milestones=FLAGS.decay_step, gamma=FLAGS.decay_gamma)
    if FLAGS.lambda_discriminator>0:
        optimizer_D = optim.Adam(D_hide_image.parameters(),lr=FLAGS.learning_rate, eps=1e-8)
        scheduler_D = optim.lr_scheduler.MultiStepLR(optimizer_D, milestones=FLAGS.decay_step, gamma=FLAGS.decay_gamma)

    # training
    epoch=FLAGS.start_epoch
    total_step=FLAGS.start_step

    # visualizer
    visualizer = Visualizer(FLAGS)

    # model save
    model_save_dir=os.path.join(FLAGS.checkpoints_dir, FLAGS.name, 'models')
    util.mkdirs(model_save_dir)

    while True:
        # epoch
        step=1

        VGG_ref_accu = 0
        L2_ref_accu = 0
        L2_mpi_rgb_accu = 0
        L2_mpi_alpha_accu = 0
        G_GAN_accu = 0
        D_real_accu = 0
        D_fake_accu = 0
        L2_render_accu = 0
        VGG_render_accu = 0
        fft_accu = 0
        reg_accu = 0



        for data in dataloader:
            if data is None:
                continue
            # network processing
            ref, mpis = data
            ref = ref.to(device)
            mpis = [mpi.to(device) for mpi in mpis]
            embedding_img = encoder([ref, mpis])
            embedding_img_for_loss=embedding_img
            
            # do color jittering and crop first
            if FLAGS.color_jitter:
                embedding_img,mpis=color_jitter(embedding_img,mpis)
            if FLAGS.random_crop_resize:
                embedding_img,mpis=random_crop_resize(embedding_img,mpis)
            
            # quantize the image to int 


            if FLAGS.quantize:
                # [-1,1] to [0,255]
                embedding_img_before = (embedding_img + 1) * 127.5

                # use reg loss to contrain the information loss in quantize
                # loss between embedding_img - embedding_img_before
                embedding_img_after = quantize(embedding_img_before)
                
                # [0,255] to [-1,1]
                embedding_img = (embedding_img_after - 127.5)/127.5

            ycbcr_after = 0.0
            ycbcr_diff = 0.0
            # jpeg compression
            if FLAGS.jpeg_compress:
                # [-1,1] to [0,1]
                embedding_img=(embedding_img+1)/2
                
                # use reg loss to contrain the information loss in quantize
                embedding_img, ycbcr_after, ycbcr_diff =jpeg_compress(embedding_img)
                # [0,1] to [-1,1]
                embedding_img=embedding_img*2-1
            
            
            decoded_mpis = decoder(embedding_img)

            # fft loss
            if FLAGS.lambda_fft>0:
                embedding_img_for_loss_fft=image_fft(embedding_img_for_loss)
                ref_fft=image_fft(ref)
            else:
                embedding_img_for_loss_fft=None
                ref_fft=None

            # resize the decoded mpis
            decoded_mpis = decoded_mpis.view(-1, FLAGS.mpi_num, 4, decoded_mpis.size(2), decoded_mpis.size(3))

            # random rander
            if FLAGS.lambda_l2_render_loss>0 or FLAGS.lambda_vgg_render_loss>0:
                decoded_mpis_render = decoded_mpis.clone()
                decoded_mpis_render[:,:,3,:,:] = (decoded_mpis_render[:,:,3,:,:]+1)/2
                mpis_cat=torch.cat([mpi.unsqueeze(1) for mpi in mpis],dim=1)
                decoded_random_render,gt_random_render=random_render(decoded_mpis_render, mpis_cat,max_translation=FLAGS.max_translation,\
                    max_rotation=FLAGS.max_rotation, mpi_num=FLAGS.mpi_num, fx=FLAGS.fx,fy=FLAGS.fy)
            else:
                decoded_random_render=None
                gt_random_render=None


            # Discriminator
            pred_feat_real=None
            pred_feat_fake=None
            pred_feat_fake_as_real=None
            if FLAGS.lambda_discriminator>0:
                pred_feat_real = D_hide_image(ref)
                pred_feat_fake = D_hide_image(embedding_img_for_loss.detach())
                pred_feat_fake_as_real = D_hide_image(embedding_img_for_loss)



            # loss
            
            loss_dict=loss_function([embedding_img_for_loss, ref, decoded_mpis, mpis, pred_feat_real, pred_feat_fake, pred_feat_fake_as_real,
                                     decoded_random_render,gt_random_render,embedding_img_for_loss_fft,ref_fft, \
                                     embedding_img_before, ycbcr_diff])
            loss_total_G = loss_dict['VGG_ref']+\
                    loss_dict['L2_ref']+\
                    loss_dict['L2_mpi_rgb']+\
                    loss_dict['L2_mpi_alpha'] + \
                    loss_dict['G_GAN'] + \
                    loss_dict['render_l2'] + \
                    loss_dict['render_vgg'] + \
                    loss_dict['fft'] 
            if FLAGS.use_reg:
                loss_total_G = loss_dict['reg']

            loss_total_D = loss_dict['D_real'] + loss_dict['D_fake']
            # update loss list

            VGG_ref_accu += loss_dict['VGG_ref']
            L2_ref_accu += loss_dict['L2_ref']
            L2_mpi_rgb_accu += loss_dict['L2_mpi_rgb']
            L2_mpi_alpha_accu += loss_dict['L2_mpi_alpha']
            G_GAN_accu += loss_dict['G_GAN']
            D_real_accu += loss_dict['D_real']
            D_fake_accu += loss_dict['D_fake']
            L2_render_accu += loss_dict['render_l2']
            VGG_render_accu += loss_dict['render_vgg']
            fft_accu +=loss_dict['fft']
            reg_accu += loss_dict['reg']


            # optmize
            optimizer_G.zero_grad()
            loss_total_G.backward()
            optimizer_G.step()

            # optimize D
            if FLAGS.lambda_discriminator>0:
                optimizer_D.zero_grad()
                loss_total_D.backward()
                optimizer_D.step()


            # print loss
            if step%FLAGS.print_freq == 0:

                VGG_ref_mean = VGG_ref_accu/FLAGS.print_freq
                L2_ref_mean = L2_ref_accu/FLAGS.print_freq
                L2_mpi_rgb_mean = L2_mpi_rgb_accu/FLAGS.print_freq
                L2_mpi_alpha_mean = L2_mpi_alpha_accu/FLAGS.print_freq
                G_GAN_mean = G_GAN_accu/FLAGS.print_freq
                D_real_mean = D_real_accu/FLAGS.print_freq
                D_fake_mean = D_fake_accu/FLAGS.print_freq
                L2_render_mean = L2_render_accu/FLAGS.print_freq
                VGG_render_mean = VGG_render_accu/FLAGS.print_freq
                fft_mean = fft_accu/FLAGS.print_freq
                reg_mean = reg_accu/FLAGS.print_freq

                visualizer.vis_print('-'*50)
                visualizer.vis_print(time.strftime("%c"))
                visualizer.vis_print('| Epoch %d | Step %d'%(epoch,step))
                visualizer.vis_print('| L2 ref loss : %.4f |  VGG ref loss : %.4f'%(L2_ref_mean, VGG_ref_mean))
                visualizer.vis_print('| L2 mpi rgb loss : %.4f | L2 mpi alpha loss : %.4f'%(L2_mpi_rgb_mean, L2_mpi_alpha_mean))
                visualizer.vis_print('| GAN : %.4f | D_real : %.4f | D_fake : %.4f'%(G_GAN_mean, D_real_mean, D_fake_mean))
                visualizer.vis_print('| L2 render : %.4f | VGG render : %.4f | FFT : %.4f | Reg: %.4f'%(L2_render_mean, VGG_render_mean, fft_mean, reg_mean))
                visualizer.vis_print('optimizer G LR:')
                for param_group in optimizer_G.param_groups:
                    visualizer.vis_print('LR: %f'%(param_group['lr']))
                if FLAGS.lambda_discriminator>0:
                    visualizer.vis_print('optimizer D LR:')
                    for param_group in optimizer_D.param_groups:
                        visualizer.vis_print('LR: %f'%(param_group['lr']))

                VGG_ref_accu = 0
                L2_ref_accu = 0
                VGG_mpi_accu = 0
                L2_mpi_rgb_accu = 0
                L2_mpi_alpha_accu = 0
                G_GAN_accu = 0
                D_real_accu = 0
                D_fake_accu = 0
                L2_render_accu = 0
                VGG_render_accu = 0
                fft_accu = 0

            # save training results
            if FLAGS.save_training_results and total_step%FLAGS.display_freq==0:
                visual_list = []
                for batch_idx in range(ref.size(0)):
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
                    gt_ref = util.tensor2image(ref[batch_idx])
                    visual_list.append(('ref_gt', gt_ref))
                    pred_ref = util.tensor2image(embedding_img_for_loss[batch_idx])
                    visual_list.append(('ref_pred', pred_ref))
                    pred_ref_color_jitter = util.tensor2image(embedding_img[batch_idx])
                    visual_list.append(('ref_pred_color_jitter', pred_ref_color_jitter))
                    compare_ref = np.concatenate([gt_ref, pred_ref], axis = 1)
                    visual_list.append(('compare_ref', compare_ref))
                    if FLAGS.lambda_l2_render_loss>0 or FLAGS.lambda_vgg_render_loss>0:
                        gt_render = util.tensor2image(gt_random_render[batch_idx])
                        visual_list.append(('gt_random_render', gt_render))
                        pred_render = util.tensor2image(decoded_random_render[batch_idx])
                        visual_list.append(('pred_random_render', pred_render))
                        compare_render = np.concatenate([gt_render, pred_render], axis = 1)
                        visual_list.append(('compare_render', compare_render))
                    visuals = OrderedDict(visual_list)
                    visualizer.save_images('step_%d_epoch_%d_batch_%d'%(total_step,epoch,batch_idx),visuals)


            # save model
            if total_step % FLAGS.model_save_freq == 0:
                visualizer.vis_print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_step))
                encoder_sasve_path=os.path.join(model_save_dir,'encoder_%03d_%08d.pth'%(epoch,total_step))
                util.save_network(encoder, encoder_sasve_path)
                decoder_sasve_path=os.path.join(model_save_dir,'decoder_%03d_%08d.pth'%(epoch,total_step))
                util.save_network(decoder, decoder_sasve_path)
                if FLAGS.lambda_discriminator>0:
                    discriminator_sasve_path=os.path.join(model_save_dir,'discriminator_%03d_%08d.pth'%(epoch,total_step))
                    util.save_network(D_hide_image, discriminator_sasve_path)


            step+=1
            total_step+=1
            if FLAGS.max_steps>0 and total_step>FLAGS.max_steps>0:
                break
        epoch+=1
        scheduler_G.step()
        if FLAGS.lambda_discriminator>0:
            scheduler_D.step()
        if FLAGS.max_epoches>0 and epoch>FLAGS.max_epoches>0:
            break
