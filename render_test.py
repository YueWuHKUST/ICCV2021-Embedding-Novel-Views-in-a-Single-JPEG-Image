import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import glob
from PIL import Image
import os
from scipy.spatial.transform import Rotation as R
from mpi_render import load_single_img,inv_depths,mpi_render_view,load_mpi
import cv2
import numpy as np
import lpips
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
# from skimage.measure import compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
# from skimage.measure import compare_psnr
# from skimage.measure import compare_ssim




@ torch.no_grad()
def render(mpis_concat,intrinsics,mpi_planes,tx,ty,tz,rx,ry,rz,device):
    translation_list=torch.tensor([tx,ty,tz]).float()
    rotation_list=torch.tensor([rz,ry,rx]).float()

    pose = torch.zeros((4,4)).unsqueeze(0).float()
    pose[0,:3,:3]=torch.from_numpy(R.from_euler('zyx', rotation_list, degrees=True).as_matrix())
    pose[0,:3,3]=translation_list
    pose[0,3,3]=1

    mpis_concat=mpis_concat.to(device)

    render_img=mpi_render_view(mpis_concat, pose, mpi_planes, intrinsics).detach().cpu()

    img_np=np.uint8(np.clip((render_img[0]*127.5+127.5),0,255))
    img_pil = Image.fromarray(img_np)
    return img_pil,img_np





@ torch.no_grad()
def metric(pred_np,gt_np,lpips_fn,log_f):
    pred_tensor=torch.from_numpy(pred_img_np).float().permute(2,0,1).unsqueeze(0)
    pred_tensor=pred_tensor/127.5-1
    gt_tensor=torch.from_numpy(gt_img_np).float().permute(2,0,1).unsqueeze(0)
    gt_tensor=gt_tensor/127.5-1
    pred_tensor=pred_tensor.to(device)
    gt_tensor=gt_tensor.to(device)
    lpips_score=lpips_fn(pred_tensor,gt_tensor).detach().cpu()[0,0,0,0]
    log_f.write('lpips: %f\n'%(lpips_score))

    psnr_score=compare_psnr(pred_np,gt_np)
    log_f.write('psnr:  %f\n'%(psnr_score))

    ssim_score=compare_ssim(pred_np,gt_np,multichannel=True)
    log_f.write('ssim:  %f\n'%(ssim_score))
    log_f.write('-'*50+'\n')
    return lpips_score,psnr_score,ssim_score





if __name__ == "__main__":
    
    use_cuda=torch.cuda.is_available()
    device=torch.device("cuda" if use_cuda else "cpu")

    flags = argparse.ArgumentParser(description='Argument for mpi rendering')

    # file
    flags.add_argument('--root_dir', type=str, default='', help='the root dir of mpis')
    flags.add_argument('--mpi_prefix_predict', type=str, default='pred_mpi_', help='file name prefix of predicted mpis')
    flags.add_argument('--mpi_extension_predict', type=str, default='.png', help='file extension of predicted mpis')
    flags.add_argument('--mpi_prefix_gt', type=str, default='gt_mpi_', help='file name prefix of predicted mpis')
    flags.add_argument('--mpi_extension_gt', type=str, default='.png', help='file extension of ground truth mpis')
    flags.add_argument('--output_dir', type=str, default='.', help='the root dir of output')
    flags.add_argument('--save_img', action='store_true',help='If save the image')


    # camera and mpi parameters
    flags.add_argument('--fx', type=float, default=0.486242434,help='Focal length as a fraction of image width.')
    flags.add_argument('--fy', type=float, default=0.864430976,help='Focal length as a fraction of image height.')
    flags.add_argument('--min_depth', type=float, default=1,help='Minimum scene depth.')
    flags.add_argument('--max_depth', type=float, default=100,help='Maximum scene depth.')
    flags.add_argument('--toffset', type=float, default=0.017,help='translation stride')
    flags.add_argument('--roffset', type=float, default=1,help='rotation stride')
    flags.add_argument('--num_mpi_planes', type=int, default=32,help='Number of MPI planes to infer.')


    # render
    flags.add_argument('--translation_render_max_multiples', type=float, default=4.0,help='Multiples of input translation offset to render outputs at.')
    flags.add_argument('--translation_render_stride', type=float, default=2.0,help='stride of translation Multiples')
    flags.add_argument('--rotation_render_max_multiples', type=float, default=12,help='Multiples of input rotation offset to render outputs at.')
    flags.add_argument('--rotation_render_stride', type=float, default=1,help='stride of rotation Multiples')

    flags=flags.parse_args()


    # render parameter
    translation_num=int(flags.translation_render_max_multiples*2/flags.translation_render_stride+1)
    translation_render_list = list(flags.toffset*np.linspace(-flags.translation_render_max_multiples,flags.translation_render_max_multiples,translation_num))
    rotation_num=int(flags.rotation_render_max_multiples*2/flags.rotation_render_stride+1)
    rotation_render_list = list(flags.roffset*np.linspace(-flags.rotation_render_max_multiples,flags.rotation_render_max_multiples,rotation_num))

    if len(translation_render_list)==1:
        translation_render_list=[]
    if len(rotation_render_list)==1:
        rotation_render_list=[]

    print('translation render list:',translation_render_list)
    print('rotation render list:',rotation_render_list)


    # path
    dir_list=sorted(glob.glob(os.path.join(flags.root_dir,'*')))

    # intrinsics
    print(dir_list)
    img_sample_path=glob.glob(os.path.join(dir_list[0],'%s*%s'%(flags.mpi_prefix_predict,flags.mpi_extension_predict)))[0]
    img_sample=Image.open(img_sample_path)
    img_sample_w,img_sample_h=img_sample.size

    fx = img_sample_w*flags.fx
    fy = img_sample_h*flags.fy
    cx = img_sample_w*0.5
    cy = img_sample_h*0.5
    intrinsics = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy],
                             [0.0, 0.0, 1.0]]).unsqueeze(0)

    # mpi depths
    mpi_planes=inv_depths(flags.min_depth,flags.max_depth,flags.num_mpi_planes)

    #metrics lpips
    metric_fn_vgg=lpips.LPIPS(net='vgg')
    metric_fn_vgg=metric_fn_vgg.to(device)

    if not os.path.exists(flags.output_dir):
        os.makedirs(flags.output_dir)
    log_path=os.path.join(flags.output_dir,'render_metric.txt')


    lpips_accu=0.0
    psnr_accu=0.0
    ssim_accu=0.0
    cnt=0.0



    with open(log_path,'a') as log_f:
        for mpi_dir in dir_list:
            mpi_name=mpi_dir.split('/')[-1]
            output_dir=os.path.join(flags.output_dir,mpi_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            pred_mpis_concat,gt_mpis_concat=load_mpi(mpi_dir,mpi_prefix_predict=flags.mpi_prefix_predict,
                                                            mpi_extension_predict=flags.mpi_extension_predict,
                                                            mpi_prefix_gt=flags.mpi_prefix_gt,
                                                            mpi_extension_gt=flags.mpi_extension_gt)
            log_f.write('='*50+'\n')
            log_f.write('%s\n'%(mpi_name))
            log_f.write('='*50+'\n')
            print(mpi_name)

            if not (0.0 in translation_render_list):
                translation_render_list.append(0.0)

            # tx
            for tx in translation_render_list:
                pred_img_pil,pred_img_np=render(pred_mpis_concat,intrinsics,mpi_planes,tx,0,0,0,0,0,device)
                gt_img_pil,gt_img_np=render(gt_mpis_concat,intrinsics,mpi_planes,tx,0,0,0,0,0,device)
                compare_image_pil=Image.fromarray(np.concatenate([gt_img_np,pred_img_np],axis=1))
                log_f.write('translation x %f\n'%(tx))
                lpips_score,psnr_score,ssim_score=metric(pred_img_np,gt_img_np,metric_fn_vgg,log_f)
                print('tx %f'%(tx))
                print('lpips: %f'%(lpips_score))
                print('psnr:  %f'%(psnr_score))
                print('ssim:  %f'%(ssim_score))
                print('-'*50)
                lpips_accu+=lpips_score
                psnr_accu+=psnr_score
                ssim_accu+=ssim_score
                cnt+=1
                if flags.save_img:
                    pred_img_pil.save(os.path.join(output_dir,'%stx%f.png'%(flags.mpi_prefix_predict,tx)))
                    gt_img_pil.save(os.path.join(output_dir,'%stx%f.png'%(flags.mpi_prefix_gt,tx)))
                    # compare_image_pil.save(os.path.join(output_dir,'%stx%f.png'%('compare_',tx)))

            # remove 0 from render list
            if len(translation_render_list)>0:
                translation_render_list.remove(0.0)




            # ty
            for ty in translation_render_list:
                pred_img_pil,pred_img_np=render(pred_mpis_concat,intrinsics,mpi_planes,0,ty,0,0,0,0,device)
                gt_img_pil,gt_img_np=render(gt_mpis_concat,intrinsics,mpi_planes,0,ty,0,0,0,0,device)
                compare_image_pil=Image.fromarray(np.concatenate([gt_img_np,pred_img_np],axis=1))
                log_f.write('translation y %f\n'%(ty))
                lpips_score,psnr_score,ssim_score=metric(pred_img_np,gt_img_np,metric_fn_vgg,log_f)
                print('ty %f'%(ty))
                print('lpips: %f'%(lpips_score))
                print('psnr:  %f'%(psnr_score))
                print('ssim:  %f'%(ssim_score))
                print('-'*50)
                lpips_accu+=lpips_score
                psnr_accu+=psnr_score
                ssim_accu+=ssim_score
                cnt+=1
                if flags.save_img:
                    pred_img_pil.save(os.path.join(output_dir,'%sty%f.png'%(flags.mpi_prefix_predict,ty)))
                    gt_img_pil.save(os.path.join(output_dir,'%sty%f.png'%(flags.mpi_prefix_gt,ty)))
                    # compare_image_pil.save(os.path.join(output_dir,'%sty%f.png'%('compare_',ty)))

            '''
            # tz
            for tz in translation_render_list:
                pred_img_pil,pred_img_np=render(pred_mpis_concat,intrinsics,mpi_planes,0,0,tz,0,0,0,device)
                gt_img_pil,gt_img_np=render(gt_mpis_concat,intrinsics,mpi_planes,0,0,tz,0,0,0,device)
                compare_image_pil=Image.fromarray(np.concatenate([gt_img_np,pred_img_np],axis=1))
                log_f.write('translation z %f\n'%(tz))
                lpips_score,psnr_score,ssim_score=metric(pred_img_np,gt_img_np,metric_fn_vgg,log_f)
                print('tz %f'%(tz))
                print('lpips: %f'%(lpips_score))
                print('psnr:  %f'%(psnr_score))
                print('ssim:  %f'%(ssim_score))
                print('-'*50)
                lpips_accu+=lpips_score
                psnr_accu+=psnr_score
                ssim_accu+=ssim_score
                cnt+=1
                if flags.save_img:
                    pred_img_pil.save(os.path.join(output_dir,'%stz%f.png'%(flags.mpi_prefix_predict,tz)))
                    gt_img_pil.save(os.path.join(output_dir,'%stz%f.png'%(flags.mpi_prefix_gt,tz)))
                    # compare_image_pil.save(os.path.join(output_dir,'%stz%f.png'%('compare_',tz)))
            '''

            if (not (0.0 in rotation_render_list)) and len(rotation_render_list)>0:
                rotation_render_list.append(0.0)

            # rx
            for rx in rotation_render_list:
                pred_img_pil,pred_img_np=render(pred_mpis_concat,intrinsics,mpi_planes,0,0,0,rx,0,0,device)
                gt_img_pil,gt_img_np=render(gt_mpis_concat,intrinsics,mpi_planes,0,0,0,rx,0,0,device)
                compare_image_pil=Image.fromarray(np.concatenate([gt_img_np,pred_img_np],axis=1))
                log_f.write('rotation x %f\n'%(rx))
                lpips_score,psnr_score,ssim_score=metric(pred_img_np,gt_img_np,metric_fn_vgg,log_f)
                print('rx %f'%(rx))
                print('lpips: %f'%(lpips_score))
                print('psnr:  %f'%(psnr_score))
                print('ssim:  %f'%(ssim_score))
                print('-'*50)
                lpips_accu+=lpips_score
                psnr_accu+=psnr_score
                ssim_accu+=ssim_score
                cnt+=1
                if flags.save_img:
                    pred_img_pil.save(os.path.join(output_dir,'%srx%f.png'%(flags.mpi_prefix_predict,rx)))
                    gt_img_pil.save(os.path.join(output_dir,'%srx%f.png'%(flags.mpi_prefix_gt,rx)))
                    # compare_image_pil.save(os.path.join(output_dir,'%srx%f.png'%('compare_',rx)))

            # remove 0 from render list
            if len(rotation_render_list)>0:
                rotation_render_list.remove(0.0)


            # ry
            for ry in rotation_render_list:
                pred_img_pil,pred_img_np=render(pred_mpis_concat,intrinsics,mpi_planes,0,0,0,0,ry,0,device)
                gt_img_pil,gt_img_np=render(gt_mpis_concat,intrinsics,mpi_planes,0,0,0,0,ry,0,device)
                compare_image_pil=Image.fromarray(np.concatenate([gt_img_np,pred_img_np],axis=1))
                log_f.write('rotation y %f\n'%(ry))
                lpips_score,psnr_score,ssim_score=metric(pred_img_np,gt_img_np,metric_fn_vgg,log_f)
                print('ry %f'%(ry))
                print('lpips: %f'%(lpips_score))
                print('psnr:  %f'%(psnr_score))
                print('ssim:  %f'%(ssim_score))
                print('-'*50)
                lpips_accu+=lpips_score
                psnr_accu+=psnr_score
                ssim_accu+=ssim_score
                cnt+=1
                if flags.save_img:
                    pred_img_pil.save(os.path.join(output_dir,'%sry%f.png'%(flags.mpi_prefix_predict,ry)))
                    gt_img_pil.save(os.path.join(output_dir,'%sry%f.png'%(flags.mpi_prefix_gt,ry)))
                    # compare_image_pil.save(os.path.join(output_dir,'%sry%f.png'%('compare_',ry)))


            # rz
            for rz in rotation_render_list:
                pred_img_pil,pred_img_np=render(pred_mpis_concat,intrinsics,mpi_planes,0,0,0,0,0,rz,device)
                gt_img_pil,gt_img_np=render(gt_mpis_concat,intrinsics,mpi_planes,0,0,0,0,0,rz,device)
                compare_image_pil=Image.fromarray(np.concatenate([gt_img_np,pred_img_np],axis=1))
                log_f.write('rotation z %f\n'%(rz))
                lpips_score,psnr_score,ssim_score=metric(pred_img_np,gt_img_np,metric_fn_vgg,log_f)
                print('rz %f'%(rz))
                print('lpips: %f'%(lpips_score))
                print('psnr:  %f'%(psnr_score))
                print('ssim:  %f'%(ssim_score))
                print('-'*50)
                lpips_accu+=lpips_score
                psnr_accu+=psnr_score
                ssim_accu+=ssim_score
                cnt+=1
                if flags.save_img:
                    pred_img_pil.save(os.path.join(output_dir,'%srz%f.png'%(flags.mpi_prefix_predict,rz)))
                    gt_img_pil.save(os.path.join(output_dir,'%srz%f.png'%(flags.mpi_prefix_gt,rz)))
                    # compare_image_pil.save(os.path.join(output_dir,'%srz%f.png'%('compare_',rz)))

        lpips_mean=lpips_accu/cnt
        psnr_mean=psnr_accu/cnt
        ssim_mean=ssim_accu/cnt
        log_f.write('mean lpips: %f\n'%(lpips_mean))
        log_f.write('mean psnr:  %f\n'%(psnr_mean))
        log_f.write('mean ssim:  %f\n'%(ssim_mean))
        print('mean lpips: %f'%(lpips_mean))
        print('mean psnr:  %f'%(psnr_mean))
        print('mean ssim:  %f'%(ssim_mean))