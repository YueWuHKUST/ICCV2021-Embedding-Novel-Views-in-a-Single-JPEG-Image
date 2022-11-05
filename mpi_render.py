import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import glob
from PIL import Image
import os
from scipy.spatial.transform import Rotation as R


def divide_safe(num, den):
    eps = 1e-8
    den[torch.where(den==0)]=den[torch.where(den==0)]+eps
    return num/den


def inv_homography(k_s, k_t, rot, t, n_hat, a):
    """Computes inverse homography matrix between two cameras via a plane.

    Args:
        k_s: intrinsics for source cameras, [..., 3, 3] matrices
        k_t: intrinsics for target cameras, [..., 3, 3] matrices
        rot: relative rotations between source and target, [..., 3, 3] matrices
        t: [..., 3, 1], translations from source to target camera. Mapping a 3D
        point p from source to target is accomplished via rot * p + t.
        n_hat: [..., 1, 3], plane normal w.r.t source camera frame
        a: [..., 1, 1], plane equation displacement
    Returns:
        homography: [..., 3, 3] inverse homography matrices (homographies mapping
        pixel coordinates from target to source).
    """
    rot_t=torch.transpose(rot,-2,-1)
    k_t_inv =torch.inverse(k_t)
    denom = a - torch.matmul(torch.matmul(n_hat, rot_t), t)
    numerator = torch.matmul(torch.matmul(torch.matmul(rot_t, t), n_hat), rot_t)
    inv_hom = torch.matmul(
        torch.matmul(k_s, rot_t + divide_safe(numerator, denom)),k_t_inv)
    return inv_hom



def transform_points(points,homography):
    """Transforms input points according to homography.

    Args:
        points: [..., H, W, 3]; pixel (u,v,1) coordinates.
        homography: [..., 3, 3]; desired matrix transformation
    Returns:
        output_points: [..., H, W, 3]; transformed (u,v,w) coordinates.
    """

    # Because the points have two additional dimensions as they vary across the
    # width and height of an image, we need to reshape to multiply by the
    # per-image homographies.
    points_orig_shape = list(points.size())
    points_reshaped_shape = list(homography.size())
    points_reshaped_shape[-2] = -1

    points_reshaped = points.view(points_reshaped_shape)
    transformed_points = torch.matmul(points_reshaped, torch.transpose(homography,-2,-1))
    transformed_points = transformed_points.view(points_orig_shape)
    return transformed_points




def normalize_homogeneous(points):
    """Converts homogeneous coordinates to regular coordinates.

    Args:
        points: [..., n_dims_coords+1]; points in homogeneous coordinates.
    Returns:
        points_uv_norm: [..., n_dims_coords];
            points in standard coordinates after dividing by the last entry.
    """
    uv = points[..., :-1]
    w = points[..., -1].unsqueeze(-1)
    return divide_safe(uv, w)



def transform_plane_imgs(imgs, pixel_coords_trg, k_s, k_t, rot, t, n_hat, a):
    """Transforms input imgs via homographies for corresponding planes.

    Args:
    imgs: are [..., H_s, W_s, C]
    pixel_coords_trg: [..., H_t, W_t, 3]; pixel (u,v,1) coordinates.
    k_s: intrinsics for source cameras, [..., 3, 3] matrices
    k_t: intrinsics for target cameras, [..., 3, 3] matrices
    rot: relative rotation, [..., 3, 3] matrices
    t: [..., 3, 1], translations from source to target camera
    n_hat: [..., 1, 3], plane normal w.r.t source camera frame
    a: [..., 1, 1], plane equation displacement
    Returns:
    [..., H_t, W_t, C] images after bilinear sampling from input.
        Coordinates outside the image are sampled as 0.
    """

    hom_t2s_planes = inv_homography(k_s, k_t, rot, t, n_hat, a)
    pixel_coords_t2s = transform_points(pixel_coords_trg, hom_t2s_planes)
    pixel_coords_t2s = normalize_homogeneous(pixel_coords_t2s)
    imgs_s2t = bilinear_wrapper(imgs, pixel_coords_t2s)

    return imgs_s2t




def grid_sample(imgs, coords):
    # todo
    _,coords_h,coords_w,_=coords.size()

    coords[:,:,:,0]=coords[:,:,:,0]/(coords_w-1)*2-1
    coords[:,:,:,1]=coords[:,:,:,1]/(coords_h-1)*2-1

    imgs=imgs.permute(0,3,1,2)
    imgs_sampled=F.grid_sample(imgs, coords)
    imgs_sampled=imgs_sampled.permute(0,2,3,1)

    return imgs_sampled



def bilinear_wrapper(imgs, coords):
    """Wrapper around bilinear sampling function, handles arbitrary input sizes.

    Args:
    imgs: [..., H_s, W_s, C] images to resample
    coords: [..., H_t, W_t, 2], source pixel locations from which to copy
    Returns:
    [..., H_t, W_t, C] images after bilinear sampling from input.
    """
    # The bilinear sampling code only handles 4D input, so we'll need to reshape.
    init_dims = list(imgs.size())[:-3:]
    end_dims_img = list(imgs.size())[-3::]
    end_dims_coords = list(coords.size())[-3::]
    prod_init_dims = init_dims[0]
    for ix in range(1, len(init_dims)):
        prod_init_dims *= init_dims[ix]
    imgs=imgs.contiguous()
    imgs = imgs.view([prod_init_dims] + end_dims_img)
    coords = coords.view([prod_init_dims] + end_dims_coords)
    imgs_sampled = grid_sample(imgs, coords)
    imgs_sampled =imgs_sampled.view(init_dims + list(imgs_sampled.size())[-3::])
    return imgs_sampled





def planar_transform(imgs, pixel_coords_trg, k_s, k_t, rot, t, n_hat, a):
    """Transforms imgs, masks and computes dmaps according to planar transform.

    Args:
    imgs: are [L, B, H, W, C], typically RGB images per layer
    pixel_coords_trg: tensors with shape [B, H_t, W_t, 3];
        pixel (u,v,1) coordinates of target image pixels. (typically meshgrid)
    k_s: intrinsics for source cameras, [B, 3, 3] matrices
    k_t: intrinsics for target cameras, [B, 3, 3] matrices
    rot: relative rotation, [B, 3, 3] matrices
    t: [B, 3, 1] matrices, translations from source to target camera
        (R*p_src + t = p_tgt)
    n_hat: [L, B, 1, 3] matrices, plane normal w.r.t source camera frame
        (typically [0 0 1])
    a: [L, B, 1, 1] matrices, plane equation displacement
        (n_hat * p_src + a = 0)
    Returns:
    imgs_transformed: [L, ..., C] images in trg frame
    Assumes the first dimension corresponds to layers.
    """
    n_layers = list(imgs.size())[0]
    rot_rep_dims = [n_layers]
    rot_rep_dims += [1 for _ in range(len(k_s.size()))]

    cds_rep_dims = [n_layers]
    cds_rep_dims += [1 for _ in range(len(pixel_coords_trg.size()))]

    k_s = k_s.unsqueeze(0).repeat(rot_rep_dims)
    k_t = k_t.unsqueeze(0).repeat(rot_rep_dims)
    t = t.unsqueeze(0).repeat(rot_rep_dims)
    rot = rot.unsqueeze(0).repeat(rot_rep_dims)
    pixel_coords_trg = pixel_coords_trg.unsqueeze(0).repeat(cds_rep_dims)

    imgs_trg = transform_plane_imgs(imgs, pixel_coords_trg, k_s, k_t, rot, t, n_hat, a)
    return imgs_trg




def meshgrid_abs(batch, height, width, is_homogeneous=True):
    """Construct a 2D meshgrid in the absolute coordinates.

    Args:
        batch: batch size
        height: height of the grid
        width: width of the grid
        is_homogeneous: whether to return in homogeneous coordinates
    Returns:
        x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
    """
    xs = torch.linspace(0.0, width-1, width)
    ys = torch.linspace(0.0, height-1, height)
    ys, xs = torch.meshgrid(ys, xs)

    if is_homogeneous:
        ones = torch.ones_like(xs)
        coords = torch.stack([xs, ys, ones], dim=0)
    else:
        coords = torch.stack([xs, ys], dim=0)
    coords = coords.unsqueeze(0).repeat([batch, 1, 1, 1])
    return coords




def projective_forward_homography(src_images, intrinsics, pose, depths):
    """Use homography for forward warping.

    Args:
    src_images: [layers, batch, height, width, channels]
    intrinsics: [batch, 3, 3]
    pose: [batch, 4, 4]
    depths: [layers, batch]
    Returns:
    proj_src_images: [layers, batch, height, width, channels]
    """
    n_layers, n_batch, height, width, _ = list(src_images.size())
    # Format for planar_transform code:
    # rot: relative rotation, [..., 3, 3] matrices
    # t: [B, 3, 1], translations from source to target camera (R*p_s + t = p_t)
    # n_hat: [L, B, 1, 3], plane normal w.r.t source camera frame [0,0,1]
    #        in our case
    # a: [L, B, 1, 1], plane equation displacement (n_hat * p_src + a = 0)
    rot = pose[:, :3, :3]
    t = pose[:, :3, 3:]
    n_hat = torch.tensor([0,0,1]).view(1,1,1,3).to(src_images.device)
    n_hat = n_hat.repeat([n_layers, n_batch, 1, 1]).float()
    a = -depths.view([n_layers, n_batch, 1, 1])
    k_s = intrinsics
    k_t = intrinsics
    pixel_coords_trg = meshgrid_abs(n_batch, height, width).permute([0, 2, 3, 1]).to(src_images.device)
    proj_src_images = planar_transform(src_images, pixel_coords_trg, k_s, k_t, rot, t, n_hat, a)
    return proj_src_images




def over_composite(rgbas):
    """Combines a list of RGBA images using the over operation.

    Combines RGBA images from back to front with the over operation.
    The alpha image of the first image is ignored and assumed to be 1.0.

    Args:
    rgbas: A list of [batch, H, W, 4] RGBA images, combined from back to front.
    Returns:
    Composited RGB image.
    """
    for i in range(len(rgbas)):
        rgb = rgbas[i][:, :, :, 0:3]
        alpha = rgbas[i][:, :, :, 3:]
        if i == 0:
            output = rgb
        else:
            rgb_by_alpha = rgb * alpha
            output = rgb_by_alpha + output * (1.0 - alpha)
    return output





def mpi_render_view(rgba_layers, tgt_pose, planes, intrinsics):
    """Render a target view from an MPI representation.

    Args:
    rgba_layers: input MPI [batch, height, width, #planes, 4]
    tgt_pose: target pose to render from [batch, 4, 4]
    planes: list of depth for each plane
    intrinsics: camera intrinsics [batch, 3, 3]
    Returns:
    rendered view [batch, height, width, 3]
    """
    intrinsics=intrinsics.to(rgba_layers.device)
    tgt_pose=tgt_pose.to(rgba_layers.device)
    batch_size, _, _ = list(tgt_pose.size())
    depths = torch.tensor(planes).view([len(planes), 1]).to(rgba_layers.device)
    depths = depths.repeat([1, batch_size])
    depths=depths.to(rgba_layers.device)
    rgba_layers = rgba_layers.permute([3, 0, 1, 2, 4])
    proj_images = projective_forward_homography(rgba_layers, intrinsics,tgt_pose, depths)
    proj_images_list = []
    for i in range(len(planes)):
        proj_images_list.append(proj_images[i])
    output_image = over_composite(proj_images_list)
    return output_image





def inv_depths(start_depth, end_depth, num_depths):
    """Sample reversed, sorted inverse depths between a near and far plane.

    Args:
        start_depth: The first depth (i.e. near plane distance).
        end_depth: The last depth (i.e. far plane distance).
        num_depths: The total number of depths to create. start_depth and
            end_depth are always included and other depths are sampled
            between them uniformly according to inverse depth.
    Returns:
        The depths sorted in descending order (so furthest first). This order is
        useful for back to front compositing.
    """
    inv_start_depth = 1.0 / start_depth
    inv_end_depth = 1.0 / end_depth
    depths = [start_depth, end_depth]
    for i in range(1, num_depths - 1):
        fraction = float(i) / float(num_depths - 1)
        inv_depth = inv_start_depth + (inv_end_depth - inv_start_depth) * fraction
        depths.append(1.0 / inv_depth)
    depths = sorted(depths)
    return depths[::-1]



def load_single_img(path):
    img = np.array(Image.open(path))
    img = img.astype(np.float32)

    img[:, :, :3] = (img[:, :, :3] / 255.0) * 2.0 - 1.0
    img[:, :, 3:] = img[:, :, 3:] / 255.0

    img = img.astype(np.float32)
    converted = torch.from_numpy(img).unsqueeze(0)
    return converted





def random_render(pred_mpis,gt_mpis,max_translation=0.5,max_rotation=8,min_depth=1,max_depth=100,mpi_num=32,fx=0.5,fy=0.5):
    # reshape the input mpis
    gt_mpis=gt_mpis.permute(0,3,4,1,2)
    pred_mpis=pred_mpis.permute(0,3,4,1,2)

    # random pose
    pose_list=[]
    for _ in range(gt_mpis.size(0)):
        random_rotation=[np.random.uniform(-max_rotation,max_rotation) for _ in range(3)]
        translation_list=[np.random.uniform(-max_translation,max_translation) for _ in range(3)]
        pose = torch.zeros((4,4)).float()
        pose[:3,:3]=torch.from_numpy(R.from_euler('zyx', random_rotation, degrees=True).as_matrix())
        pose[:3,3]=torch.tensor(translation_list)
        pose=pose.unsqueeze(0)
        pose_list.append(pose)
    poses=torch.cat(pose_list,dim=0)

    # depth
    mpi_planes=inv_depths(min_depth,max_depth,mpi_num)

    # intrinsics
    img_h=gt_mpis.size(1)
    img_w=gt_mpis.size(2)
    fx = img_w*fx
    fy = img_h*fy
    cx = img_w*0.5
    cy = img_h*0.5
    intrinsics = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy],
                             [0.0, 0.0, 1.0]]).unsqueeze(0).repeat(gt_mpis.size(0),1,1)

    # render
    gt_render_img=mpi_render_view(gt_mpis, poses, mpi_planes, intrinsics)
    gt_render_img=gt_render_img.permute(0,3,1,2)
    pred_render_img=mpi_render_view(pred_mpis, poses, mpi_planes, intrinsics)
    pred_render_img=pred_render_img.permute(0,3,1,2)

    return pred_render_img,gt_render_img



def render_from_pose(input_mpis,pose,min_depth=1,max_depth=100,mpi_num=32,fx=0.5,fy=0.5):
    '''
    input_mpi shape: B,mpi_num,4,h,w
    output shape: B, h, w, 3
    pose shape: B, 4, 4
    '''
    # reshape the input mpis
    input_mpis=input_mpis.permute(0,3,4,1,2)

    # depth
    mpi_planes=inv_depths(min_depth,max_depth,mpi_num)

    # intrinsics
    img_h=input_mpis.size(1)
    img_w=input_mpis.size(2)
    fx = img_w*fx
    fy = img_h*fy
    cx = img_w*0.5
    cy = img_h*0.5
    intrinsics = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy],
                             [0.0, 0.0, 1.0]]).unsqueeze(0).repeat(input_mpis.size(0),1,1)

    # render
    render_img=mpi_render_view(input_mpis, pose, mpi_planes, intrinsics)

    return render_img









def load_mpi(mpi_dir,mpi_prefix_predict='pred_mpi',mpi_extension_predict='.png',mpi_prefix_gt='gt_mpi',mpi_extension_gt='.png'):
    predict_mpi_paths=sorted(glob.glob(os.path.join(mpi_dir,'%s*%s'%(mpi_prefix_predict,mpi_extension_predict))))
    gt_mpi_paths=sorted(glob.glob(os.path.join(mpi_dir,'%s*%s'%(mpi_prefix_gt,mpi_extension_gt))))

    pred_mpis_list = []
    for mpi_path in predict_mpi_paths:
        pred_mpis_list.append(load_single_img(mpi_path).unsqueeze(3))
    pred_mpis_concat = torch.cat(pred_mpis_list, dim=3)
    gt_mpis_list = []
    for mpi_path in gt_mpi_paths:
        gt_mpis_list.append(load_single_img(mpi_path).unsqueeze(3))
    gt_mpis_concat = torch.cat(gt_mpis_list, dim=3)
    return pred_mpis_concat,gt_mpis_concat





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


    # camera and mpi parameters
    flags.add_argument('--fx', type=float, default=0.5,help='Focal length as a fraction of image width.')
    flags.add_argument('--fy', type=float, default=0.5,help='Focal length as a fraction of image height.')
    flags.add_argument('--min_depth', type=float, default=1,help='Minimum scene depth.')
    flags.add_argument('--max_depth', type=float, default=100,help='Maximum scene depth.')
    flags.add_argument('--toffset', type=float, default=0.05,help='translation stride')
    flags.add_argument('--roffset', type=float, default=1,help='rotation stride')
    flags.add_argument('--num_mpi_planes', type=int, default=32,help='Number of MPI planes to infer.')


    # render
    flags.add_argument('--translation_render_multiples', type=str, default='-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12',
                                help='Multiples of input translation offset to render outputs at.')
    flags.add_argument('--rotation_render_multiples', type=str, default='-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12',
                                help='Multiples of input rotation offset to render outputs at.')

    flags=flags.parse_args()


    # path
    dir_list=glob.glob(os.path.join(flags.root_dir,'*'))

    # intrinsics
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

    # render parameter
    translation_render_list = [float(x) for x in flags.translation_render_multiples.split(',')]
    rotation_render_list = [float(x) for x in flags.rotation_render_multiples.split(',')]

    import cv2
    import numpy as np

    for mpi_dir in dir_list:
        mpi_name=mpi_dir.split('/')[-1]
        pred_mpis_concat,gt_mpis_concat=load_mpi(mpi_dir,mpi_prefix_predict=flags.mpi_prefix_predict,
                                                         mpi_extension_predict=flags.mpi_extension_predict,
                                                         mpi_prefix_gt=flags.mpi_prefix_gt,
                                                         mpi_extension_gt=flags.mpi_extension_gt)

        for rt_idx in range(3):
            for render_idx in range(len(translation_render_list)):
                # rotation x with translation x
                translation_list=torch.zeros(3)
                translation_list[rt_idx]+=translation_render_list[render_idx]*flags.toffset
                rotation_list=[0,0,0]
                if rt_idx<2:
                    rotation_list[rt_idx+1]+=rotation_render_list[render_idx]*flags.roffset

                pose = torch.zeros((4,4)).unsqueeze(0).float()
                pose[0,:3,:3]=torch.from_numpy(R.from_euler('zyx', rotation_list, degrees=True).as_matrix())
                pose[0,:3,3]=translation_list

                pred_mpis_concat=pred_mpis_concat.to(device)
                gt_mpis_concat=gt_mpis_concat.to(device)

                pred_render_img=mpi_render_view(pred_mpis_concat, pose, mpi_planes, intrinsics).cpu()
                gt_render_img=mpi_render_view(gt_mpis_concat, pose, mpi_planes, intrinsics).cpu()

                pred_img_cv2=np.uint8(np.clip((pred_render_img[0]*127.5+127.5),0,255))
                gt_img_cv2=np.uint8(np.clip((gt_render_img[0]*127.5+127.5),0,255))
                pred_img_pil = Image.fromarray(pred_img_cv2)
                gt_img_pil = Image.fromarray(gt_img_cv2)
                output_dir=os.path.join(flags.output_dir,mpi_name)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                pred_img_pil.save(os.path.join(output_dir,'%sx%.2f_y%.2f_z%.2f_y%.2f_x%.2f.png'%(flags.mpi_prefix_predict,translation_list[0],translation_list[2],translation_list[2],rotation_list[1],rotation_list[2])))
                gt_img_pil.save(os.path.join(output_dir,'%sx%.2f_y%.2f_z%.2f_y%.2f_x%.2f.png'%(flags.mpi_prefix_gt,translation_list[0],translation_list[2],translation_list[2],rotation_list[1],rotation_list[2])))
        print('%s render finished'%(mpi_name))

