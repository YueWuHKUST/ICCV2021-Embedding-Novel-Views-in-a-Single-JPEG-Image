#!/usr/bin/python
#
import collections
import math
import os.path
import torch
#import torch.nn as nn
import torchvision.transforms as transforms

from . import utils
import glob
import numpy as np 
from PIL import Image 
import tensorflow as tf 
from stereomag.mpi import MPI
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

batch_size = 1#FLAGS.batch_size
min_depth = 1#FLAGS.min_depth
max_depth = 100#FLAGS.max_depth
num_psv_planes = 32#FLAGS.num_psv_planes
num_mpi_planes = 32#FLAGS.num_mpi_planes
img_height = 128
img_width = 128
which_color_pred = 'bg'
model = MPI()


def preprocess_image(image):
    """Preprocess the image for CNN input.

    Args:
        image: the input image in either float [0, 1] or uint8 [0, 255]
    Returns:
        A new image converted to float with range [-1, 1]
    """
    image = image / 255.0
    return image * 2 - 1

def deprocess_image(image):
    image = (image + 1.) / 2. * 255.0
    image = np.minimum(np.maximum(image, 0.0), 255.0)
    return image.astype(np.uint8)

def deprocess_save(image):
    image = (image + 1.) / 2. * 255.0
    image = np.minimum(np.maximum(image, 0.0), 255.0)[0,...]
    return image.astype(np.uint8)

with tf.name_scope('input_data'):
    raw_tgt_image = tf.placeholder(tf.float32, [batch_size, img_height, img_width, 3])
    raw_tgt_image_ = preprocess_image(raw_tgt_image)
    raw_ref_image = tf.placeholder(tf.float32, [batch_size, img_height, img_width, 3])
    raw_ref_image_ = preprocess_image(raw_ref_image)
    raw_src_images = tf.placeholder(tf.float32, [batch_size, img_height, img_width, 6])
    raw_src_images_ = preprocess_image(raw_src_images)
    raw_tgt_pose = tf.placeholder(tf.float32, [batch_size, 4, 4])
    raw_ref_pose = tf.placeholder(tf.float32, [batch_size, 4, 4])
    raw_src_poses = tf.placeholder(tf.float32, [batch_size, 2, 4, 4])
    raw_intrinsics = tf.placeholder(tf.float32, [batch_size, 3, 3])
    _, num_source, _, _ = raw_src_poses.get_shape().as_list()


with tf.name_scope('setup'):
    psv_planes = model.inv_depths(min_depth, max_depth, num_psv_planes)
    mpi_planes = model.inv_depths(min_depth, max_depth, num_mpi_planes)
with tf.name_scope('inference'):
    pred, stereo_net_output = model.infer_mpi(raw_src_images_, raw_ref_image_, raw_ref_pose, raw_src_poses,
                            raw_intrinsics, which_color_pred, num_mpi_planes,
                            psv_planes)
    rgba_layers = pred['rgba_layers']


saver=tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt=tf.train.get_checkpoint_state("./pretrain_mpi/siggraph_model_20180701/")
if ckpt:
   print('loaded '+ckpt.model_checkpoint_path)
   # uncomment the following line if finetuning from pretrained 256x512 model
   #saver_a = tf.train.Saver([v for v in tf.trainable_variables() if "net" in v.name])
   #saver_a.restore(sess, ckpt.model_checkpoint_path)
   
   # comment the following line if finetuning from pretrained 256 x512 model
   saver.restore(sess,ckpt.model_checkpoint_path)
  

def get_transform(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


class ViewSequence(object):
  '''
    collections.namedtuple('ViewSequence'
                           ['id', 'timestamp', 'intrinsics', 'pose', 'image'])):
  '''
  def name(self):
    return 'ViewSequence'

  def initialize(self, opt):
    self.img_root = opt.image_dir
    self.opt = opt
    self.txt_root = opt.txt_dir
    self.isTrain = opt.isTrain
    self.num_source = opt.num_source
    if self.isTrain is True:
      self.phase = 'train'
    else:
      self.phase = 'test'
    self.min_stride = opt.min_stride
    self.max_stride = opt.max_stride
    self.min_scale = opt.min_scale
    self.max_scale = opt.max_scale
    self.sample_len = opt.sample_len
    self.image_width = opt.image_width
    self.image_height = opt.image_height

    self.all_paths = self.load_files(self.img_root + self.phase + "/", self.txt_root + self.phase + "/")
    self.transform = get_transform()
    
  def load_data(self, index):
    ### Sample length 10 frames


    #print("debug")
    paths = self.all_paths[index]
    img_path = paths[0]
    seq_len = len(img_path)
    txt_path = paths[1]
    camera_params = self.parse_camera_lines(txt_path)
    stride = int(np.random.uniform(self.min_stride, self.max_stride + 1))
    #print("debug")
    # Now pick the starting index.
    # If the subsequence starts at index i, then its final element will be at
    # index i + (length - 1) * stride, which must be less than the length of
    # the sequence. Therefore i must be less than maxval, where:
    maxval = seq_len - (self.sample_len - 1) * stride
    start_id = int(np.random.uniform(0, maxval))
    assert len(camera_params) == len(img_path)
    camera_params_sub = self.subsequence(
        start_id, start_id + 1 + (self.sample_len - 1) * stride, stride, camera_params)
    
    img_paths = self.load_corresponding_images(txt_path, camera_params_sub)

    ### Sample length two
    tgt_idx = int(np.random.uniform(0, self.sample_len))
    shuffled_inds = np.arange(self.sample_len)
    np.random.shuffle(shuffled_inds)
    src_inds = shuffled_inds[:self.num_source]
    ref_idx = src_inds[0]

    tgt_image_path = img_paths[tgt_idx]
    ref_image_path = img_paths[ref_idx]
    
    ## Load Image
    src_images_list = []
    for i in src_inds:
      cnt_path = img_paths[i]
      cnt_image = self.read_image(cnt_path)
      src_images_list.append(cnt_image)

    tgt_image = self.read_image(tgt_image_path)
    ref_image = self.read_image(ref_image_path)

    tgt_timestamp, tgt_intrinsics, tgt_pose = self.generate_pose(tgt_idx, camera_params_sub)
    ref_timestamp, ref_intrinsics, ref_pose = self.generate_pose(ref_idx, camera_params_sub)
    src_timestamps = []
    src_intrinsics = []
    src_poses = []
    for p in src_inds:
      src_timestamp, src_intrinsic, src_pose = self.generate_pose(p, camera_params_sub)
      src_timestamps.append(src_timestamp)
      src_intrinsics.append(src_intrinsic)
      src_poses.append(src_pose)
    src_poses_cat = np.concatenate([np.expand_dims(q, axis=0) for q in src_poses], axis=0)
    src_timestamp_cat = np.concatenate([np.expand_dims(q, axis=0) for q in src_timestamps], axis=0)

    ### data augmentation

    offset_x, offset_y, scaled_size = self.random_scale_crop_params(self.min_scale, self.max_scale, self.image_height, self.image_width)
    tgt_image, tgt_intrinsics = self.crop_image_and_adjust_intrinsics(tgt_image, tgt_intrinsics, offset_y, offset_x, \
      self.image_height, self.image_width, scaled_size, tgt_image_path)
    tgt_image = np.array(tgt_image)
    ref_image, ref_intrinsics = self.crop_image_and_adjust_intrinsics(ref_image, ref_intrinsics, offset_y, offset_x, \
      self.image_height, self.image_width, scaled_size, ref_image_path)
    ref_image = np.array(ref_image)
    ref_intrinsics = self.make_intrinsics_matrix(ref_intrinsics[0] * self.image_width,
                                        ref_intrinsics[1] * self.image_height,
                                        ref_intrinsics[2] * self.image_width,
                                        ref_intrinsics[3] * self.image_height)



    src_image_aug = []
    #src_intrinsic_aug = []
    for i in range(self.num_source):
      cnt_src_image = src_images_list[i]
      cnt_src_intrinsic = src_intrinsics[i]
      cnt_src_image_aug, cnt_src_intrinsic_aug = self.crop_image_and_adjust_intrinsics(cnt_src_image, cnt_src_intrinsic, offset_y, \
        offset_x, self.image_height, self.image_width, scaled_size, img_paths[src_inds[i]])
      cnt_src_image_aug = np.array(cnt_src_image_aug)
      src_image_aug.append(cnt_src_image_aug)
      #print("cnt_src_image_aug", cnt_src_image_aug.shape)
    src_image_aug_cat = np.concatenate([p for p in src_image_aug], axis=2)
    #print("src_image", src_image_aug_cat.shape)



    #print(src_timestamp_cat)
    # 2x4x4
    ## Load poses
    # Put everything into a dictionary.
    instance = {}
    instance['tgt_image'] = np.expand_dims(tgt_image, axis=0)
    instance['ref_image'] = np.expand_dims(ref_image, axis=0)
    #print(instance['ref_image'].shape)
    instance['src_images'] = np.expand_dims(src_image_aug_cat, axis=0)
    instance['tgt_pose'] = np.expand_dims(tgt_pose, axis=0)
    instance['src_poses'] = np.expand_dims(src_poses_cat, axis=0)
    instance['intrinsics'] = np.expand_dims(ref_intrinsics, axis=0)
    instance['ref_pose'] = np.expand_dims(ref_pose, axis=0)
    #instance['ref_name'] = sequence.id
    instance['src_timestamps'] = np.expand_dims(src_timestamp_cat, axis=0)
    instance['ref_timestamp'] = np.expand_dims(np.array([ref_timestamp]), axis=0)
    instance['tgt_timestamp'] = np.expand_dims(np.array([tgt_timestamp]), axis=0)

    true_rgba = sess.run(rgba_layers,feed_dict={raw_tgt_image:instance['tgt_image'],\
                                    raw_ref_image: instance['ref_image'],\
                                    raw_src_images: instance['src_images'],\
                                    raw_tgt_pose : instance['tgt_pose'], \
                                    raw_ref_pose : instance['ref_pose'], \
                                    raw_intrinsics: instance['intrinsics'], \
                                    raw_src_poses: instance['src_poses']})

    ### construct pytorch data
    #print(ref_image)
    ref_image = (ref_image / 255.0) * 2.0 -1.0
    # batch x h x w x 3
    ref_image = np.transpose(ref_image, [2, 0, 1])
    ref_image = torch.from_numpy(np.expand_dims(ref_image, axis=0))
    true_rgba = np.transpose(true_rgba, [0, 3, 4, 1, 2])
    # batch x num x 4 x height x width
    RGBA = torch.from_numpy(true_rgba)
    # RGBA range
    # [batch_size, img_height, img_width, num_mpi_planes, 4]
    # RGB[-1, 1] + Alpha[0,1]
    return_list = {'MPI': RGBA, 'ref': ref_image}
    return return_list

  
  def generate_pose(self, idx, camera_pose):
    cnt_pose = camera_pose[idx]
    timestamp, intrinsics, poses = cnt_pose
    filler =  np.array([[0., 0., 0., 1.]])
    #print(poses.shape, filler.shape)
    poses_ = np.concatenate([poses, filler], axis=0)
    #print(intrinsics)
    #intrinsics = self.make_intrinsics_matrix(intrinsics[0] * self.image_width,
    #                                    intrinsics[1] * self.image_height,
    #                                    intrinsics[2] * self.image_width,
    #                                    intrinsics[3] * self.image_height)
    #print(intrinsics)
    return float(timestamp), intrinsics, poses_


  def read_image(self, path):
    try:
      img = Image.open(path)
    except:
      print("failed image path = ", path)
    return img

  def make_intrinsics_matrix(self, fx, fy, cx, cy):
    # Assumes batch input.
    r1 = np.array([[fx, 0., cx]])
    r2 = np.array([[0, fy,  cy]])
    r3 = np.array([[0., 0., 1.]])
    intrinsics = np.concatenate([r1, r2, r3], axis=0)
    return intrinsics

  def load_corresponding_images(self, txt_path, camera_params_sub):
    # img_path is the root
    txt_folder = txt_path.split("/")[-1][:-4]
    image_root = self.img_root + self.phase + "/" + txt_folder + "/"
    
    path = []
    for i in range(len(camera_params_sub)):
      timestamp = camera_params_sub[i][0]
      #print("image_root", image_root)
      #print(timestamp)
      image_full_path = image_root + timestamp + ".png"
      #print(image_full_path)
      assert os.path.exists(image_full_path)
      path.append(image_full_path)
    return path

  def subsequence(self, start, end, stride, camera_params):
    #print(start, stride, end)
    #print("len camera params", len(camera_params))
    cnt_camera = camera_params[start:end:stride]
    uniform_random = np.random.uniform(0.0, 1.0)
    #print("cnt_camera", cnt_camera)
    if uniform_random < 0.5:
      #reverse
      cnt_camera = cnt_camera[::-1]
    #print("cnt_camera", cnt_camera)
    return cnt_camera
    

  def random_scale_crop_params(self, min_scale, max_scale, height, width):
    scale_factor = np.random.uniform(min_scale, max_scale)
    scale_height = int(height * scale_factor)
    scale_width = int(width * scale_factor)
    scaled_size = [scale_height, scale_width]
    offset_limit_y = scale_height - height + 1
    offset_limit_x = scale_width  - width + 1
    offset_y = int(np.random.uniform(0, offset_limit_y))
    offset_x = int(np.random.uniform(0, offset_limit_x))
    return offset_x, offset_y, scaled_size

  def load_files(self, img_root, txt_root):
    # root = /disk1/yue/data/stereo_mpi/train
    scenes = sorted(os.listdir(img_root))
    #print("image sequences number = ", len(scenes))
    data = []
    for i in range(10000):#len(scenes)):
      #print("process %02d/%02d"%(i, len(scenes)))
      sub_dir = img_root  + scenes[i] + "/"
      #print(sub_dir)
      imgs = sorted(glob.glob(sub_dir + "*.png"))
      required_length = (self.sample_len - 1) * self.max_stride + 1
      #print("required length", required_length)
      #print("len img", len(imgs))
      if len(imgs) < required_length:
        continue
      txt_path = txt_root + scenes[i] + ".txt"
      if os.path.exists(txt_path):
        data.append((imgs, txt_path))
    #$np.save("all_paths.npy", data)
    return data

  def string2number(self, string):
    num_list = []
    str_ = string.split(" ")
    for i in range(len(str_)):
      if i == len(str_) - 1:
        num_list.append(float(str_[i][:-1]))
      elif i == 0:
        num_list.append(str_[i])
      else:
        num_list.append(float(str_[i]))
    return num_list

  def build_matrix(self, elements):
    """Stacks elements along two axes to make a tensor of matrices.

    Args:
      elements: [n, m] matrix of tensors, each with shape [...].

    Returns:
      [..., n, m] tensor of matrices, resulting from concatenating
        the individual tensors.
    """
    rows = [np.stack(row_elements, axis=-1) for row_elements in elements]
    return np.stack(rows, axis=-2)





  def parse_single_line(self, data):
    data = self.string2number(data)
    timestamps = data[0]
    intrinsics = data[1:5]
    #print([data[7:11], data[11:15], data[15:19]])
    poses = self.build_matrix([data[7:11], data[11:15], data[15:19]])
    return timestamps, intrinsics, poses

  def parse_camera_lines(self, txt_path):
    """Reads a camera file, returning a single ViewSequence (without images).

    Args:
      lines: [N] string tensor of camera lines

    Returns:
      The corresponding length N sequence, as a ViewSequence.
    """
    # The first line contains the YouTube video URL.
    # Format of each subsequent line: timestamp fx fy px py k1 k2 row0 row1  row2
    # Column number:                  0         1  2  3  4  5  6  7-10 11-14 15-18
    f = open(txt_path, 'r')
    data = f.readlines()
    youtube_url = data[0]
    camera_params = []
    for i in range(1, len(data)):
      camera_params.append(self.parse_single_line(data[i]))
    return camera_params

  def crop_image_and_adjust_intrinsics(self, 
      image, intrinsics, offset_y, offset_x, height, width, scaled_size, path):
    """Crop images and adjust instrinsics accordingly.

    Args:
      image: [..., H, W, C] images
      intrinsics: [..., 4] normalised camera intrinsics
      offset_y: y-offset in pixels from top of image
      offset_x: x-offset in pixels from left of image
      height: height of region to be cropped
      width: width of region to be cropped

    Returns:
      [..., height, width, C] cropped images,
      [..., 4] adjusted intrinsics
    """

    scale_height, scale_width =  scaled_size
    try:
      scaled_image = image.resize(size=(scale_width, scale_height))
    except:
      print("failed file", path)
      print("scale size", scaled_size)
    # intrinsics = [fx fy cx cy]
    # Convert to pixels, offset, and normalise to cropped size.
    pixel_intrinsics = intrinsics * np.array([scale_width, scale_height, scale_width, scale_height])
    #print("intrinsics", intrinsics)
    #print("pixel intrinsics", pixel_intrinsics)
    cropped_pixel_intrinsics = pixel_intrinsics - [0.0, 0.0, offset_x, offset_y]
    cropped_intrinsics = cropped_pixel_intrinsics / [width, height, width, height]
    #cropped_images = tf.image.crop_to_bounding_box(
    #    image, offset_y, offset_x, height, width)
    cropped_images = scaled_image.crop(box=[offset_x, offset_y, offset_x+width, offset_y+height])
    
    return cropped_images, cropped_intrinsics
    
  def __len__(self):
    #print("debug paths", len(self.all_paths))
    return 10#len(self.all_paths)
