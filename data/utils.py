import math
import numpy as np
import PIL.Image as pil
from scipy import signal
#import torch 
def write_image(filename, image):
  """Save image to disk."""
  byte_image = np.clip(image, 0, 255).astype('uint8')
  image_pil = pil.fromarray(byte_image)
  image_pil.save(filename)


def write_pose(filename, pose):
  with open(filename, 'w') as fh:
    for i in range(3):
      for j in range(4):
        fh.write('%f ' % (pose[i, j]))


def write_intrinsics(fh, intrinsics):
  fh.write('%f ' % intrinsics[0, 0])
  fh.write('%f ' % intrinsics[1, 1])
  fh.write('%f ' % intrinsics[0, 2])
  fh.write('%f ' % intrinsics[1, 2])



