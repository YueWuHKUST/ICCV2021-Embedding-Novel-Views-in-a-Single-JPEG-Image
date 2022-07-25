import os
import random, glob
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import glob
import os
import cv2, time

## Dataloader for Real Estate dataset
class RealDataset(data.Dataset):
    def __init__(self, opt):
        super(RealDataset,self).__init__()
        self.opt = opt
        if self.opt.isTrain == True:
            phase = 'train'
        else:
            phase = 'test_final'
        self.phase = phase
        self.all_image_paths=self.load_all_image_paths(os.path.join(opt.image_dir,phase))
        self.n_of_seqs = len(self.all_image_paths)                 # number of sequences to train
        print("Load number of mpi paths = %d"%self.n_of_seqs)

        self.transforms_rgb = transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5, 0.5)])
        self.transforms_alpha = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        data = self.all_image_paths[index]
        rgb_paths = data[0]
        alpha_paths = data[1]
        ref_path = data[2]

        try:
            ref_img=Image.open(ref_path)

        except:
            print('open failed: %s'%(ref_path))
            return None
        w_orig,h_orig=ref_img.size
        if w_orig<self.opt.image_width*(1+self.opt.random_resize_ratio) or h_orig<self.opt.image_height*(1+self.opt.random_resize_ratio):
            return None

        # resize first
        # crop next
        random_reize_w = int(np.random.uniform(1.0, 1.0 + self.opt.random_resize_ratio)*self.opt.image_width)
        random_reize_h = int(np.random.uniform(1.0, 1.0 + self.opt.random_resize_ratio)*self.opt.image_height)

        if random_reize_w > self.opt.image_width:
            random_crop_x_1 = np.random.randint(random_reize_w - self.opt.image_width)
        else:
            random_crop_x_1 = 0

        if random_reize_h > self.opt.image_height:
            random_crop_y_1 = np.random.randint(random_reize_h - self.opt.image_height)
        else:
            random_crop_y_1 = 0

        random_crop_x_2 = random_crop_x_1 + self.opt.image_width
        random_crop_y_2 = random_crop_y_1 + self.opt.image_height


        ref_img_np = np.array(ref_img)
        ref_img_np = cv2.resize(ref_img_np, (random_reize_w, random_reize_h))
        ref_img_np = ref_img_np[random_crop_y_1:random_crop_y_2,random_crop_x_1:random_crop_x_2]
        ref_img_tensor = self.transforms_rgb(ref_img_np)

        mpi_list=[]
        for idx in range(len(rgb_paths)):
            rgb_path=rgb_paths[idx]
            alpha_path=alpha_paths[idx]
            try:
                rgb_img=Image.open(rgb_path)
            except:
                print('open failed: %s'%(rgb_path))
                return None

            rgb_img_np = np.array(rgb_img)
            rgb_img_np = cv2.resize(rgb_img_np, (random_reize_w, random_reize_h))
            rgb_img_np = rgb_img_np[random_crop_y_1:random_crop_y_2,random_crop_x_1:random_crop_x_2]
            
            rgb_img_tensor=self.transforms_rgb(rgb_img_np)

            try:
                alpha_img=Image.open(alpha_path)
            except:
                print('open failed: %s'%(alpha_path))
                return None
            

            alpha_img_np = np.array(alpha_img)
            alpha_img_np = cv2.resize(alpha_img_np, (random_reize_w, random_reize_h))
            alpha_img_np = alpha_img_np[random_crop_y_1:random_crop_y_2,random_crop_x_1:random_crop_x_2]
            alpha_img_tensor=self.transforms_alpha(alpha_img_np)

            rgb_alpha_tensor=torch.cat((rgb_img_tensor,alpha_img_tensor),dim=0)
            mpi_list.append(rgb_alpha_tensor)

        return [ref_img_tensor, mpi_list]

    def __len__(self):
        return self.n_of_seqs

    def name(self):
        return 'RealEstate Dataset'

    def load_all_image_paths(self, root_dir):
        video_list = sorted(os.listdir(root_dir))
        if self.opt.load_path_npy:
            data_list = np.load(self.opt.path_npy, allow_pickle=True)
        else:
            data_list = []
            cnt = 0
            for video in video_list:
                cnt = cnt + 1
                if cnt % 1000 == 0:
                    print(cnt)
                video_dir = os.path.join(root_dir,video)
                frame_list = sorted(os.listdir(video_dir))
                for frame in frame_list:
                    frame_dir=os.path.join(video_dir,frame)
                    rgb_paths = sorted(glob.glob(os.path.join(frame_dir,'mpi_rgb_*.png')))
                    alpha_paths = sorted(glob.glob(os.path.join(frame_dir,'mpi_alpha_*.png')))
                    ref_path = sorted(glob.glob(os.path.join(frame_dir,'src_image_0.png')))

                    if (not len(rgb_paths)==self.opt.mpi_num) or (not len(alpha_paths)==self.opt.mpi_num) or (not len(ref_path)==1):
                        continue
                    data_list.append((rgb_paths, alpha_paths,ref_path[0],video))
                #if cnt > 10000:
                #    break
            #np.save("path_%s.npy"%self.opt.dataset, data_list)
        return data_list




class RealDatasetRender(RealDataset):
    def __getitem__(self, index):
        data = self.all_image_paths[index]
        rgb_paths = data[0]
        alpha_paths = data[1]
        ref_path = data[2]

        try:
            ref_img=Image.open(ref_path)

        except:
            print('open failed: %s'%(ref_path))
            return None
        w_orig,h_orig=ref_img.size
        if w_orig<self.opt.image_width*(1+self.opt.random_resize_ratio) or h_orig<self.opt.image_height*(1+self.opt.random_resize_ratio):
            return None

        # resize first
        # crop next
        random_reize_w = int(np.random.uniform(1.0, 1.0 + self.opt.random_resize_ratio)*self.opt.image_width)
        random_reize_h = int(np.random.uniform(1.0, 1.0 + self.opt.random_resize_ratio)*self.opt.image_height)

        if random_reize_w > self.opt.image_width:
            random_crop_x_1 = np.random.randint(random_reize_w - self.opt.image_width)
        else:
            random_crop_x_1 = 0

        if random_reize_h > self.opt.image_height:
            random_crop_y_1 = np.random.randint(random_reize_h - self.opt.image_height)
        else:
            random_crop_y_1 = 0

        random_crop_x_2 = random_crop_x_1 + self.opt.image_width
        random_crop_y_2 = random_crop_y_1 + self.opt.image_height


        ref_img_np = np.array(ref_img)
        ref_img_np = cv2.resize(ref_img_np, (random_reize_w, random_reize_h))
        ref_img_np = ref_img_np[random_crop_y_1:random_crop_y_2,random_crop_x_1:random_crop_x_2]
        ref_img_tensor = self.transforms_rgb(ref_img_np)

        mpi_list=[]
        for idx in range(len(rgb_paths)):
            rgb_path=rgb_paths[idx]
            alpha_path=alpha_paths[idx]
            try:
                rgb_img=Image.open(rgb_path)
            except:
                print('open failed: %s'%(rgb_path))
                return None

            rgb_img_np = np.array(rgb_img)
            rgb_img_np = cv2.resize(rgb_img_np, (random_reize_w, random_reize_h))
            rgb_img_np = rgb_img_np[random_crop_y_1:random_crop_y_2,random_crop_x_1:random_crop_x_2]
            
            rgb_img_tensor=self.transforms_rgb(rgb_img_np)

            try:
                alpha_img=Image.open(alpha_path)
            except:
                print('open failed: %s'%(alpha_path))
                return None
            

            alpha_img_np = np.array(alpha_img)
            alpha_img_np = cv2.resize(alpha_img_np, (random_reize_w, random_reize_h))
            alpha_img_np = alpha_img_np[random_crop_y_1:random_crop_y_2,random_crop_x_1:random_crop_x_2]
            alpha_img_tensor=self.transforms_alpha(alpha_img_np)

            rgb_alpha_tensor=torch.cat((rgb_img_tensor,alpha_img_tensor),dim=0)
            mpi_list.append(rgb_alpha_tensor)

        return [ref_img_tensor, mpi_list, ref_path]