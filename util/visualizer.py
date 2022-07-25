### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import os
import time
from . import util
from . import html
import scipy.misc 
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.name = opt.name
        log_dir=os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdir(log_dir)
        self.log_name = os.path.join(log_dir, 'log.txt')
        self.image_save_dir=os.path.join(log_dir, 'train_result')
        self.image_result_dir = os.path.join(log_dir, 'test_result')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ log (%s) ================\n' % now)
            for arg in vars(self.opt):
                log_file.write('%s: %s\n'%(str(arg),str(getattr(self.opt,arg))))

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in sorted(errors.items()):
            if v != 0:
                message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
            
    def print_current_errors_mean(self, epoch, i, errors, t, all_loss):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        cnt = 0
        for k, v in sorted(errors.items()):
            #if v != 0:
            loss_k = all_loss[:, cnt]
            cnt = cnt + 1
            mean_loss = np.mean(loss_k[np.where(loss_k)])
            message += '%s: %.3f ' % (k, mean_loss)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def save_images(self, save_dir, visuals):
        img_dir = os.path.join(self.image_save_dir, save_dir)
        util.mkdir(img_dir)
        for label, image_numpy in visuals.items():
            save_ext = 'png'
            image_name = '%s.%s' % (label, save_ext)
            save_path = os.path.join(img_dir, image_name)
            util.save_image(image_numpy, save_path)
    
    def save_images_test(self, index, visuals):
        img_dir = os.path.join(self.image_result_dir, '%04d/'%index)
        util.mkdir(img_dir)
        for label, image_numpy in visuals.items():
            save_ext = 'png'
            image_name = '%s.%s' % (label, save_ext)
            save_path = os.path.join(img_dir, image_name)
            util.save_image(image_numpy, save_path)


    def vis_print(self, message):
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

