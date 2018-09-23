import sys
import matplotlib.pyplot as plt

import numpy as np

import h5py
from ulti import blend, grayscale, my_imshow, psnr
from Dataset import Dataset
import time


import scipy
import scipy.io
import scipy.misc
import scipy.io as sio
import os

# dataset_test = ChairsSDHomDataset(filename = '/fileserver/haitian/dataset/ChairsSDHom/test_bicubic.h5')
# for i in range(0,100):
#     print 'next batch'
#     dataset_test.nextBatch(batchsize=4, shuffle=False, view_mode = 'Random', residue = False, augmentation = False, index_inc = False)
#     dataset_test.nextBatch(batchsize=4, shuffle=False, view_mode = 'Random', residue = False, augmentation = False, index_inc = True)


# dataset_train = Dataset(filename = '../HT_sr/flowdata/LF_video_Dataset/train_bicubic.h5', scale = 8)
# dataset_train.nextBatch(batchsize=8,shuffle=True,view_mode = 'Random',residue = True, augmentation = True)
# dataset_train.nextBatch(batchsize=8,shuffle=True,view_mode = 'Random',residue = True, augmentation = True)

# # dataset_train.nextBatch(batchsize=8,shuffle=False,view_mode = 'Random',residue = False, augmentation = True)
# # dataset_test = Dataset(filename = '../HT_sr/flowdata/LF_video_Dataset/test_bicubic.h5')


class SintelDataset(Dataset):
    def __init__(self, filename = None, scale = 8):
        # load a dictionary that contains all arrays
        self.arrays = dict()
        self.loadArrays(filename, scale = scale) 
        # set iterator counter
        self.rng_clip = np.random.RandomState(100)           # random generator
        self.rng_frame_difference = np.random.RandomState(300)
        self.rng_frame = np.random.RandomState(400)
        self.rng_augmentation = np.random.RandomState(200)
        self.idx_clip = 0
        self.idx_frame1 = 0
        return

    def loadArrays(self, filename, scale = 8):
        self.scale = scale
        f = h5py.File(filename,'r')
        self.arrays['img_HR'] = f.get('/img_HR')
        self.arrays['img_LR'] = f.get('/img_LR_'+str(scale))
        self.arrays['img_LR_upsample'] = f.get('/img_LR_'+str(scale)+'_upsample')
        self.arrays['img_MDSR'] = f.get('/MDSR_'+str(scale))
        self.frame_number = f.get('/frame_number')

        print 'loading dataset: ', filename
        print 'img_HR ', self.arrays['img_HR'].shape
        print 'img_LR ', self.arrays['img_LR'].shape
        print 'img_LR_upsample ', self.arrays['img_LR_upsample'].shape
        if not self.arrays['img_MDSR'] is None:
            print 'img_MDSR ', self.arrays['img_MDSR'].shape
        else:
            print 'cannot find img_MDSR'
        print  'frame number ', self.frame_number.shape
        
        self.size_clip = self.arrays['img_HR'].shape[0]
        self.size_C = self.arrays['img_HR'].shape[2]
        self.size_H = self.arrays['img_HR'].shape[3]
        self.size_W = self.arrays['img_HR'].shape[4]
        return

    def inc_idx(self, d_frame=None):
        self.idx_frame1 = self.idx_frame1 + 1
        c = self.idx_clip
        if self.idx_frame1 + d_frame >= self.frame_number[c,0]:
            self.idx_clip = (self.idx_clip + 1)%self.size_clip
            self.idx_frame1 = 0
        self.loopover = self.idx_frame1 == 0 and self.idx_clip == 0  # flag for iterate a single shot of dataset 
        return 

    def genIndex_list(self, batch_size, shuffle, index_inc = True, view_mode = ''):
        #  __genIndex_list(batch_size, random, index_inc = True, frame_mode):   generate list of index to be sample with
        #       batch_size:          number of sample to be generated
        #       random:             set True if want to randomly generate sample
        #           index_inc:           increase index (affective if random=False)
        #       mode:               choose frame mode
        # clip_list, frame1_list, frame2_list  = self.genIndex_list(batchsize, shuffle, index_inc = index_inc, view_mode = view_mode)

        clip_list = []
        frame1_list = []
        frame2_list = []
        for i in range(batch_size):
            # gen frame difference
            if view_mode == 'Random':   #[1,2,3,4,5]     # for training
                d_frame = 1 #self.rng_frame_difference.randint(1,5)
            elif view_mode == 'Fixed-inv': #[1]            # for testing
                d_frame = 1
            # gen frame1
            if shuffle:
                c = self.rng_clip.randint(0,self.size_clip-1)
                frame1 = self.rng_frame.randint(0,self.frame_number[c,0]-1-d_frame)
                frame2 = frame1 + d_frame
                # print 'rng limit:', self.frame_number[c,0]-1-d_frame, 'frame no:', self.frame_number[c,0], 'd_frame:', d_frame
            else:
                if index_inc:
                    self.inc_idx(d_frame = d_frame)
                c = self.idx_clip
                frame1 = self.idx_frame1
                frame2 = frame1 + d_frame
            # append list
            clip_list.append(c)
            frame1_list.append(frame1)
            frame2_list.append(frame2)
        return clip_list, frame1_list, frame2_list

    def debug(self):
        print self.frame_number.shape
        # exit()
        for c in range(self.size_clip):
            print self.frame_number[c,0]
            continue
            for f in range(self.frame_number[c,0]):
                print c, f,'in ', self.size_clip, self.frame_number[c,0]
                a = np.asarray(self.arrays['img_HR'][c,f,:,:,:], dtype = np.float32) / 255.0 


    # def __genViewPosition(self, view_mode, specified_view = None):
    #     if (view_mode == 'Random'):
    #         rnd_y = self.rng_viewpoint.randint(0,7)
    #         rnd_x = self.rng_viewpoint.randint(0,7)
    #         while True:
    #             rnd_y_ref = self.rng_viewpoint.randint(0,7)
    #             rnd_x_ref = self.rng_viewpoint.randint(0,7)
    #             if (rnd_y_ref!=rnd_y) or (rnd_x_ref!=rnd_x):
    #                 break
    #     elif (view_mode == 'Random_FixOffset'):        #  disparicy = (-3,-3)
    #         rnd_y = self.rng_viewpoint.randint(6,7)
    #         rnd_x = self.rng_viewpoint.randint(6,7)
    #         rnd_y_ref = rnd_y-6
    #         rnd_x_ref = rnd_x-6
    #     elif (view_mode == 'Fixed'):
    #         rnd_y = 0
    #         rnd_x = 0
    #         rnd_y_ref = 3
    #         rnd_x_ref = 3
    #     elif (view_mode == 'Fixed-inv'):
    #         rnd_y = 3
    #         rnd_x = 3
    #         rnd_y_ref = 0
    #         rnd_x_ref = 0
    #     elif (view_mode == 'Fixed-inv-large'):
    #         rnd_y = 7
    #         rnd_x = 7
    #         rnd_y_ref = 0
    #         rnd_x_ref = 0
    #     elif (view_mode == 'specified'):
    #         rnd_y, rnd_x, rnd_y_ref, rnd_x_ref = tuple(specified_view)
    #     return rnd_y,rnd_x,rnd_y_ref,rnd_x_ref

    def nextBatch_new(self, batchsize = 8, shuffle = False, view_mode = 'Random', augmentation = False, index_inc = True, crop_shape = None, SR=True, Dual = False):
        # nextBatch_new(batchsize = 8, shuffle = False, view_mode = 'Random', augmentation = False, index_inc = True)
        #       generate a dictionary that contains HR, LR and SR images of two views
        buff = dict()
        # init 
        input_img1_LR = np.zeros([batchsize,3,self.size_H,self.size_W], dtype = np.float32)
        input_img2_LR = np.zeros([batchsize,3,self.size_H,self.size_W], dtype = np.float32)
        input_img1_HR = np.zeros([batchsize,3,self.size_H,self.size_W], dtype = np.float32)
        input_img2_HR = np.zeros([batchsize,3,self.size_H,self.size_W], dtype = np.float32)
        if SR:
            input_img1_SR = np.zeros([batchsize,3,self.size_H,self.size_W], dtype = np.float32)
            if Dual:
                input_img2_SR = np.zeros([batchsize,3,self.size_H,self.size_W], dtype = np.float32)
        t_read = time.time()
        # print '.....', shuffle, view_mode
        clip_list, frame1_list, frame2_list  = self.genIndex_list(batchsize, shuffle, index_inc = index_inc, view_mode = view_mode)
        for k in range(batchsize):
            # generate img number
            c = clip_list[k]
            frame1 = frame1_list[k]
            frame2 = frame2_list[k]

            # augmentation
            if augmentation and self.rng_augmentation.randint(0, 1) > 0.5:
                frame1, frame2 = frame2, frame1
            # LR
            input_img1_LR[k,:,:,:] = np.asarray(self.arrays['img_LR_upsample'][c,frame1,:,:,:], dtype = np.float32) / 255.0 
            if Dual:
                input_img2_LR[k,:,:,:] = np.asarray(self.arrays['img_LR_upsample'][c,frame2,:,:,:], dtype = np.float32) / 255.0 
            # HR
            input_img1_HR[k,:,:,:] = np.asarray(self.arrays['img_HR'][c,frame1,:,:,:], dtype = np.float32) / 255.0 
            input_img2_HR[k,:,:,:] = np.asarray(self.arrays['img_HR'][c,frame2,:,:,:], dtype = np.float32) / 255.0 
            # SR
            if SR:
                input_img1_SR[k,:,:,:] = np.asarray(self.arrays['img_MDSR'][c,frame1,:,:,:], dtype = np.float32) / 255.0 
                if Dual:
                    input_img2_SR[k,:,:,:] = np.asarray(self.arrays['img_MDSR'][c,frame2,:,:,:], dtype = np.float32) / 255.0 
        t_aug = time.time()

        # data augmentation
        if augmentation:
            augmentation_config = self.augmentation_array_config()
            input_img1_LR = self.augmentation_array(input_img1_LR, augmentation_config)
            if Dual:
                input_img2_LR = self.augmentation_array(input_img2_LR, augmentation_config)
            input_img1_HR = self.augmentation_array(input_img1_HR, augmentation_config) 
            input_img2_HR = self.augmentation_array(input_img2_HR, augmentation_config)
            if SR:
                input_img1_SR = self.augmentation_array(input_img1_SR, augmentation_config)
                if Dual:
                    input_img2_SR = self.augmentation_array(input_img2_SR, augmentation_config)
        t_crop = time.time()
        
        # crop image
        if not crop_shape is None:
            input_img1_LR = input_img1_LR[:,:,0:crop_shape[0],0:crop_shape[1]]
            if Dual:
                input_img2_LR = input_img2_LR[:,:,0:crop_shape[0],0:crop_shape[1]]
            input_img1_HR = input_img1_HR[:,:,0:crop_shape[0],0:crop_shape[1]]
            input_img2_HR = input_img2_HR[:,:,0:crop_shape[0],0:crop_shape[1]]
            if SR:
                input_img1_SR = input_img1_SR[:,:,0:crop_shape[0],0:crop_shape[1]]
                if Dual:
                    input_img2_SR = input_img2_SR[:,:,0:crop_shape[0],0:crop_shape[1]]
        t_end = time.time()
        
        # pack buffer
        buff['input_img1_LR'] = input_img1_LR
        if Dual:
            buff['input_img2_LR'] = input_img2_LR
        buff['input_img1_HR'] = input_img1_HR
        buff['input_img2_HR'] = input_img2_HR
        if SR:
            buff['input_img1_SR'] = input_img1_SR
            if Dual:
                buff['input_img2_SR'] = input_img2_SR
        return buff

    def generate_compare_experiment_data(self, save_path):
        crop_shape = [416,1024]
        # init
        input_img1_LR = np.zeros([3,self.size_H,self.size_W], dtype = np.float32)
        input_img1_HR = np.zeros([3,self.size_H,self.size_W], dtype = np.float32)
        input_img2_HR = np.zeros([3,self.size_H,self.size_W], dtype = np.float32)
        input_img1_SR = np.zeros([3,self.size_H,self.size_W], dtype = np.float32)

        for v in range(1,6):
            folder = save_path+'/LR(x-'+str(v)+')-REF(x)'
            print folder
            if not os.path.exists(folder):
                os.mkdir(folder)
            if not os.path.exists(folder+'/LR/'):
                os.mkdir(folder+'/LR/')
            if not os.path.exists(folder+'/GT/'):
                os.mkdir(folder+'/GT/')
            if not os.path.exists(folder+'/REF/'):
                os.mkdir(folder+'/REF/')
            if not os.path.exists(folder+'/MDSR/'):
                os.mkdir(folder+'/MDSR/')
            if not os.path.exists(folder+'/LR_upsample/'):
                os.mkdir(folder+'/LR_upsample/')

            for c in range(self.size_clip):
                for f in range(v,self.frame_number[c,0]):
                    name = str(c)+'_'+str(f)+'.png'
                    print name
                    img_LR_upsample = np.asarray(self.arrays['img_LR_upsample'][c,f-v,:,0:crop_shape[0],0:crop_shape[1]], dtype = np.float32) / 255.0
                    img_LR = np.asarray(self.arrays['img_LR'][c,f-v,:,0:crop_shape[0]/self.scale,0:crop_shape[1]/self.scale], dtype = np.float32) / 255.0
                    input_img1_HR = np.asarray(self.arrays['img_HR'][c,f-v,:,0:crop_shape[0],0:crop_shape[1]], dtype = np.float32) / 255.0 
                    input_img2_HR = np.asarray(self.arrays['img_HR'][c,f,:,0:crop_shape[0],0:crop_shape[1]], dtype = np.float32) / 255.0 
                    # input_img1_SR = np.asarray(self.arrays['img_MDSR'][c,f,:,0:crop_shape[0],0:crop_shape[1]], dtype = np.float32) / 255.0

                    # scipy.misc.toimage(np.squeeze(np.transpose(img_LR_upsample,axes=(1,2,0))), cmin=0.0, cmax=1.0).save(folder+'/LR_upsample/'+name)
                    scipy.misc.toimage(np.squeeze(np.transpose(img_LR,axes=(1,2,0))), cmin=0.0, cmax=1.0).save(folder+'/LR/'+name)
                    # scipy.misc.toimage(np.squeeze(np.transpose(input_img1_HR,axes=(1,2,0))), cmin=0.0, cmax=1.0).save(folder+'/GT/'+name)
                    # scipy.misc.toimage(np.squeeze(np.transpose(input_img2_HR,axes=(1,2,0))), cmin=0.0, cmax=1.0).save(folder+'/REF/'+name)
                    # scipy.misc.toimage(np.squeeze(np.transpose(input_img1_SR,axes=(1,2,0))), cmin=0.0, cmax=1.0).save(folder+'/MDSR/'+name)

    def generate_MDSR_input_data(self, save_path=''):
        crop_shape = [320,512]
        # init 
        img_LR = np.zeros([3,self.size_H,self.size_W], dtype = np.float32)

        for c in range(self.size_clip):
            print c, ' of ', self.size_clip
            for f in range(self.frame_number[c,0]):
                img_LR = np.asarray(self.arrays['img_LR'][c,f,:,:,:], dtype = np.float32) / 255.0 
                scipy.misc.toimage(img_LR, cmin=0.0, cmax=1.0).save(save_path+'/'+str(c)+'_'+str(f)+'.png')

# dataset_train = SintelDataset(filename = '/fileserver/haitian/dataset/sintel_dataset/train_x4_x8.h5', scale = 8)
# dataset_train.generate_MDSR_input_data('/fileserver/haitian/haitian_backup/HT_sr/flowdata/LF_video_Dataset/MDSR/SintelDataset_train_LR_x8')
# dataset_train = SintelDataset(filename = '/fileserver/haitian/dataset/sintel_dataset/train_x4_x8.h5', scale = 4)
# dataset_train.generate_MDSR_input_data('/fileserver/haitian/haitian_backup/HT_sr/flowdata/LF_video_Dataset/MDSR/SintelDataset_train_LR_x4')
# dataset_train = SintelDataset(filename = '/fileserver/haitian/dataset/sintel_dataset/test_x4_x8.h5', scale = 8)
# dataset_train.generate_MDSR_input_data('/fileserver/haitian/haitian_backup/HT_sr/flowdata/LF_video_Dataset/MDSR/SintelDataset_test_LR_x8')
# dataset_train = SintelDataset(filename = '/fileserver/haitian/dataset/sintel_dataset/test_x4_x8.h5', scale = 4)
# dataset_train.generate_MDSR_input_data('/fileserver/haitian/haitian_backup/HT_sr/flowdata/LF_video_Dataset/MDSR/SintelDataset_test_LR_x4')

# dataset_train = SintelDataset(filename = '/fileserver/haitian/dataset/sintel_dataset/test_x4_x8.h5', scale = 8)
# dataset_train.generate_compare_experiment_data('/fileserver/haitian/haitian_backup/ECCV_RefSR_exp/Sintel_dataset')

# dataset_train = SintelDataset(filename = '/fileserver/haitian/dataset/sintel_dataset/train_x4_x8.h5', scale = 8)
# dataset_train.debug()