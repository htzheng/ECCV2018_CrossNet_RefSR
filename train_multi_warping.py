import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import time
import cPickle as pickle
import math

# from eval import eval_net
sys.path.insert(0,'./ref_utils/')
sys.path.insert(0,'./Model/')
from Model import MultiscaleWarpingNet 
#from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch

from LFDataset import LFDataset
from FlowerDataset import FlowerDataset
import matplotlib.pyplot as plt
import CustomLoss 



def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    
    if mse > 1000:
        return -100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def train_net(net, gpu=False, config={}):

    dataset_train = config['dataset_train']
    dataset_test = config['dataset_test'] 

    print('Starting training...')
    
    if config['optim'] == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=config['lr'], momentum=0.9, weight_decay=0.0005)
    elif config['optim'] == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr = config['lr'], weight_decay = 0.00005)    


    if config['loss'] == 'EuclideanLoss':
        criterion = CustomLoss.EuclideanLoss()
    elif config['loss'] == 'CharbonnierLoss':
        criterion = CustomLoss.CharbonnierLoss()
    elif config['loss'] == 'MSELoss':
        criterion = nn.MSELoss()
    else:
        print 'None loss type'
        sys.exit(0)


    loss_count = 0
    time_start = time.time()
    #flag = 1
    #buff_list = []
    for iter_ in range(config['checkpoint'],config['max_iter']):
        
        # reset the generators
        #train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, img_scale)
        #val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale)
          
        buff = dataset_train.nextBatch_new(batchsize=config['batch_size'], shuffle=True, view_mode = 'Random', augmentation = True, offset_augmentation=config['data_displacement_augmentation'], crop_shape = config['train_data_crop_shape'],Dual = config['Dual'])
        #buff_list.append(buff)
        #flag += 1
        #if flag > 5:
        #    file_ = open('./train_buff','wb')
        #    pickle.dump(buff_list,file_)
        #    file_.close()
        #    break
            


        label_img = buff['input_img1_HR']
        label_img = torch.from_numpy(label_img)

        if gpu:
            label_img = label_img.cuda()

        net_pred = net(buff,mode = 'input_img1_HR')
        # net_pred_flat = net_pred.view(-1)
        # label_img_flat = label_img.view(-1)
        loss = criterion(net_pred, label_img)
        #print (loss)
        loss_count += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (iter_ + 1) % config['snapshot'] == 0:
            torch.save(net.state_dict(),
                       config['checkpoints_dir'] + 'CP{}.pth'.format(iter_ + 1))
            print('Checkpoint {} saved !'.format(iter_ + 1))
        
        
        if (iter_ + 1) % config['display'] == 0:
            time_end = time.time()    
            time_cost = time_end - time_start
            
            #------------------------------------------------
            pre_npy = net_pred.data.cpu().numpy()
            label_img_npy = label_img.data.cpu().numpy()

            psnr_ = 0
            for i in range(pre_npy.shape[0]):
                psnr_ += psnr(pre_npy[i],label_img_npy[i]) / pre_npy.shape[0] 
            #    print (i,psnr(pre_npy[i],label_img_npy[i]))
            #------------------------------------------------------

            #buff_val = dataset_test.nextBatch_new(batchsize=config['batch_size'], shuffle=True, view_mode = 'Random', augmentation = False, offset_augmentation=config['data_displacement_augmentation'], crop_shape = config['train_data_crop_shape'])
            
            #val_img1_LR = buff_val['input_img1_LR']
            #val_img2_HR = buff_val['input_img2_HR']

            #val_img = np.concatenate((val_img1_LR,val_img2_HR),axis = 1)
            #val_label_img = buff['input_img1_HR']

            #val_img = torch.from_numpy(val_img)
            #val_img = val_img.cuda()
            #with torch.no_grad():
            #    val_pred = net(val_img) 
            #    val_pred_npy = val_pred.cpu().numpy()
            #    psnr_ = 0
            #    for i in range(val_pred_npy.shape[0]):
            #        psnr_ += psnr(val_pred_npy[i],val_label_img[i]) / val_pred_npy.shape[0]
                    #print (i,psnr(val_pred_npy[i],val_label_img[i]))
       
      
            print ('iter:%d    time: %.2fs / %diters   lr: %.8f   %s: %.7f   psnr: %.2f'%(iter_ + 1,time_cost,config['display'],config['lr'],config['loss'],loss_count / config['display'], psnr_))
            loss_count = 0
            time_start = time.time()

        if (iter_ + 1) % config['step_size'] == 0:
            config['lr'] = config['lr'] * config['gamma']
            if config['optim'] == 'SGD':
                optimizer = optim.SGD(net.parameters(), lr=config['lr'] * config['gamma'], momentum=0.9, weight_decay=0.0005)
            elif config['optim'] == 'Adam':
                optimizer = optim.Adam(net.parameters(), lr = config['lr'], weight_decay = 0.00005)





def get_args():
    parser = OptionParser()

    parser.add_option('--batch_size', dest='batch_size', default=8,
                      type='int', help='batch size')
    parser.add_option('--lr', dest='lr', default=0.0001,
                      type='float', help='learning rate')
    parser.add_option('--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('--checkpoint_file', dest='load',
                      default=False, help='load file model')

    parser.add_option('--checkpoint',dest = 'checkpoint',default = 0,type = 'int',help = 'snapshot')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default= 8 , help='downscaling factor of LR')

    parser.add_option('--loss',dest = 'loss',default='EuclideanLoss',help = 'loss type')

    parser.add_option('--dataset',dest = 'dataset',default = 'LFvideo',help = 'dataset type')

    parser.add_option('--gamma',dest = 'gamma',type = 'float', default = 0.2,help = 'lr decay')

    parser.add_option('--step_size',dest = 'step_size',type = 'float',default = 60000,help = 'step_size')

    parser.add_option('--max_iter',dest = 'max_iter',default = 1000000,type = 'int',help = 'max_iter')

    parser.add_option('--checkpoints_dir',dest = 'checkpoints_dir',default = './checkpoints/',help = 'checkpoints_dir')

    parser.add_option('--snapshot',dest = 'snapshot',default = 5000,type = 'float',help = 'snapshot')

    parser.add_option('--display',dest = 'display',default = 10,type = 'float',help = 'display')
    
    parser.add_option('--optim', dest = 'optim', default = 'SGD', help = 'optimizer type')    

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':


    args = get_args()
    net = MultiscaleWarpingNet()
    dataset_name = args.dataset
    scale = args.scale
    
    if dataset_name=='LFvideo':
        dataset_train = LFDataset(filename = '/fileserver/haitian/dataset/lf_video_dataset/train_x4_x8.h5', scale = scale)
        dataset_test = LFDataset(filename = '/fileserver/haitian/dataset/lf_video_dataset/test_x4_x8.h5', scale = scale)
        H,W = (320,512)

    elif dataset_name=='Flower':
        dataset_train = FlowerDataset(filename = '/fileserver/haitian/dataset/flower_dataset/train_x4_x8.h5', scale = scale)
        dataset_test = FlowerDataset(filename = '/fileserver/haitian/dataset/flower_dataset/test_x4_x8.h5', scale = scale)
        H,W = (320,512)

    config = {}
    config['dataset_train'] = dataset_train
    config['dataset_test'] = dataset_test
    config['data_displacement_augmentation'] = False
    config['train_data_crop_shape'] = [H,W]
    config['max_iter'] = args.max_iter
    config['snapshot'] = args.snapshot
    config['display'] = args.display
    config['lr'] = args.lr
    config['batch_size'] = args.batch_size
    config['step_size'] = args.step_size
    config['gamma'] = args.gamma
    config['checkpoints_dir'] = args.checkpoints_dir
    config['loss'] = args.loss
    config['checkpoint'] = args.checkpoint
    config['optim'] = args.optim
    config['Dual'] = False

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))
        

    if args.gpu:
        net.cuda()
        #net = nn.DataParallel(net)
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,gpu=args.gpu,config = config)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
