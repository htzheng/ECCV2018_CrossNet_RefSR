import Dataset
import time

import scipy
import scipy.io
import scipy.misc
import scipy.io as sio

import matplotlib.pyplot as plt
from ulti import my_imshow, psnr, psnr_with_mask, epe, save_snapshot, load_snapshot, load_params, save_params, yes_no_promt

import numpy as np

import os
import re

class Solver_forwardwarp(object):
    def __init__(self, dataset_train, dataset_test, model, train_config):
        self.model = model
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.train_config = train_config
        self.save_folder = train_config['savefolder'] + self.get_training_folder()
        self.model.compile()
        return

    def get_training_folder(self):
        training_path = self.model.name + '-lr_'+ str(self.train_config['base_lr'])
        return training_path

    def inference(self, input_list):
        output_list = self.model.fun_test(input_list)
        return output_list

    # def load_model(self):
    #     print 'Load pretrained model'
    #     if (model_sythsis_net=='ResidueWarping' or model_sythsis_net=='AlphaBlending'):
    #         load_params(net['params_all_flow'], model_load_pretrained_filename)
    #     else:
    #         exit('not implemented!')
    #     return

    def __create_dict(self, list1, list2):
        results_dic = {}
        for i in range(len(list1)):
            results_dic[list1[i]] = list2[i]
        return results_dic

    def train(self):
        # load config
        base_lr = self.train_config['base_lr']
        BATCH_NUM = self.train_config['BATCH_NUM']
        lr_decay = self.train_config['lr_decay']
        config_maxiter = self.train_config['maxiter'] + 1
        data_augmentation = self.train_config['train_data_augmentation']
        data_displacement_augmentation = self.train_config['train_data_displacement_augmentation']
        reload_step = self.train_config['reload_step']
        save_folder = self.save_folder

        ## test function
        test_fun = self.model.performance_function
        compare_display = self.model.compare_display
        show_mode = self.model.show_mode

        ## 
        test_step = self.train_config['test_step']
        display_step = 10
        show_step = 500  #2000
        snapshot_step = self.train_config['snapshot_step']
        train_data_crop_shape = self.train_config['train_data_crop_shape']
        # log filename

        if self.train_config['reload_step'] == 0:
            reload_step = -1
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            if os.path.exists(save_folder+'/log_train.txt'):    # if log already exist
                with open(save_folder+'/log_train.txt','r') as f:
                    last_iter = 0
                    for line in f:
                        last_iter = re.findall("[-+]?\d*\.\d+|\d+", line)[0]
                print 'the training log in folder "', save_folder ,'" already exists, last iteration is:' , last_iter
                flag = yes_no_promt('do you still want to write log to that folder? (y/n)\n')
                if flag==False:
                    exit('abort training')

            f_log_train = open(save_folder+'/log_train.txt','w') # train log
            f_log_train_PSNR = open(save_folder+'/log_train_PSNR.txt','w') # train log
            f_log_val_PSNR = open(save_folder+'/log_val_PSNR.txt','w') # val log
            f_log_val_fixed_PSNR = open(save_folder+'/log_val_fixed_PSNR.txt','w') # val log

            # load pretrained model
            if self.train_config['load_pretrained']:
                print 'load pretrained params: ', self.train_config['pretrained_filename']
                if self.train_config['pretrained_params'] == 'all':
                    load_params(self.model.net['params_all'], self.train_config['pretrained_filename'])
                elif self.train_config['pretrained_params'] == 'flow':
                    load_params(self.model.net['params_flow'], self.train_config['pretrained_filename'])

            # compute learning rate scale
            lr_scale = 1.0 
        else:                           # continue training
            if self.train_config['load_pretrained']:
                raw_input('warning: pretraining option is not available for resuming training.')
                exit()
            if not os.path.exists(save_folder+'/'+str(self.train_config['reload_step'])+'.updates'):
                raw_input('warning: cannot resume file: '+ save_folder+'/'+str(self.train_config['reload_step'])+'.updates')
                exit()
            
            # compute learning rate scale
            lr_scale = 1.0 
            for it in range(0, reload_step+1):
                ## learning rate decay
                if (it in lr_decay):
                    lr_scale = lr_scale * lr_decay[it]
                    print 'new lr scale is set to: ', it, lr_scale * base_lr
            
            # load updates
            load_snapshot(self.model.net['params_all'], self.model.T_updates, save_folder+'/'+str(it))

            # modify the new log file, such that the old log is not deleted
            f_log_train = open(save_folder+'/log_train'+str(self.train_config['reload_step'])+'.txt','w') # train log
            f_log_train_PSNR = open(save_folder+'/log_train_PSNR'+str(self.train_config['reload_step'])+'.txt','w') # train log
            f_log_val_PSNR = open(save_folder+'/log_val_PSNR'+str(self.train_config['reload_step'])+'.txt','w') # val log
            f_log_val_fixed_PSNR = open(save_folder+'/log_val_fixed_PSNR'+str(self.train_config['reload_step'])+'.txt','w') # val log

        # tt = time.time()
        # for it in range(170000):
        #     if it%1000 == 0:
        #         print 'empty loop: ',it,' time: ', time.time()-tt
        #         tt = time.time()
        #     data_buffer_train = self.dataset_train.nextBatch_new_fake(batchsize=BATCH_NUM, shuffle=True, view_mode = 'Random', augmentation = True, offset_augmentation=data_displacement_augmentation, crop_shape = train_data_crop_shape)

        ##### train
        tt = time.time()
        for it in range(reload_step+1, config_maxiter):
            ## learning rate decay
            if (it in lr_decay):
                lr_scale = lr_scale * lr_decay[it]
                print it, lr_scale * base_lr

            ## generate minibatch
            t_generator = time.time()
            data_buffer_train = self.dataset_train.nextBatch_new(batchsize=BATCH_NUM, shuffle=True, view_mode = 'Random', augmentation = True, offset_augmentation=data_displacement_augmentation, crop_shape = train_data_crop_shape)
            train_input_list  = self.dataset_test.append_list(data_buffer_train, self.model.list_train_input)
            train_input_list.append(float(lr_scale * base_lr))  #['input_img1_LR','input_img2_HR']
            # print 'generator processing time: ', time.time() - t_generator

            ## 
            # plt.subplot(321)
            # my_imshow(data_buffer_train['input_img1_LR'][np.newaxis,0,:,:,:])
            # plt.subplot(322)
            # my_imshow(data_buffer_train['input_img2_HR'][np.newaxis,0,:,:,:])
            # plt.subplot(323)
            # my_imshow(data_buffer_train['input_img1_HR'][np.newaxis,0,:,:,:])
            # # plt.subplot(324)
            # # my_imshow(data_buffer_train['input_SRResNet_NN_LR'][np.newaxis,0,:,:,:])
            # # plt.subplot(325)
            # # my_imshow(data_buffer_train['input_SRResNet_NN_REF'][np.newaxis,0,:,:,:])
            # plt.show()

            ## test
            # test_input_list_train = self.dataset_test.append_list(data_buffer_train, self.model.list_test_input)
            # print data_buffer_train.keys()
            # print self.model.list_test_input
            # print test_input_list_train
            # results_train = self.model.fun_test(*tuple(test_input_list_train))
            # print results_train[0].shape, results_train[1].shape
            # exit()

            # train_input_list  = self.dataset_test.append_list(data_buffer_train, ['input_img1_LR','input_img2_HR','input_img1_HR'])
            # print train_input_list[0].shape, train_input_list[1].shape, train_input_list[2].shape
            # print self.model.fun_test2(*tuple(train_input_list))[0]
            # exit()

            ## train and record loss
            t_trainor = time.time()
            loss = self.model.fun_train(*tuple(train_input_list))
            f_log_train.write(str(it)+', '+str(loss)+'\n')
            f_log_train.flush()
            # print 'train processing time: ', time.time() - t_trainor

            ## test
            if (it % test_step ==0):
                ####### train set
                data_buffer_test_train = data_buffer_train 
                test_input_list_train = self.dataset_test.append_list(data_buffer_test_train, self.model.list_test_input)
                results_train = self.model.fun_test(*tuple(test_input_list_train))
                # create dictionary
                results_dic_train = self.__create_dict(self.model.list_test_output, results_train)

                ## compute PSNR
                psnr_val_train = test_fun(*tuple(self.dataset_test.append_list(results_dic_train, self.model.list_compare )))
                print compare_display, '(train)', psnr_val_train

                ## record
                f_log_train_PSNR.write(str(it)+', '+self.convert_string(psnr_val_train)+str(lr_scale * base_lr)+'\n')
                f_log_train_PSNR.flush()

                ####### val set (fixed index and random view)
                # data_buffer_test = self.dataset_test.nextBatch_new(batchsize=BATCH_NUM, shuffle=False, view_mode = 'Random', augmentation = False, index_inc = False, crop_shape = train_data_crop_shape)
                # test_input_list = self.dataset_test.append_list(data_buffer_test, self.model.list_test_input)
                # results = self.model.fun_test(*tuple(test_input_list))
                # # create dictionary
                # results_dic = self.__create_dict(self.model.list_test_output, results)y
                # # compute PSNR 
                # psnr_val = test_fun(*tuple(self.dataset_test.append_list(results_dic, self.model.list_compare ))) 
                # print compare_display, '(test)', psnr_val
                # # record
                # f_log_val_PSNR.write(str(it)+', '+self.convert_string(psnr_val)+str(lr_scale * base_lr)+'\n')
                # f_log_val_PSNR.flush()

                if False:  #show_mode == 'psnr' or show_mode == 'psnr_with_mask':
                    ####### val set (fixed index and fixed view)
                    data_buffer_test_fixed = self.dataset_test.nextBatch_new(batchsize=BATCH_NUM, shuffle=False, view_mode = 'Fixed-inv', augmentation = False, offset_augmentation=data_displacement_augmentation, index_inc = True, crop_shape = train_data_crop_shape)
                    test_input_list_fixed = self.dataset_test.append_list(data_buffer_test_fixed, self.model.list_test_input)
                    results_fixed = self.model.fun_test(*tuple(test_input_list_fixed))
                    # create dictionary
                    results_dic_fixed = self.__create_dict(self.model.list_test_output, results_fixed)
                    # compute PSNR 
                    psnr_val_fixed = test_fun(*tuple(self.dataset_test.append_list(results_dic_fixed, self.model.list_compare )))
                    print compare_display, '(test_fix)', psnr_val_fixed
                    # record
                    f_log_val_fixed_PSNR.write(str(it)+', '+self.convert_string(psnr_val_fixed)+str(lr_scale * base_lr)+'\n')
                    f_log_val_fixed_PSNR.flush()

                ####### display
                if False: #(it % show_step == 0):
                    print 'save image to', save_folder
                    self.show_result(results_dic_fixed, save_folder, it, show_mode = show_mode, folder='Fixed_')

            if (it % display_step)==0:
                print it,loss,' time: ',time.time()-tt, ' lr:', lr_scale * base_lr
                tt = time.time()

            if (it % snapshot_step)==0:
                print 'saving snapshot at iter: ', it
                save_snapshot(self.model.net['params_all'], self.model.T_updates, save_folder+'/'+str(it))

        return

    def test(self):
        # not implemented
        return

    def convert_string(self, l):
        if type(l) is tuple:
            return ''.join(str(e)+', ' for e in l)
        else:
            return str(l)+', '

    def show_result(self, data_buffer, save_folder, iteration, show_mode, folder = ''):
        show = False
        
        for key, value in data_buffer.iteritems():
            print key

        if show_mode == 'psnr':
            f = data_buffer['flow_12']
            
            if show:
                plt.subplot(321)
                my_imshow(data_buffer['input_img1_HR'][np.newaxis,0,:,:,:])
                plt.subplot(322)
                my_imshow(data_buffer['input_img2_HR'][np.newaxis,0,:,:,:])
                plt.subplot(323)
                my_imshow(data_buffer['warp_21'][np.newaxis,0,:,:,:])
                # plt.subplot(324)
                # my_imshow(data_buffer['hole_21'][np.newaxis,0,:,:,:])
                # my_imshow(5 *np.abs(data_buffer['warp_21']-data_buffer['input_img1_HR']) [np.newaxis,0,:,:,:])
                f_visual_limit = 5.0
                plt.subplot(325)
                plt.imshow(f[0,0,:,:], vmin=-f_visual_limit, vmax=f_visual_limit, interpolation="nearest")
                plt.subplot(326)
                plt.imshow(f[0,1,:,:], vmin=-f_visual_limit, vmax=f_visual_limit, interpolation="nearest")

                plt.show(False)
                plt.draw()
                plt.pause(0.01)

            scipy.misc.toimage(np.squeeze(np.transpose(data_buffer['input_img1_HR'][np.newaxis,0,:,:,:],axes=(0,2,3,1))), cmin=0.0, cmax=1.0).save(save_folder+'/'+folder+'img1_HR_c_'+str(iteration)+'.png')
            scipy.misc.toimage(np.squeeze(np.transpose(data_buffer['input_img2_HR'][np.newaxis,0,:,:,:],axes=(0,2,3,1))), cmin=0.0, cmax=1.0).save(save_folder+'/'+folder+'img2_HR_c_'+str(iteration)+'.png')
            # scipy.misc.toimage(np.squeeze(np.transpose(data_buffer['input_img1_LR'][np.newaxis,0,:,:,:],axes=(0,2,3,1))), cmin=0.0, cmax=1.0).save(save_folder+'/'+folder+'img1_LR_c_'+str(iteration)+'.png')
            scipy.misc.toimage(np.squeeze(np.transpose(data_buffer['warp_21'][np.newaxis,0,:,:,:],axes=(0,2,3,1))), cmin=0.0, cmax=1.0).save(save_folder+'/'+folder+'warp_21_'+str(iteration)+'.png')
            scipy.misc.toimage(np.squeeze(np.transpose(data_buffer['sythsis_output'][np.newaxis,0,:,:,:],axes=(0,2,3,1))), cmin=0.0, cmax=1.0).save(save_folder+'/'+folder+'sythsis_output_'+str(iteration)+'.png')
            # scipy.misc.toimage(np.squeeze(np.transpose(np.abs(data_buffer['warp_21']-data_buffer['input_img1_HR'])[np.newaxis,0,:,:,:],axes=(0,2,3,1))), cmin=0.0, cmax=1.0).save(save_folder+'/'+folder+'err_'+str(iteration)+'.png')

            if 'flow_12' in data_buffer.keys():
                sio.savemat(save_folder+'/'+folder+'flow_'+str(iteration)+'.mat', {'flow': data_buffer['flow_12'][0,:,:,:]} )

        elif show_mode == 'psnr_with_mask':
            f = data_buffer['flow_21']

            if show:
                plt.subplot(321)
                my_imshow(data_buffer['input_img1_HR'][np.newaxis,0,:,:,:])
                plt.subplot(322)
                my_imshow(data_buffer['input_img2_HR'][np.newaxis,0,:,:,:])
                plt.subplot(323)
                my_imshow(data_buffer['warp_21'][np.newaxis,0,:,:,:])
                plt.subplot(324)
                my_imshow(data_buffer['hole_21'][np.newaxis,0,:,:,:])

                f_visual_limit = 5.0
                plt.subplot(325)
                plt.imshow(f[0,0,:,:], vmin=-f_visual_limit, vmax=f_visual_limit, interpolation="nearest")
                plt.subplot(326)
                plt.imshow(f[0,1,:,:], vmin=-f_visual_limit, vmax=f_visual_limit, interpolation="nearest")
                plt.show(False)
                plt.draw()
                plt.pause(0.01)

            scipy.misc.toimage(np.squeeze(np.transpose(data_buffer['input_img1_HR'][np.newaxis,0,:,:,:],axes=(0,2,3,1))), cmin=0.0, cmax=1.0).save(save_folder+'/'+folder+'img1_HR_c_'+str(iteration)+'.png')
            scipy.misc.toimage(np.squeeze(np.transpose(data_buffer['input_img2_HR'][np.newaxis,0,:,:,:],axes=(0,2,3,1))), cmin=0.0, cmax=1.0).save(save_folder+'/'+folder+'img2_HR_c_'+str(iteration)+'.png')
            # scipy.misc.toimage(np.squeeze(np.transpose(data_buffer['input_img1_LR'][np.newaxis,0,:,:,:],axes=(0,2,3,1))), cmin=0.0, cmax=1.0).save(save_folder+'/'+folder+'img1_LR_c_'+str(iteration)+'.png')
            scipy.misc.toimage(np.squeeze(np.transpose(data_buffer['warp_21'][np.newaxis,0,:,:,:],axes=(0,2,3,1))), cmin=0.0, cmax=1.0).save(save_folder+'/'+folder+'warp_21_'+str(iteration)+'.png')
            scipy.misc.toimage(np.squeeze(np.transpose(data_buffer['sythsis_output'][np.newaxis,0,:,:,:],axes=(0,2,3,1))), cmin=0.0, cmax=1.0).save(save_folder+'/'+folder+'sythsis_output_'+str(iteration)+'.png')
            # scipy.misc.toimage(np.squeeze(np.transpose(np.abs(data_buffer['warp_21']-data_buffer['input_img1_HR'])[np.newaxis,0,:,:,:],axes=(0,2,3,1))), cmin=0.0, cmax=1.0).save(save_folder+'/'+folder+'err_'+str(iteration)+'.png')

            # scipy.misc.toimage(data_buffer['feat_img1_LR'][0,0,:,:], cmin=np.min(data_buffer['feat_img1_LR'][0,0,:,:]), cmax=np.max(data_buffer['feat_img1_LR'][0,0,:,:])).save(save_folder+'/'+folder+'feat1_LR_'+str(iteration)+'.png')
            # scipy.misc.toimage(data_buffer['feat_img2_HR'][0,0,:,:], cmin=np.min(data_buffer['feat_img2_HR'][0,0,:,:]), cmax=np.max(data_buffer['feat_img2_HR'][0,0,:,:])).save(save_folder+'/'+folder+'feat2_HR_'+str(iteration)+'.png')
            # scipy.misc.toimage(data_buffer['corr_21'][0,0,:,:], cmin=np.min(data_buffer['corr_21'][0,0,:,:]), cmax=np.max(data_buffer['corr_21'][0,0,:,:])).save(save_folder+'/'+folder+'corr_21_'+str(iteration)+'.png')
            scipy.misc.toimage(data_buffer['hole_21'][0,0,:,:], cmin=0., cmax=1.).save(save_folder+'/'+folder+'hole_21_'+str(iteration)+'.png')
            scipy.misc.toimage(data_buffer['V_21'][0,0,:,:], cmin=0., cmax=1.).save(save_folder+'/'+folder+'V_21_'+str(iteration)+'.png')
            scipy.misc.toimage(data_buffer['W_21'][0,0,:,:], cmin=0., cmax=1.).save(save_folder+'/'+folder+'W_21_'+str(iteration)+'.png')


            # if 'input_img1_SR' in data_buffer.keys():
            #     scipy.misc.toimage(data_buffer['input_img1_SR'][0,:,:,:], cmin=0, cmax=1).save(save_folder+'/'+folder+'img1_SR_'+str(iteration)+'.png')
            # if 'hole_21' in data_buffer.keys():
            #     scipy.misc.toimage(data_buffer['hole_21'][0,0,:,:], cmin=0, cmax=1).save(save_folder+'/'+folder+'hole_'+str(iteration)+'.png')
            # if 'V_21' in data_buffer.keys():
            #     scipy.misc.toimage(data_buffer['V_21'][0,0,:,:], cmin=0, cmax=2.0).save(save_folder+'/'+folder+'V_'+str(iteration)+'.png')  #np.max(results_dic_fixed['V_21'][0,0,:,:])
            # if 'W_21' in data_buffer.keys():
            #     scipy.misc.toimage(data_buffer['W_21'][0,0,:,:], cmin=-1.0, cmax=1.0).save(save_folder+'/'+folder+'W_'+str(iteration)+'.png')  #np.max(results_dic_fixed['V_21'][0,0,:,:])

            if 'flow_21' in data_buffer.keys():
                sio.savemat(save_folder+'/'+folder+'flow_'+str(iteration)+'.mat', {'flow': data_buffer['flow_21'][0,:,:,:]} )
        elif show_mode == 'psnr_with_mask_bidir':
            f = data_buffer['flow_21']

            if show:
                plt.subplot(321)
                my_imshow(data_buffer['input_img1_HR'][np.newaxis,0,:,:,:])
                plt.subplot(322)
                my_imshow(data_buffer['input_img2_HR'][np.newaxis,0,:,:,:])
                plt.subplot(323)
                my_imshow(data_buffer['warp_21'][np.newaxis,0,:,:,:])
                plt.subplot(324)
                my_imshow(data_buffer['hole_1'][np.newaxis,0,:,:,:])
                # my_imshow(5 *np.abs(data_buffer['warp_21']-data_buffer['input_img1_HR']) [np.newaxis,0,:,:,:])
                f_visual_limit = 5.0
                plt.subplot(325)
                plt.imshow(f[0,0,:,:], vmin=-f_visual_limit, vmax=f_visual_limit, interpolation="nearest")
                plt.subplot(326)
                plt.imshow(f[0,1,:,:], vmin=-f_visual_limit, vmax=f_visual_limit, interpolation="nearest")

                # plt.show()
                plt.show(False)
                plt.draw()
                plt.pause(0.01)

            scipy.misc.toimage(np.squeeze(np.transpose(data_buffer['input_img1_HR'][np.newaxis,0,:,:,:],axes=(0,2,3,1))), cmin=0.0, cmax=1.0).save(save_folder+'/'+folder+'img1_HR_c_'+str(iteration)+'.png')
            scipy.misc.toimage(np.squeeze(np.transpose(data_buffer['input_img2_HR'][np.newaxis,0,:,:,:],axes=(0,2,3,1))), cmin=0.0, cmax=1.0).save(save_folder+'/'+folder+'img2_HR_c_'+str(iteration)+'.png')
            # scipy.misc.toimage(np.squeeze(np.transpose(data_buffer['input_img1_LR'][np.newaxis,0,:,:,:],axes=(0,2,3,1))), cmin=0.0, cmax=1.0).save(save_folder+'/'+folder+'img1_LR_c_'+str(iteration)+'.png')
            scipy.misc.toimage(np.squeeze(np.transpose(data_buffer['warp_21'][np.newaxis,0,:,:,:],axes=(0,2,3,1))), cmin=0.0, cmax=1.0).save(save_folder+'/'+folder+'warp_21_'+str(iteration)+'.png')
            scipy.misc.toimage(np.squeeze(np.transpose(data_buffer['sythsis_output'][np.newaxis,0,:,:,:],axes=(0,2,3,1))), cmin=0.0, cmax=1.0).save(save_folder+'/'+folder+'sythsis_output_'+str(iteration)+'.png')
            # scipy.misc.toimage(np.squeeze(np.transpose(np.abs(data_buffer['warp_21']-data_buffer['input_img1_HR'])[np.newaxis,0,:,:,:],axes=(0,2,3,1))), cmin=0.0, cmax=1.0).save(save_folder+'/'+folder+'err_'+str(iteration)+'.png')

            # scipy.misc.toimage(data_buffer['feat_img1_LR'][0,0,:,:], cmin=np.min(data_buffer['feat_img1_LR'][0,0,:,:]), cmax=np.max(data_buffer['feat_img1_LR'][0,0,:,:])).save(save_folder+'/'+folder+'feat1_LR_'+str(iteration)+'.png')
            # scipy.misc.toimage(data_buffer['feat_img2_HR'][0,0,:,:], cmin=np.min(data_buffer['feat_img2_HR'][0,0,:,:]), cmax=np.max(data_buffer['feat_img2_HR'][0,0,:,:])).save(save_folder+'/'+folder+'feat2_HR_'+str(iteration)+'.png')
            # scipy.misc.toimage(data_buffer['corr_21'][0,0,:,:], cmin=np.min(data_buffer['corr_21'][0,0,:,:]), cmax=np.max(data_buffer['corr_21'][0,0,:,:])).save(save_folder+'/'+folder+'corr_21_'+str(iteration)+'.png')
            scipy.misc.toimage(data_buffer['hole_1'][0,0,:,:], cmin=0., cmax=1.).save(save_folder+'/'+folder+'hole_1_'+str(iteration)+'.png')
            scipy.misc.toimage(data_buffer['V_1'][0,0,:,:], cmin=0., cmax=1.).save(save_folder+'/'+folder+'V_1_'+str(iteration)+'.png')
            # scipy.misc.toimage(data_buffer['W_21'][0,0,:,:], cmin=0., cmax=1.).save(save_folder+'/'+folder+'W_21_'+str(iteration)+'.png')


            # if 'input_img1_SR' in data_buffer.keys():
            #     scipy.misc.toimage(data_buffer['input_img1_SR'][0,:,:,:], cmin=0, cmax=1).save(save_folder+'/'+folder+'img1_SR_'+str(iteration)+'.png')
            # if 'hole_21' in data_buffer.keys():
            #     scipy.misc.toimage(data_buffer['hole_21'][0,0,:,:], cmin=0, cmax=1).save(save_folder+'/'+folder+'hole_'+str(iteration)+'.png')
            # if 'V_21' in data_buffer.keys():
            #     scipy.misc.toimage(data_buffer['V_21'][0,0,:,:], cmin=0, cmax=2.0).save(save_folder+'/'+folder+'V_'+str(iteration)+'.png')  #np.max(results_dic_fixed['V_21'][0,0,:,:])
            # if 'W_21' in data_buffer.keys():
            #     scipy.misc.toimage(data_buffer['W_21'][0,0,:,:], cmin=-1.0, cmax=1.0).save(save_folder+'/'+folder+'W_'+str(iteration)+'.png')  #np.max(results_dic_fixed['V_21'][0,0,:,:])

            if 'flow_21' in data_buffer.keys():
                sio.savemat(save_folder+'/'+folder+'flow_'+str(iteration)+'.mat', {'flow': data_buffer['flow_21'][0,:,:,:]} )

        elif show_mode == 'epe':
            'flow', 'input_flow_cent', 'input_LR','input_REF'

            f = data_buffer['flow']
            f_gt = data_buffer['input_flow_cent']

            plt.subplot(321)
            my_imshow(data_buffer['input_LR'][np.newaxis,0,:,:,:])
            plt.subplot(322)
            my_imshow(data_buffer['input_REF'][np.newaxis,0,:,:,:])

            f_visual_limit = 5.0
            plt.subplot(323)
            plt.imshow(f_gt[0,0,:,:], vmin=-f_visual_limit, vmax=f_visual_limit, interpolation="nearest")
            plt.subplot(324)
            plt.imshow(f_gt[0,1,:,:], vmin=-f_visual_limit, vmax=f_visual_limit, interpolation="nearest")

            plt.subplot(325)
            plt.imshow(f[0,0,:,:], vmin=-f_visual_limit, vmax=f_visual_limit, interpolation="nearest")
            plt.subplot(326)
            plt.imshow(f[0,1,:,:], vmin=-f_visual_limit, vmax=f_visual_limit, interpolation="nearest")

            plt.show(False)
            plt.draw()
            plt.pause(0.01)

            # scipy.misc.toimage(np.squeeze(np.transpose(np.abs(data_buffer['HR_output']-data_buffer['input_HR_cent'])[np.newaxis,0,:,:,:],axes=(0,2,3,1))), cmin=0.0, cmax=1.0).save(save_folder+'/err_'+str(iteration)+'.png')


            # scipy.misc.toimage(data_buffer['feat_LR'][0,0,:,:], cmin=np.min(data_buffer['feat_LR'][0,0,:,:]), cmax=np.max(data_buffer['feat_LR'][0,0,:,:])).save(save_folder+'/feat_LR'+str(iteration)+'.png')
            # scipy.misc.toimage(data_buffer['feat_REF'][0,0,:,:], cmin=np.min(data_buffer['feat_REF'][0,0,:,:]), cmax=np.max(data_buffer['feat_REF'][0,0,:,:])).save(save_folder+'/feat_REF'+str(iteration)+'.png')
            # scipy.misc.toimage(data_buffer['corr'][0,0,:,:], cmin=np.min(data_buffer['corr'][0,0,:,:]), cmax=np.max(data_buffer['corr'][0,0,:,:])).save(save_folder+'/corr'+str(iteration)+'.png')
        return

class Inferencer(object):
    def __init__(self, dataset_test, model, preload_model, savepath):
        self.dataset_test = dataset_test
        self.model = model
        self.preload_model = preload_model
        self.savepath = savepath
        print model.net.keys()
        print model.net['params_all']
        load_snapshot(self.model.net['params_all'], self.model.T_updates, preload_model)


    def __create_dict(self, list1, list2):
        results_dic = {}
        for i in range(len(list1)):
            results_dic[list1[i]] = list2[i]
        return results_dic

    # def save_result(self):
    #     save_folder = self.savepath

    #     additional_folder = None 
    #     # additional_folder = '/fileserver/haitian/haitian_backup/HT_sr/SRResNet_After_BMVC/result_LF-(-7,-7)'
    #     # additional_folder = '/fileserver/haitian/haitian_backup/HT_sr/SRResNet_After_BMVC/result_LF-(-3,-3)-train'

    #     for i in range(268+1):
    #         print i
    #         data_buffer_test_fixed = self.dataset_test.nextBatch_new(batchsize=1, shuffle=False, view_mode = 'Fixed-inv', augmentation = False, index_inc = True, crop_shape = (320,512) )
    #         test_input_list_fixed = self.dataset_test.append_list(data_buffer_test_fixed, self.model.list_test_input)
    #         results_fixed = self.model.fun_test(*tuple(test_input_list_fixed))
    #         # create dictionary
    #         results_dic_fixed = self.__create_dict(self.model.list_test_output, results_fixed)

    #         if not additional_folder is None:
    #             scipy.misc.toimage(np.squeeze(np.transpose(results_dic_fixed['input_img1_HR_c'][np.newaxis,0,:,:,:],axes=(0,2,3,1))), cmin=0.0, cmax=1.0).save(additional_folder+'/GT/'+str(i)+'.png')
    #             scipy.misc.toimage(np.squeeze(np.transpose(results_dic_fixed['input_img2_HR_c'][np.newaxis,0,:,:,:],axes=(0,2,3,1))), cmin=0.0, cmax=1.0).save(additional_folder+'/REF/'+str(i)+'.png')
           
    #         if 'warp_21' in results_dic_fixed.keys():
    #             scipy.misc.toimage(np.squeeze(np.transpose(results_dic_fixed['warp_21'][np.newaxis,0,:,:,:],axes=(0,2,3,1))), cmin=0.0, cmax=1.0).save(save_folder+'/warp_'+str(i)+'.png')

    #         scipy.misc.toimage(np.squeeze(np.transpose(results_dic_fixed['sythsis_output'][np.newaxis,0,:,:,:],axes=(0,2,3,1))), cmin=0.0, cmax=1.0).save(save_folder+'/'+str(i)+'.png')
    #         if 'hole_21' in results_dic_fixed.keys():
    #             scipy.misc.toimage(results_dic_fixed['hole_21'][0,0,:,:], cmin=0, cmax=1).save(save_folder+'/hole_'+str(i)+'.png')
    #         if 'V_21' in results_dic_fixed.keys():
    #             scipy.misc.toimage(results_dic_fixed['V_21'][0,0,:,:], cmin=0, cmax=2.0).save(save_folder+'/V_'+str(i)+'.png')  #np.max(results_dic_fixed['V_21'][0,0,:,:])
    #         if 'W_21' in results_dic_fixed.keys():
    #             scipy.misc.toimage(results_dic_fixed['W_21'][0,0,:,:], cmin=-1.0, cmax=1.0).save(save_folder+'/W_'+str(i)+'.png')  #np.max(results_dic_fixed['V_21'][0,0,:,:])
    #         if 'flow_12' in results_dic_fixed.keys():
    #             sio.savemat(save_folder+'/flow12_'+str(i)+'.mat', {'flow': results_dic_fixed['flow_12'][0,:,:,:]} )

    def save_result(self):
        save_folder = self.savepath
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        additional_folder = None
        for i in range(268+1):
            print i
            data_buffer_test_fixed = self.dataset_test.nextBatch_new(batchsize=1, shuffle=False, view_mode = 'Fixed-inv', augmentation = False, index_inc = True, crop_shape = (320,512) )
            test_input_list_fixed = self.dataset_test.append_list(data_buffer_test_fixed, self.model.list_test_input)
            results_fixed = self.model.fun_test(*tuple(test_input_list_fixed))
            # create dictionary
            results_dic_fixed = self.__create_dict(self.model.list_test_output, results_fixed)
            scipy.misc.toimage(np.squeeze(np.transpose(results_dic_fixed['sythsis_output'][np.newaxis,0,:,:,:],axes=(0,2,3,1))), cmin=0.0, cmax=1.0).save(save_folder+'/'+str(i)+'.png')
            

        time_b = time.time()
        ## brightness changes
        img_aug = img_aug + config_brightness_changes
        ## multiplicative color changes
        img_aug = img_aug * config_multiplicative_color_changes
        time_b = time.time() - time_b

    def save_results_lfvideo(self, keyword = 'a-method', maxsize = 269, visualize_flow = True):
        time_list = []

        save_path = self.savepath
        for v in range(1,8):
            save_folder = save_path+'/LR('+str(v)+','+str(v)+')-REF(0,0)/' + keyword
            print 'save_folder: ', save_folder
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)

            for i in range(maxsize):    
                # print i
                data_buffer_test_fixed = self.dataset_test.nextBatch_new(batchsize=1, shuffle=False, view_mode = 'specified', specified_view = v, augmentation = False, index_inc = True, crop_shape = (320,512) )
                test_input_list_fixed = self.dataset_test.append_list(data_buffer_test_fixed, self.model.list_test_input)
                time_1 = time.time()
                results_fixed = self.model.fun_test(*tuple(test_input_list_fixed))
                time_2 = time.time()
                time_list.append(time_2-time_1)

                # create dictionary
                results_dic_fixed = self.__create_dict(self.model.list_test_output, results_fixed)
                scipy.misc.toimage(np.squeeze(np.transpose(results_dic_fixed['sythsis_output'][np.newaxis,0,:,:,:],axes=(0,2,3,1))), cmin=0.0, cmax=1.0).save(save_folder+'/'+str(i)+'.png')
                if visualize_flow:
                    sio.savemat(save_folder+'/'+'flow_'+str(i)+'.mat', {'warp': results_dic_fixed['warp_21'][0,:,:,:],
                                                                        'HR2_conv4': results_dic_fixed['HR2_conv4'][0,0,:,:],
                                                                        'warp_21_conv4': results_dic_fixed['warp_21_conv4'][0,0,:,:],
                                                                        'HR2_conv3': results_dic_fixed['HR2_conv3'][0,0,:,:],
                                                                        'warp_21_conv3': results_dic_fixed['warp_21_conv3'][0,0,:,:],
                                                                        'HR2_conv2': results_dic_fixed['HR2_conv2'][0,0,:,:],
                                                                        'warp_21_conv2': results_dic_fixed['warp_21_conv2'][0,0,:,:],
                                                                        'HR2_conv1': results_dic_fixed['HR2_conv1'][0,0,:,:],
                                                                        'warp_21_conv1': results_dic_fixed['warp_21_conv1'][0,0,:,:],
                                                                        'flow0': results_dic_fixed['flow_12'][0,:,:,:],
                                                                        'flow6': results_dic_fixed['flow6'][0,:,:,:],
                                                                        'flow5': results_dic_fixed['flow5'][0,:,:,:],
                                                                        'flow4': results_dic_fixed['flow4'][0,:,:,:],
                                                                        'flow3': results_dic_fixed['flow3'][0,:,:,:],
                                                                        'flow2': results_dic_fixed['flow2'][0,:,:,:],
                                                                        'flow1': results_dic_fixed['flow1'][0,:,:,:]})

            # print 'time list: ', time_list
            print 'avg time: ', sum(time_list) / float(len(time_list))


    def save_results_supplementary(self, maxsize = 0, savepath = '', keyword = '', crop_shape=None):
        print savepath
        def write_image(img_tensor, filename):
            scipy.misc.toimage(np.squeeze(np.transpose(img_tensor,axes=(0,2,3,1))), cmin=0.0, cmax=1.0).save(filename)
            return

        if not os.path.exists(savepath):
            os.mkdir(savepath)
        if not os.path.exists(savepath+'REF/'):
            os.mkdir(savepath+'REF/')
        if not os.path.exists(savepath+'LR/'):
            os.mkdir(savepath+'LR/')
        if not os.path.exists(savepath+'LR_upsample/'):
            os.mkdir(savepath+'LR_upsample/')
        if not os.path.exists(savepath+'MDSR/'):
            os.mkdir(savepath+'MDSR/')
        if not os.path.exists(savepath+'GT/'):
            os.mkdir(savepath+'GT/')
        if not os.path.exists(savepath+'CorresNet/'):
            os.mkdir(savepath+'CorresNet/')
        
        for img_num in range(0,maxsize,9):
            print img_num
            save_path = self.savepath
            buffer_0_0 = self.dataset_test.get_image(img_num,0,0, crop_shape = crop_shape)
            REF = buffer_0_0['HR']
            write_image(REF, savepath+'REF/'+str(img_num)+'.png')

            for dx in range(0,8):
                for dy in range(0,8):
                    buffer_y_x = self.dataset_test.get_image(img_num,dy,dx)

                    HR = buffer_y_x['HR']
                    LR = buffer_y_x['LR']
                    MDSR = buffer_y_x['MDSR']
                    LR_upsample = buffer_y_x['LR_upsample']

                    write_image(LR,savepath+'LR/'+str(img_num)+'_'+str(dy)+'_'+str(dx)+'.png')
                    write_image(HR,savepath+'GT/'+str(img_num)+'_'+str(dy)+'_'+str(dx)+'.png')
                    write_image(MDSR,savepath+'MDSR/'+str(img_num)+'_'+str(dy)+'_'+str(dx)+'.png')
                    write_image(LR_upsample,savepath+'LR_upsample/'+str(img_num)+'_'+str(dy)+'_'+str(dx)+'.png')

                    results = self.model.fun_inference(LR_upsample, REF, MDSR)
                    results = results[0]
                    write_image(results,savepath+'CorresNet/'+str(img_num)+'_'+str(dy)+'_'+str(dx)+'.png')

        return

    def inference(self, input_list):
        output_list = self.model.fun_test(input_list)
        return output_list