import Dataset
import time

import scipy
import scipy.io
import scipy.misc

import matplotlib.pyplot as plt
from ulti import my_imshow, psnr, epe, save_snapshot, load_snapshot, load_params, save_params, yes_no_promt

import numpy as np

import os
import re

class Solver(object):
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
        config_maxiter = self.train_config['maxiter']
        data_augmentation = self.train_config['train_data_augmentation']
        reload_step = self.train_config['reload_step']

        save_folder = self.save_folder

        ## test function
        if (self.train_config['test_mode']=='epe'):
            test_fun = epe
            test_display = '   EPE: '
            show_mode = 'epe'
        elif (self.train_config['test_mode']=='psnr'):
            test_fun = psnr
            test_display = '  PSNR: '
            show_mode = 'psnr'

        ## 
        test_step = 5
        display_step = 10
        show_step = 2000
        snapshot_step = self.train_config['snapshot_step']
        dataset_residue = self.train_config['dataset_residue']
        # log filename

        if self.train_config['reload_step'] == 0:
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
                load_params(self.model.net['params_all'], self.train_config['pretrained_filename'])
            
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

        ##### train
        tt = time.time()
        for it in range(reload_step+1, config_maxiter):
            ## learning rate decay
            if (it in lr_decay):
                lr_scale = lr_scale * lr_decay[it]
                print it, lr_scale * base_lr

            ## generate minibatch
            data_buffer_train = self.dataset_train.nextBatch(batchsize=BATCH_NUM, shuffle=True, view_mode = 'Random', residue = dataset_residue, augmentation = True)

            train_input_list = self.dataset_test.append_list(data_buffer_train, self.model.list_train_input)
            train_input_list.append(lr_scale * base_lr)

            ## 
            # plt.subplot(321)
            # my_imshow(data_buffer_train['input_LR'][np.newaxis,0,:,:,:])
            # plt.subplot(322)
            # my_imshow(data_buffer_train['input_REF'][np.newaxis,0,:,:,:])
            # plt.subplot(323)
            # my_imshow(data_buffer_train['input_HR'][np.newaxis,0,:,:,:])
            # plt.subplot(324)
            # my_imshow(data_buffer_train['input_SRResNet_NN_LR'][np.newaxis,0,:,:,:])
            # plt.subplot(325)
            # my_imshow(data_buffer_train['input_SRResNet_NN_REF'][np.newaxis,0,:,:,:])
            # plt.show()

            ## train and record loss
            loss = self.model.fun_train(*tuple(train_input_list))
            f_log_train.write(str(it)+', '+str(loss)+'\n')
            f_log_train.flush()


            ## test
            if (it % test_step ==0):
                ####### train set
                data_buffer_test_train = data_buffer_train     # self.dataset_train.nextBatch(batchsize=BATCH_NUM, shuffle=True, view_mode = 'Random', residue = False, augmentation = True)
                test_input_list_train = self.dataset_test.append_list(data_buffer_test_train, self.model.list_test_input)
                results_train = self.model.fun_test(*tuple(test_input_list_train))
                # create dictionary
                results_dic_train = self.__create_dict(self.model.list_test_output, results_train)
                ## compute PSNR
                psnr_val_train = test_fun(*tuple(self.dataset_test.append_list(results_dic_train, self.model.list_compare_list )))
                print test_display, psnr_val_train

                ## record
                f_log_train_PSNR.write(str(it)+', '+str(psnr_val_train)+', '+str(lr_scale * base_lr)+'\n')
                f_log_train_PSNR.flush()

                ####### val set (fixed index and random view)
                data_buffer_test = self.dataset_test.nextBatch(batchsize=BATCH_NUM, shuffle=False, view_mode = 'Random', residue = dataset_residue, augmentation = False, index_inc = False)
                test_input_list = self.dataset_test.append_list(data_buffer_test, self.model.list_test_input)
                results = self.model.fun_test(*tuple(test_input_list))
                # create dictionary
                results_dic = self.__create_dict(self.model.list_test_output, results)
                # compute PSNR 
                psnr_val = test_fun(*tuple(self.dataset_test.append_list(results_dic, self.model.list_compare_list )))
                print test_display, psnr_val
                # record
                f_log_val_PSNR.write(str(it)+', '+str(psnr_val)+', '+str(lr_scale * base_lr)+'\n')
                f_log_val_PSNR.flush()

                if show_mode == 'psnr':
                    ####### val set (fixed index and fixed view)
                    data_buffer_test_fixed = self.dataset_test.nextBatch(batchsize=BATCH_NUM, shuffle=False, view_mode = 'Fixed', residue = dataset_residue, augmentation = False, index_inc = True)
                    test_input_list_fixed = self.dataset_test.append_list(data_buffer_test_fixed, self.model.list_test_input)
                    results_fixed = self.model.fun_test(*tuple(test_input_list_fixed))
                    # create dictionary
                    results_dic_fixed = self.__create_dict(self.model.list_test_output, results_fixed)
                    # compute PSNR 
                    psnr_val_fixed = test_fun(*tuple(self.dataset_test.append_list(results_dic_fixed, self.model.list_compare_list )))
                    print test_display, psnr_val_fixed
                    # record
                    f_log_val_fixed_PSNR.write(str(it)+', '+str(psnr_val_fixed)+', '+str(lr_scale * base_lr)+'\n')
                    f_log_val_fixed_PSNR.flush()

                ####### display
                if (it % show_step == 0):
                    print 'save image to', save_folder
                    self.show_result(results_dic, save_folder, it, show_mode = show_mode, folder='')
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

    def show_result(self, data_buffer, save_folder, iteration, show_mode, folder = ''):
        for key, value in data_buffer.iteritems():
            print key

        if show_mode == 'psnr':
            f = data_buffer['flow']

            plt.subplot(321)
            my_imshow(data_buffer['input_HR_cent'][np.newaxis,0,:,:,:])
            plt.subplot(322)
            my_imshow(data_buffer['input_REF_cent'][np.newaxis,0,:,:,:])
            plt.subplot(323)
            my_imshow(data_buffer['HR_output'][np.newaxis,0,:,:,:])
            plt.subplot(324)
            my_imshow(5 *np.abs(data_buffer['HR_output']-data_buffer['input_HR_cent']) [np.newaxis,0,:,:,:])

            f_visual_limit = 5.0
            plt.subplot(325)
            plt.imshow(f[0,0,:,:], vmin=-f_visual_limit, vmax=f_visual_limit, interpolation="nearest")
            plt.subplot(326)
            plt.imshow(f[0,1,:,:], vmin=-f_visual_limit, vmax=f_visual_limit, interpolation="nearest")

            plt.show(False)
            plt.draw()
            plt.pause(0.01)

            scipy.misc.toimage(np.squeeze(np.transpose(data_buffer['input_HR_cent'][np.newaxis,0,:,:,:],axes=(0,2,3,1))), cmin=0.0, cmax=1.0).save(save_folder+'/'+folder+'img1_'+str(iteration)+'.png')
            scipy.misc.toimage(np.squeeze(np.transpose(data_buffer['input_REF_cent'][np.newaxis,0,:,:,:],axes=(0,2,3,1))), cmin=0.0, cmax=1.0).save(save_folder+'/'+folder+'img2_'+str(iteration)+'.png')
            scipy.misc.toimage(np.squeeze(np.transpose(data_buffer['HR_output'][np.newaxis,0,:,:,:],axes=(0,2,3,1))), cmin=0.0, cmax=1.0).save(save_folder+'/'+folder+'warp_'+str(iteration)+'.png')
            scipy.misc.toimage(np.squeeze(np.transpose(np.abs(data_buffer['HR_output']-data_buffer['input_HR_cent'])[np.newaxis,0,:,:,:],axes=(0,2,3,1))), cmin=0.0, cmax=1.0).save(save_folder+'/'+folder+'err_'+str(iteration)+'.png')

            scipy.misc.toimage(data_buffer['feat_LR'][0,0,:,:], cmin=np.min(data_buffer['feat_LR'][0,0,:,:]), cmax=np.max(data_buffer['feat_LR'][0,0,:,:])).save(save_folder+'/'+folder+'feat_LR'+str(iteration)+'.png')
            scipy.misc.toimage(data_buffer['feat_REF'][0,0,:,:], cmin=np.min(data_buffer['feat_REF'][0,0,:,:]), cmax=np.max(data_buffer['feat_REF'][0,0,:,:])).save(save_folder+'/'+folder+'feat_REF'+str(iteration)+'.png')
            scipy.misc.toimage(data_buffer['corr'][0,0,:,:], cmin=np.min(data_buffer['corr'][0,0,:,:]), cmax=np.max(data_buffer['corr'][0,0,:,:])).save(save_folder+'/'+folder+'corr'+str(iteration)+'.png')

            if 'occlusion' in data_buffer.keys():
                scipy.misc.toimage(data_buffer['occlusion'][0,0,:,:], cmin=0, cmax=1).save(save_folder+'/'+folder+'occlusion'+str(iteration)+'.png')
            if 'input_SRResNet_NN_LR_cent' in data_buffer.keys():
                scipy.misc.toimage(data_buffer['input_SRResNet_NN_LR_cent'][0,:,:,:], cmin=0, cmax=1).save(save_folder+'/'+folder+'SRResNet_NN_LR'+str(iteration)+'.png')

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
        self.savepath = savepath
        load_snapshot(self.model.net['params_all'], self.model.T_updates, preload_model)

    def __create_dict(self, list1, list2):
        results_dic = {}
        for i in range(len(list1)):
            results_dic[list1[i]] = list2[i]
        return results_dic

    def save_result(self):
        # save_folder = '/fileserver/haitian/haitian_backup/HT_sr/SRResNet_After_BMVC/result_LF-(-3,-3)/ene2end_SR_warp'
        # save_folder = '/fileserver/haitian/haitian_backup/HT_sr/SRResNet_After_BMVC/result_LF-(-3,-3)/FlowNetS_backward'
        # save_folder = '/fileserver/haitian/haitian_backup/HT_sr/SRResNet_After_BMVC/result_LF-(-3,-3)/FlowNetS_forward_0.45'
        save_folder = self.self.savepath

        for i in range(268+1):
            print i
            data_buffer_test_fixed = self.dataset_test.nextBatch(batchsize=1, shuffle=False, view_mode = 'Fixed-inv', residue = False, augmentation = False, index_inc = True)
            test_input_list_fixed = self.dataset_test.append_list(data_buffer_test_fixed, self.model.list_test_input)
            results_fixed = self.model.fun_test(*tuple(test_input_list_fixed))
            # create dictionary
            results_dic_fixed = self.__create_dict(self.model.list_test_output, results_fixed)

            # scipy.misc.toimage(np.squeeze(np.transpose(results_dic_fixed['input_HR_cent'][np.newaxis,0,:,:,:],axes=(0,2,3,1))), cmin=0.0, cmax=1.0).save(save_folder+'/'+'img1_'+'.png')
            # scipy.misc.toimage(np.squeeze(np.transpose(results_dic_fixed['input_REF_cent'][np.newaxis,0,:,:,:],axes=(0,2,3,1))), cmin=0.0, cmax=1.0).save(save_folder+'/'+'img2_'+'.png')
            scipy.misc.toimage(np.squeeze(np.transpose(results_dic_fixed['HR_output'][np.newaxis,0,:,:,:],axes=(0,2,3,1))), cmin=0.0, cmax=1.0).save(save_folder+'/'+str(i)+'.png')

    def inference(self, input_list):
        output_list = self.model.fun_test(input_list)
        return output_list

    