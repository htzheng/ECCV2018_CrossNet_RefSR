from LFDataset import LFDataset

# dataset_train = LFDataset(filename = '../../HT_sr/flowdata/LF_video_Dataset/train_bicubic.h5')
dataset_test = LFDataset(filename = '../../HT_sr/flowdata/LF_video_Dataset/test_bicubic.h5')
dataset_test.nextBatch_new(batchsize=8,shuffle=True,view_mode = 'Random', augmentation = True, crop_shape = (320,512))