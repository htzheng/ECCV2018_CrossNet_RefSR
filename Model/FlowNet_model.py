
from net_utils import *
import torch


class FlowNet(nn.Module):

    def __init__(self, in_ch ):
        super(FlowNet, self).__init__()

        activation = 'leaky_relu'
        init_type = 'w_init_leaky'
        reduced_model = False

        if reduced_model == False:
        # flow unit
            self.conv1 = conv_activation(in_ch, 64, kernel_size = 7, stride = 2, padding = 3, activation = activation,init_type = init_type )
            self.conv2 = conv_activation(64, 128, kernel_size = 5, stride = 2, padding = 2, activation = activation,init_type = init_type )

            self.conv3 = conv_activation(128, 256, kernel_size = 5, stride = 2, padding = 2, activation = activation,init_type = init_type )
            self.conv3_1 = conv_activation(256, 256, kernel_size = 3, stride = 1, padding = 1, activation = activation,init_type = init_type )

            self.conv4 = conv_activation(256, 512, kernel_size = 3, stride = 2, padding = 1, activation = activation,init_type = init_type )
            self.conv4_1 = conv_activation(512, 512, kernel_size = 3, stride = 1, padding = 1, activation = activation,init_type = init_type )

            self.conv5 = conv_activation(512, 512, kernel_size = 3, stride = 2, padding = 1, activation = activation,init_type = init_type )
            self.conv5_1 = conv_activation(512, 512, kernel_size = 3, stride = 1, padding = 1, activation = activation,init_type = init_type )

            self.conv6 = conv_activation(512, 1024, kernel_size = 3, stride = 2, padding = 1, activation = activation,init_type = init_type )
            self.conv6_1 = conv_activation(1024, 1024, kernel_size = 3, stride = 1, padding = 1, activation = activation,init_type = init_type )
                                

            # refine unit
            self.flow6 = flow(1024,2)
            self.flow6_up = upsample(2,2)
            self.deconv5 = leaky_deconv(1024,256)

            # in_ch = 512 + 256 + 2 = 770
            self.flow5 = flow(770, 2)
            self.flow5_up = upsample( 2, 2)
            self.deconv4 = leaky_deconv(770, 256)

            #in_ch = 512 + 256 + 2 = 770
            self.flow4 = flow(770,2)
            self.flow4_up = upsample(2, 2)
            self.deconv3 = leaky_deconv(770, 128)

            #in_ch = 256 + 128 + 2
            self.flow3 = flow(386,2)
            self.flow3_up = upsample(2, 2)
            self.deconv2 = leaky_deconv(386, 64)

            #in_ch = 64 + 128 + 2 = 194
            self.flow2 = flow(194,2)
            self.flow2_up = upsample(2, 2)
            self.deconv1 = leaky_deconv(194, 64)

            #in_ch = 64 + 64 + 2
            self.flow1 = flow(130,2)
            self.flow1_up = upsample(2, 2)
            self.deconv0 = leaky_deconv(130, 64)

            #in_ch = 3 + 64 + 2 = 69
            self.concat0_conv1 = conv_activation(69, 16,kernel_size = 7, stride = 1, padding = 3,activation = 'selu', init_type = 'w_init')
            self.concat0_conv2 = conv_activation(16, 16,kernel_size = 7, stride = 1, padding = 3,activation = 'selu', init_type = 'w_init')
            self.flow_12 = flow(16,2)

    def forward(self, input_img1_LR, input_img2_HR):


        input_ = torch.cat((input_img1_LR,input_img2_HR),1)
        conv1 = self.conv1(input_)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv3_1 = self.conv3_1(conv3)

        conv4 = self.conv4(conv3_1)
        conv4_1 = self.conv4_1(conv4)

        conv5 = self.conv5(conv4_1)
        conv5_1 = self.conv5_1(conv5)

        conv6 = self.conv6(conv5_1)
        conv6_1 = self.conv6_1(conv6)

        flow6 = self.flow6(conv6_1)
        flow6_up = self.flow6_up(flow6)
        deconv5 = self.deconv5(conv6_1)

        concat5 = torch.cat((conv5_1,deconv5,flow6_up),1)
        flow5 = self.flow5(concat5)
        flow5_up = self.flow5_up(flow5)
        deconv4 = self.deconv4(concat5)

        concat4 = torch.cat((conv4_1,deconv4,flow5_up),1)
        flow4 = self.flow4(concat4)
        flow4_up = self.flow4_up(flow4)
        deconv3 = self.deconv3(concat4)

        concat3 = torch.cat((conv3_1,deconv3,flow4_up),1)
        flow3 = self.flow3(concat3)
        flow3_up = self.flow3_up(flow3)
        deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((conv2,deconv2,flow3_up),1)
        flow2 = self.flow2(concat2)
        flow2_up = self.flow2_up(flow2)
        deconv1 = self.deconv1(concat2)

        concat1 = torch.cat((conv1,deconv1,flow2_up),1)
        flow1 = self.flow1(concat1)
        flow1_up = self.flow1_up(flow1)
        deconv0 = self.deconv0(concat1)

        concat0 = torch.cat((input_img2_HR,deconv0,flow1_up),1)
        concat0_conv1 = self.concat0_conv1(concat0)
        concat0_conv2 = self.concat0_conv2(concat0_conv1)
        flow_12 = self.flow_12(concat0_conv2)


        output_dic = {
                    'flow_12_1':flow_12,
                    'flow_12_2':flow1,
                    'flow_12_3':flow2,
                    'flow_12_4':flow3
                 }

        return output_dic



