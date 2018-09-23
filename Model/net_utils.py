# encoding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


def init_weights(w, init_type):

    if init_type == 'w_init_relu':
        nn.init.kaiming_uniform_(w, nonlinearity = 'relu')
    elif init_type == 'w_init_leaky':
        nn.init.kaiming_uniform_(w, nonlinearity = 'leaky_relu')
    elif init_type == 'w_init':
        nn.init.uniform_(w)

def activation(activation):

    if activation == 'relu':
        return nn.ReLU(inplace = True)
    elif activation == 'leaky_relu':
        return nn.LeakyReLU(negative_slope = 0.1 ,inplace = True )
    elif activation == 'selu':
        return nn.SELU(inplace = True)
    elif activation == 'linear':
        return nn.Linear()

# --------------------FlowNet parts--------------------------------------

# class conv_activation(nn.Module):

#     def __init__(self, in_ch, out_ch, kernel_size = 0 , stride = 0, padding = 0, activation = 'relu', init_type = 'w_init_relu'):
#         super(conv_activation, self).__init__()

#         self.conv = nn.Conv2d(in_ch, out_ch, kernel_size,stride, padding)
#         init_weights(self.conv, init_type = init_type)
#         self.activation = activation(activation = activation)

#     def forward(self, x):
#         x1 = self.conv(x)
#         x2 = self.activation(x1)

#         return x2


# class flow(nn.Module):

#     def __init__(self,in_ch, out_ch = 2, kernel_size = 3 , stride = 1, padding = 1, activation = 'linear', init_type = 'w_init' ):
#         super(flow, self).__init__()

#         self.conv = nn.Conv2d(in_ch, out_ch, kernel_size,stride, padding)
#         init_weights(self.conv, init_type = init_type)
#         self.activation = activation(activation = activation)

#     def forward(self,x):
#         x1 = self.conv(x)
#         x2 = self.activation(x1)

#         return x2

# class leaky_deconv(nn.Module):

#     def __init__(self,in_ch, out_ch, deconv = 'default',activation = 'leaky_relu', init_type = 'w_init_leaky' ):
#         super(leaky_deconv, self).__init__()

#         if deconv == 'default':
#             self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
#             init_weights(self.up, init_type = init_type)
#             self.conv = conv_activation(in_ch, out_ch, kernel_size = 1, stride = 1 ,padding = 0, activation = activation,init_type = init_type)
#         else:
#             #TODO
#             print 'deconv type errors'
#             sys.exit(0)

#     def forward(self,x):
#         x1 = self.up(x)
#         x2 = self.conv(x1)

#         return x2

# class upsample(nn.Module):

#     def __init__(self,in_ch, out_ch, deconv = 'default',activation = 'linear', init_type = 'w_init' ):
#         super(upsample, self).__init__()

#         if deconv == 'default':
#             self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
#             init_weights(self.up, init_type = init_type)
#             self.conv = conv_activation(in_ch, out_ch, kernel_size = 1, stride = 1 ,padding = 0, activation = activation,init_type = init_type)
#         else:
#             #TODO
#             print 'deconv type errors'
#             sys.exit(0)

#     def forward(self,x):
#         x1 = self.up(x)
#         x2 = self.conv(x1)

#         return x2


# ---------------------------------fuction------------------------------------
def conv_activation(in_ch, out_ch , kernel_size = 3, stride = 1, padding = 1, activation = 'relu', init_type = 'w_init_relu'):


    if activation == 'relu':
        return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding),
                nn.ReLU(inplace = True))

    elif activation == 'leaky_relu':
        return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding),
                nn.LeakyReLU(negative_slope = 0.1 ,inplace = True ))

    elif activation == 'selu':
        return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding),
                nn.SELU(inplace = True))

    elif activation == 'linear':
        return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding))


def flow(in_ch, out_ch , kernel_size = 3, stride = 1, padding = 1, activation = 'linear' , init_type = 'w_init'):

    if activation == 'relu':
        return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding),
                nn.ReLU(inplace = True))

    elif activation == 'leaky_relu':
        return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding),
                nn.LeakyReLU(negative_slope = 0.1 ,inplace = True ))

    elif activation == 'selu':
        return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding),
                nn.SELU(inplace = True))

    elif activation == 'linear':
        return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding))


def upsample(in_ch, out_ch):

    return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True)



def leaky_deconv(in_ch, out_ch):

    return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True),
                nn.LeakyReLU(0.1,inplace=True)
                        )

def deconv_activation(in_ch, out_ch ,activation = 'relu' ):

    if activation == 'relu':
        return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True),
                nn.ReLU(inplace = True))

    elif activation == 'leaky_relu':
        return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True),
                nn.LeakyReLU(negative_slope = 0.1 ,inplace = True ))

    elif activation == 'selu':
        return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True),
                nn.SELU(inplace = True))

    elif activation == 'linear':
        return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True))

#-----------------------------UNet----------------------------------

class Encoder(nn.Module):

    def __init__(self,in_ch,activation = 'selu', init_type = 'w_init'):
        super(Encoder, self).__init__()

        self.layer_f = conv_activation(in_ch, 64 , kernel_size = 5 ,stride = 1,padding = 2, activation = activation, init_type = init_type)

        self.conv1 = conv_activation(64, 64 , kernel_size = 5 ,stride = 1,padding = 2, activation = activation, init_type = init_type)

        self.conv2 = conv_activation(64, 64 , kernel_size = 5 ,stride = 2,padding = 2, activation = activation, init_type = init_type)

        self.conv3 = conv_activation(64, 64 , kernel_size = 5 ,stride = 2,padding = 2, activation = activation, init_type = init_type)

        self.conv4 = conv_activation(64, 64 , kernel_size = 5 ,stride = 2,padding = 2, activation = activation, init_type = init_type)


    def forward(self,x):

        layer_f = self.layer_f(x)
        conv1 = self.conv1(layer_f)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        return conv1,conv2,conv3,conv4


class UNet_decoder_2(nn.Module):

    def __init__(self, activation = 'selu' , init_type = 'w_init'):
        super(UNet_decoder_2, self).__init__()

        self.warp_deconv4 = deconv_activation(128, 64,activation = activation)
        # in_ch = 64 + 64 +64
        self.warp_deconv3 = deconv_activation(192 , 64,activation = activation)
        #in_ch
        self.warp_deconv2 = deconv_activation(192, 64,activation = activation)

        self.post_fusion1 = conv_activation(192, 64, kernel_size = 5, stride = 1, padding = 2,activation = activation,init_type = init_type)
        self.post_fusion2 = conv_activation(64, 64, kernel_size = 5, stride = 1, padding = 2, activation = activation,init_type = init_type)

        self.final = conv_activation(64, 3, kernel_size = 5,stride = 1, padding = 2,activation = 'linear', init_type = init_type)




    def forward(self,LR_conv1, LR_conv2, LR_conv3, LR_conv4, warp_conv1, warp_conv2, warp_conv3, warp_conv4):

        concat0 = torch.cat((LR_conv4,warp_conv4),1)
        warp_deconv4 = self.warp_deconv4(concat0)

        concat1 = torch.cat((warp_deconv4,LR_conv3,warp_conv3),1)
        warp_deconv3 = self.warp_deconv3(concat1)

        concat2 = torch.cat((warp_deconv3,LR_conv2,warp_conv2),1)
        warp_deconv2 = self.warp_deconv2(concat2)

        concat3 = torch.cat((warp_deconv2,LR_conv1,warp_conv1),1)
        post_fusion1 = self.post_fusion1(concat3)

        post_fusion2 = self.post_fusion2(post_fusion1)
        final = self.final(post_fusion1)

        return final

