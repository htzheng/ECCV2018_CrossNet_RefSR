from lasagne.nonlinearities import LeakyRectify, rectify, selu, sigmoid, linear
from lasagne.init import Constant, GlorotUniform, HeNormal, HeUniform
import lasagne
from lasagne.layers import InputLayer, MergeLayer, ElemwiseMergeLayer, ConcatLayer, SliceLayer, Conv2DLayer, Deconv2DLayer, NonlinearityLayer, ExpressionLayer, SubpixelReshuffleLayer, Elementwise_mul_mask

import theano.tensor as T

##################################   activation and initialization
relu = rectify
leaky_rectify = LeakyRectify(0.1)
SELU_activation = selu
linear = linear

W_init = HeUniform
W_init_leaky = HeUniform(gain=1.407)
W_init_relu = HeUniform(gain=1.414)
W_init_linear = HeUniform(gain=1.0)
W_init_SELU = HeNormal(gain=1.0)

def Charbonnier(x):
    return T.sqrt(T.sqr(x)+0.000001)

def encoder(layer, name = '', activation = SELU_activation, W_init = W_init_SELU):
    layer_f = conv_activation(layer, num_filters=64, filter_size=5, stride=1, activation = activation, init = W_init)
    # 320  scale0
    conv1 = conv_activation(layer_f, num_filters=64, filter_size=5, stride=1, activation = activation, init = W_init)
    # 160  scale1
    conv2 = conv_activation(conv1, num_filters=64, filter_size=5, stride=2, activation = activation, init = W_init)
    # 80  scale2
    conv3 = conv_activation(conv2, num_filters=64, filter_size=5, stride=2, activation = activation, init = W_init)
    # 40  scale3
    conv4 = conv_activation(conv3, num_filters=64, filter_size=5, stride=2, activation = activation, init = W_init)
    return conv1, conv2, conv3, conv4

def UNet_decoder(LR_conv4, warp_conv1, warp_conv2, warp_conv3, warp_conv4, activation = SELU_activation, W_init = W_init_SELU): 
    if not LR_conv4 is None:
        # 80
        warp_deconv4 = Deconv2DLayer(ConcatLayer([LR_conv4, warp_conv4]), num_filters=64, filter_size=4, stride=2, crop=1, W = W_init, b=Constant(0.), nonlinearity=activation)
    else:
        # 80
        warp_deconv4 = Deconv2DLayer(warp_conv4, num_filters=64, filter_size=4, stride=2, crop=1, W = W_init, b=Constant(0.), nonlinearity=activation)
        
    # 160
    warp_deconv3 = Deconv2DLayer(ConcatLayer([warp_deconv4, warp_conv3]), num_filters=64, filter_size=4, stride=2, crop=1, W = W_init, b=Constant(0.), nonlinearity=activation)
    # 320
    warp_deconv2 = Deconv2DLayer(ConcatLayer([warp_deconv3, warp_conv2]), num_filters=64, filter_size=4, stride=2, crop=1, W = W_init, b=Constant(0.), nonlinearity=activation)
    # final 
    post_fusion1 = Conv2DLayer(warp_deconv2, 64, 5, pad=2, W = W_init, b=Constant(0.), nonlinearity=activation)
    post_fusion2 = Conv2DLayer(post_fusion1, 64, 5, pad=2, W = W_init, b=Constant(0.), nonlinearity=activation)
    final = Conv2DLayer(post_fusion1, 3, 5, pad=2, W = W_init_linear, b=Constant(0.), nonlinearity=linear)
    return final

# def UNet_decoder(LR_conv4, warp_conv1, warp_conv2, warp_conv3, warp_conv4, activation = SELU_activation, W_init = W_init_SELU): 
#     # 80
#     warp_deconv4 = Deconv2DLayer(ConcatLayer([LR_conv4, warp_conv4]), num_filters=64, filter_size=4, stride=2, crop=1, W = W_init, b=Constant(0.), nonlinearity=activation)
#     # 160
#     warp_deconv3 = Deconv2DLayer(ConcatLayer([warp_deconv4, warp_conv3]), num_filters=64, filter_size=4, stride=2, crop=1, W = W_init, b=Constant(0.), nonlinearity=activation)
#     # 320
#     warp_deconv2 = Deconv2DLayer(ConcatLayer([warp_deconv3, warp_conv2]), num_filters=64, filter_size=4, stride=2, crop=1, W = W_init, b=Constant(0.), nonlinearity=activation)
#     # final 
#     post_fusion1 = Conv2DLayer(warp_deconv2, 64, 5, pad=2, W = W_init, b=Constant(0.), nonlinearity=activation)
#     post_fusion2 = Conv2DLayer(post_fusion1, 64, 5, pad=2, W = W_init, b=Constant(0.), nonlinearity=activation)
#     final = Conv2DLayer(post_fusion1, 3, 5, pad=2, W = W_init_linear, b=Constant(0.), nonlinearity=linear)
#     return final

# def UNet_decoder(LR_conv4, warp_conv1, warp_conv2, warp_conv3, warp_conv4, activation = SELU_activation, W_init = W_init_SELU): 
#     # 80
#     warp_deconv4 = Deconv2DLayer(ConcatLayer([LR_conv4, warp_conv4]), num_filters=64, filter_size=4, stride=2, crop=1, W = W_init, b=Constant(0.), nonlinearity=activation)
#     # 160
#     warp_deconv3 = Deconv2DLayer(ConcatLayer([warp_deconv4, warp_conv3]), num_filters=64, filter_size=4, stride=2, crop=1, W = W_init, b=Constant(0.), nonlinearity=activation)
#     # 320
#     warp_deconv2 = Deconv2DLayer(ConcatLayer([warp_deconv3, warp_conv2]), num_filters=64, filter_size=4, stride=2, crop=1, W = W_init, b=Constant(0.), nonlinearity=activation)
#     # final 
#     post_fusion1 = Conv2DLayer(warp_deconv2, 64, 5, pad=2, W = W_init, b=Constant(0.), nonlinearity=activation)
#     post_fusion2 = Conv2DLayer(post_fusion1, 64, 5, pad=2, W = W_init, b=Constant(0.), nonlinearity=activation)
#     final = Conv2DLayer(post_fusion1, 3, 5, pad=2, W = W_init_linear, b=Constant(0.), nonlinearity=linear)
#     return final

def UNet_decoder_2(LR_conv1, LR_conv2, LR_conv3, LR_conv4, warp_conv1, warp_conv2, warp_conv3, warp_conv4, activation = SELU_activation, W_init = W_init_SELU): 
    # 80
    warp_deconv4 = Deconv2DLayer(ConcatLayer([LR_conv4, warp_conv4]), num_filters=64, filter_size=4, stride=2, crop=1, W = W_init, b=Constant(0.), nonlinearity=activation)
    # 160
    warp_deconv3 = Deconv2DLayer(ConcatLayer([warp_deconv4, LR_conv3, warp_conv3]), num_filters=64, filter_size=4, stride=2, crop=1, W = W_init, b=Constant(0.), nonlinearity=activation)
    # 320
    warp_deconv2 = Deconv2DLayer(ConcatLayer([warp_deconv3, LR_conv2, warp_conv2]), num_filters=64, filter_size=4, stride=2, crop=1, W = W_init, b=Constant(0.), nonlinearity=activation)
    # final 
    post_fusion1 =   Conv2DLayer(ConcatLayer([warp_deconv2, LR_conv1, warp_conv1]), 64, 5, pad=2, W = W_init, b=Constant(0.), nonlinearity=activation)
    post_fusion2 =   Conv2DLayer(post_fusion1, 64, 5, pad=2, W = W_init, b=Constant(0.), nonlinearity=activation)
    final = Conv2DLayer(post_fusion1, 3, 5, pad=2, W = W_init_linear, b=Constant(0.), nonlinearity=linear)
    return final

def UNet_decoder_3(LR_conv1, LR_conv2, LR_conv3, LR_conv4, warp_conv1, warp_conv2, warp_conv3, warp_conv4): 
    # 80
    mask4 = Conv2DLayer(ConcatLayer([LR_conv4, warp_conv4]), 64, 5, pad=2, W = W_init_SELU, b=Constant(0.), nonlinearity=sigmoid)
    warp_conv4_m = ElemwiseMergeLayer([warp_conv4, mask4], T.mul)
    warp_deconv4 = Deconv2DLayer(ConcatLayer([LR_conv4, warp_conv4_m]), num_filters=64, filter_size=4, stride=2, crop=1, W = W_init_SELU, b=Constant(0.), nonlinearity=SELU_activation)

    # 160
    mask3 = Conv2DLayer(ConcatLayer([warp_deconv4, LR_conv3, warp_conv3]), 64, 5, pad=2, W = W_init_SELU, b=Constant(0.), nonlinearity=sigmoid)
    warp_conv3_m = ElemwiseMergeLayer([warp_conv3, mask3], T.mul)
    warp_deconv3 = Deconv2DLayer(ConcatLayer([warp_deconv4, LR_conv3, warp_conv3_m]), num_filters=64, filter_size=4, stride=2, crop=1, W = W_init_SELU, b=Constant(0.), nonlinearity=SELU_activation)
    # 320
    mask2 = Conv2DLayer(ConcatLayer([warp_deconv3, LR_conv2, warp_conv2]), 64, 5, pad=2, W = W_init_SELU, b=Constant(0.), nonlinearity=sigmoid)
    warp_conv2_m = ElemwiseMergeLayer([warp_conv2, mask2], T.mul)
    warp_deconv2 = Deconv2DLayer(ConcatLayer([warp_deconv3, LR_conv2, warp_conv2_m]), num_filters=64, filter_size=4, stride=2, crop=1, W = W_init_SELU, b=Constant(0.), nonlinearity=SELU_activation)
    # final 
    mask1 = Conv2DLayer(ConcatLayer([warp_deconv2, LR_conv1, warp_conv1]), 64, 5, pad=2, W = W_init_SELU, b=Constant(0.), nonlinearity=sigmoid)
    warp_conv1_m = ElemwiseMergeLayer([warp_conv1, mask1], T.mul)
    post_fusion1 =  Conv2DLayer(ConcatLayer([warp_deconv2, LR_conv1, warp_conv1_m]), 64, 5, pad=2, W = W_init_SELU, b=Constant(0.), nonlinearity=SELU_activation)

    post_fusion2 =  Conv2DLayer(post_fusion1, 64, 5, pad=2, W = W_init_SELU, b=Constant(0.), nonlinearity=SELU_activation)
    final = Conv2DLayer(post_fusion1, 3, 5, pad=2, W = W_init_linear, b=Constant(0.), nonlinearity=linear)

    test = Conv2DLayer(final, 3, 5, pad=2, W = W_init_linear, b=Constant(0.), nonlinearity=linear)
    return final

def UNet_decoder_4(LR_conv1, LR_conv2, LR_conv3, LR_conv4, warp_conv1, warp_conv2, warp_conv3, warp_conv4): 
    # 80
    mask4 = Conv2DLayer(ConcatLayer([LR_conv4, warp_conv4]), 1, 5, pad=2, W = W_init_SELU, b=Constant(0.), nonlinearity=sigmoid)
    warp_conv4_m = Elementwise_mul_mask(warp_conv4, mask4)
    warp_deconv4 = Deconv2DLayer(ConcatLayer([LR_conv4, warp_conv4_m]), num_filters=64, filter_size=4, stride=2, crop=1, W = W_init_SELU, b=Constant(0.), nonlinearity=SELU_activation)

    # 160
    mask3 = Conv2DLayer(ConcatLayer([warp_deconv4, LR_conv3, warp_conv3]), 1, 5, pad=2, W = W_init_SELU, b=Constant(0.), nonlinearity=sigmoid)
    warp_conv3_m = Elementwise_mul_mask(warp_conv3, mask3)
    warp_deconv3 = Deconv2DLayer(ConcatLayer([warp_deconv4, LR_conv3, warp_conv3_m]), num_filters=64, filter_size=4, stride=2, crop=1, W = W_init_SELU, b=Constant(0.), nonlinearity=SELU_activation)
    # 320
    mask2 = Conv2DLayer(ConcatLayer([warp_deconv3, LR_conv2, warp_conv2]), 1, 5, pad=2, W = W_init_SELU, b=Constant(0.), nonlinearity=sigmoid)
    warp_conv2_m = Elementwise_mul_mask(warp_conv2, mask2)
    warp_deconv2 = Deconv2DLayer(ConcatLayer([warp_deconv3, LR_conv2, warp_conv2_m]), num_filters=64, filter_size=4, stride=2, crop=1, W = W_init_SELU, b=Constant(0.), nonlinearity=SELU_activation)
    # final 
    mask1 = Conv2DLayer(ConcatLayer([warp_deconv2, LR_conv1, warp_conv1]), 1, 5, pad=2, W = W_init_SELU, b=Constant(0.), nonlinearity=sigmoid)
    warp_conv1_m = Elementwise_mul_mask(warp_conv1, mask1)

    post_fusion1 =  Conv2DLayer(ConcatLayer([warp_deconv2, LR_conv1, warp_conv1_m]), 64, 5, pad=2, W = W_init_SELU, b=Constant(0.), nonlinearity=SELU_activation)
    post_fusion2 =  Conv2DLayer(post_fusion1, 64, 5, pad=2, W = W_init_SELU, b=Constant(0.), nonlinearity=SELU_activation)
    final = Conv2DLayer(post_fusion1, 3, 5, pad=2, W = W_init_linear, b=Constant(0.), nonlinearity=linear)

    test = Conv2DLayer(final, 3, 5, pad=2, W = W_init_linear, b=Constant(0.), nonlinearity=linear)
    return final

##################################   correlation layer
import_corr_layer = False
if import_corr_layer:
    from correlation_layer import CorrelationOp
    class CorrelationLayer(MergeLayer):
        def __init__(self, first_layer, second_layer,
                    pad_size=20, kernel_size=1, stride1=1, stride2=2,
                    max_displacement=20, **kwargs):
            super(CorrelationLayer, self).__init__(
                [first_layer, second_layer], **kwargs)
            self.pad_size = pad_size
            self.kernel_size = kernel_size
            self.stride1 = stride1
            self.stride2 = stride2
            self.max_displacement = max_displacement
            self.bottom_shape = lasagne.layers.get_output_shape(first_layer)

        def get_output_shape_for(self, input_shapes):
            # This fake op is just for inferring shape
            op = CorrelationOp(
                self.bottom_shape,
                pad_size=self.pad_size,
                kernel_size=self.kernel_size,
                stride1=self.stride1,
                stride2=self.stride2,
                max_displacement=self.max_displacement)
            return (input_shapes[0][0], op.top_channels, op.top_height, op.top_width)

        def get_output_for(self, inputs, **kwargs):
            op = CorrelationOp(
                self.bottom_shape,
                pad_size=self.pad_size,
                kernel_size=self.kernel_size,
                stride1=self.stride1,
                stride2=self.stride2,
                max_displacement=self.max_displacement)
            return op(*inputs)[2]


################################   FlowNet layers
def conv_activation(input_layer,num_filters = 1, filter_size=7, name = '', pad='same', activation = None, init = W_init_linear, **kwargs):
    return Conv2DLayer(input_layer, num_filters, filter_size,  name = name, pad=pad,
                       flip_filters=False, W = init, b = Constant(0.), nonlinearity = activation, **kwargs) 

def flow(input_layer, filter_size=3, pad=1,**kwargs):
    return Conv2DLayer(
        input_layer, num_filters=2, filter_size=filter_size, stride=1,
        nonlinearity=linear, pad=pad, W = W_init(gain=1.0), b = Constant(0.), flip_filters=False, **kwargs) 

def leaky_deconv(input_layer, num_filters=64, activation = None, init = W_init_linear, deconv = 'default',  **kwargs):
    stride = 2
    if deconv == 'default':
        return Deconv2DLayer(
            input_layer, num_filters = num_filters, nonlinearity = activation,
            filter_size=4, stride=2, crop=1, W = init,  b=Constant(0.), flip_filters=False, **kwargs)   # flip_filters=True(original) False(mine)
    elif deconv == 'subpixel':
        deconv_layer = Conv2DLayer(input_layer,
                                num_filters=num_filters*stride*stride, filter_size=3, 
                                pad = 1, nonlinearity = activation,
                                W = init, b=Constant(0.), name = 'subpixel_conv')
        return SubpixelReshuffleLayer(deconv_layer, num_filters, stride, name = 'subpixel_shuffle')

def upsample(input_layer, deconv = 'default', **kwargs):
    stride = 2
    if deconv == 'default':
        return Deconv2DLayer(
            input_layer, num_filters=2, filter_size=4, stride=2,
            crop=1, W = W_init(gain=1.0), b=Constant(0.), nonlinearity=linear, flip_filters=False, **kwargs) # flip_filters=True(original) False(mine)
    elif deconv == 'subpixel':
        deconv_layer = Conv2DLayer(input_layer,
                                num_filters=2*stride*stride, filter_size=3, 
                                pad = 1, nonlinearity = linear,
                                W = W_init(gain=1.0), b=Constant(0.), name = 'flow_subpixel_conv')
        return SubpixelReshuffleLayer(deconv_layer, 2, stride, name = 'flow_subpixel_shuffle')

#################################   FlowNet subpixel
# def leaky_deconv(input_layer, activation = None, init = W_init_linear,  **kwargs):
#     deconv = lasagne.layers.Conv2DLayer(input_layer,
#                             num_filters=num_filters*stride*stride, filter_size=3, 
#                             pad = (filter_size-1)/2, nonlinearity = linear,
#                             W = init ,b=Constant(0.) )
#     deconv = lasagne.layers.SubpixelReshuffleLayer(deconv,num_filters,stride,name = name+'_linear_shuffle')

#     return Deconv2DLayer(
#         input_layer, nonlinearity=activation,
#         filter_size=4, stride=2, crop=1, W = init,  b=Constant(0.), flip_filters=False, **kwargs)   # flip_filters=True(original) False(mine)

# def upsample(input_layer, **kwargs):
#     return Deconv2DLayer(
#         input_layer, num_filters=2, filter_size=4, stride=2,
#         crop=1, W = W_init(gain=1.0), b=Constant(0.), nonlinearity=linear, flip_filters=False, **kwargs) # flip_filters=True(original) False(mine)


###############################    My layers
def conv_activation_bn(input_layer, name = '', pad='same', activation = 'relu', W_init = -1, use_bn = True, **kwargs):
    if use_bn:
        conv = Conv2DLayer(input_layer, name = name+'_linear', nonlinearity=linear, pad=pad,
                        flip_filters=False, W = W_init, b = Constant(0.), **kwargs) 
        bn = BatchNormLayer(conv, name = name+'_bn')
        out = NonlinearityLayer(bn, name = name+'_activation', nonlinearity = activation)
    else:
        out = Conv2DLayer(input_layer, name = name, nonlinearity=activation, pad=pad,
                        flip_filters=False, W = W_init, b = Constant(0.), **kwargs) 
    return out

def upsample_bn(input_layer, name = '', num_filters = None, filter_size= None, stride=None, crop=None, 
                activation = 'relu', use_bn = True, W_init = 1, deconv_mode = None, **kwargs):
    if (deconv_mode == ''):
        deconv = Deconv2DLayer(input_layer, name = name+'_linear',nonlinearity=linear, num_filters=num_filters, filter_size=filter_size, stride=stride,
            crop=crop, W = W_init, b=Constant(0.), flip_filters=False, **kwargs)   
    elif (deconv_mode == 'Subpixel'):
        deconv = lasagne.layers.Conv2DLayer(input_layer,name=name+'_linear',
                                num_filters=num_filters*stride*stride, filter_size=3, 
                                pad = (filter_size-1)/2, nonlinearity = linear,
                                W = W_init,b=Constant(0.))
        deconv = lasagne.layers.SubpixelReshuffleLayer(deconv,num_filters,stride,name = name+'_linear_shuffle')

    if use_bn:
        bn = BatchNormLayer(deconv, name = name+'_bn')
        out = NonlinearityLayer(bn, name = name+'_activation', nonlinearity = activation)
    else:
        out = NonlinearityLayer(deconv, name = name+'_activation', nonlinearity = activation)
    return out

###############################    Not used
def residue_block(network_in, block_name, _num_filters = 64, _filter_size = (3, 3), _pad = 1):
    relu_W_init = lasagne.init.GlorotNormal()
    linear_W_init = lasagne.init.GlorotNormal()

    network_conv1 = lasagne.layers.Conv2DLayer(network_in,name=block_name+'_conv1',
                            num_filters=_num_filters, filter_size=_filter_size,
                            pad = _pad, nonlinearity = lasagne.nonlinearities.rectify,
                            W=relu_W_init,b=None)
    network_batchnorm1 = lasagne.layers.batch_norm(network_conv1,name=block_name+'_batchnorm1', axes='auto')
    network_conv2 = lasagne.layers.Conv2DLayer(network_batchnorm1,name=block_name+'_conv2',
                            num_filters=_num_filters, filter_size=_filter_size,
                            pad = _pad, nonlinearity = lasagne.nonlinearities.linear,
                            W=linear_W_init,b=None)
    network_batchnorm2 = lasagne.layers.batch_norm(network_conv2,name=block_name+'_batchnorm2', axes='auto')

    def element_wise_add(x1,x2):
        return x1+x2
    network_residue_add = lasagne.layers.ElemwiseMergeLayer([network_in, network_batchnorm2], element_wise_add, name=block_name+'_add')
    return network_residue_add

########################### Densenet related functions
def dense_block(network, num_layers, growth_rate, dropout, name_prefix):
    # concatenated 3x3 convolutions
    for n in range(num_layers):
        conv = bn_relu_conv(network, channels=growth_rate,
                            filter_size=5, dropout=dropout,
                            name_prefix=name_prefix + '_l%02d' % (n + 1))
        network = ConcatLayer([network, conv], axis=1,
                              name=name_prefix + '_l%02d_join' % (n + 1))
    return network

def dense_block_selu(network, num_layers, growth_rate, dropout, name_prefix, activation = 'relu', use_bn = True,  W_init = 0):
    # concatenated 3x3 convolutions
    for n in range(num_layers):
        conv = bn_relu_conv(network, channels=growth_rate,
                            filter_size=5, dropout=dropout,
                            name_prefix=name_prefix + '_l%02d' % (n + 1), activation = activation, use_bn = use_bn, W_init = W_init)
        network = ConcatLayer([network, conv], axis=1,
                              name=name_prefix + '_l%02d_join' % (n + 1))
    return network

def transition(network, dropout, name_prefix):
    # a transition 1x1 convolution followed by avg-pooling
    network = bn_relu_conv(network, channels=network.output_shape[1],
                           filter_size=1, dropout=dropout,
                           name_prefix=name_prefix)
    network = Pool2DLayer(network, 2, mode='average_inc_pad',
                          name=name_prefix + '_pool')
    return network

def bn_relu_conv(network, channels, filter_size, dropout, name_prefix, activation = rectify, use_bn = True,  W_init = W_init_relu):
    if use_bn:
        network = BatchNormLayer(network, name=name_prefix + '_bn')
    network = NonlinearityLayer(network, nonlinearity = activation,
                                name=name_prefix + '_relu')
    network = Conv2DLayer(network, channels, filter_size, pad='same',
                          W = W_init, b = None, nonlinearity = None,
                          name=name_prefix + '_conv')
    if dropout:
        network = DropoutLayer(network, dropout)
    return network

class DenseNetInit(lasagne.init.Initializer):
    """
    Reproduces the initialization scheme of the authors' Torch implementation.
    At least for the 40-layer networks, lasagne.init.HeNormal works just as
    fine, though. Kept here just in case. If you want to swap in this scheme,
    replace all W= arguments in all the code above with W=DenseNetInit().
    """
    def sample(self, shape):
        import numpy as np
        rng = lasagne.random.get_rng()
        if len(shape) >= 4:
            # convolutions use Gaussians with stddev of sqrt(2/fan_out), see
            # https://github.com/liuzhuang13/DenseNet/blob/cbb6bff/densenet.lua#L85-L86
            # and https://github.com/facebook/fb.resnet.torch/issues/106
            fan_out = shape[0] * np.prod(shape[2:])
            W = rng.normal(0, np.sqrt(2. / fan_out),
                           size=shape)
        elif len(shape) == 2:
            # the dense layer uses Uniform of range sqrt(1/fan_in), see
            # https://github.com/torch/nn/blob/651103f/Linear.lua#L21-L43
            fan_in = shape[0]
            W = rng.uniform(-np.sqrt(1. / fan_in), np.sqrt(1. / fan_in),
                            size=shape)
        return lasagne.utils.floatX(W)


######################## 