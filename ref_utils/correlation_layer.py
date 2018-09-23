import numpy as np
import theano
import theano.tensor as T

from theano import Apply
from theano.gof import COp
from theano.gradient import grad_undefined
#from theano.sandbox.cuda import  as_cuda_ndarray_variable, GpuOp
from theano.sandbox.cuda import GpuOp
class CorrelationBaseOp(GpuOp, COp):
  __props__ = ('top_width', 'top_height', 'top_channels', 'pad_size',
               'kernel_radius', 'kernel_size', 'stride1', 'stride2',
               'max_displacement', 'neighborhood_grid_radius',
               'neighborhood_grid_width')

  func_file = "./correlation_layer.cu"

  def __init__(self, func_name, bottom_shape, **kwargs):
    super(CorrelationBaseOp, self).__init__(self.func_file, func_name)

    # Default parameters taken from the FlowNetC model
    self.pad_size = kwargs.get('pad_size', 20)
    self.kernel_size = kwargs.get('kernel_size', 1)
    self.stride1 = kwargs.get('stride1', 1)
    self.stride2 = kwargs.get('stride2', 2)
    self.max_displacement = kwargs.get('max_displacement', 20)

    self.bottom_shape = bottom_shape

    self.kernel_radius = (self.kernel_size - 1) // 2
    border_size = self.max_displacement + self.kernel_radius

    paddedbottomheight = self.bottom_shape[2] + 2 * self.pad_size
    paddedbottomwidth = self.bottom_shape[3] + 2 * self.pad_size

    self.top_width = int(np.ceil(float(paddedbottomwidth - border_size * 2) / float(self.stride1)))
    self.top_height = int(np.ceil(float(paddedbottomheight - border_size * 2) / float(self.stride1)))

    assert self.top_width >= 1
    assert self.top_height >= 1

    self.neighborhood_grid_radius = self.max_displacement / self.stride2
    self.neighborhood_grid_width = self.neighborhood_grid_radius * 2 + 1

    self.top_channels = self.neighborhood_grid_width ** 2

  def get_op_params(self):
    return [('TOP_WIDTH', str(self.top_width)),
            ('TOP_HEIGHT', str(self.top_height)),
            ('TOP_CHANNELS', str(self.top_channels)),
            ('PAD_SIZE', str(self.pad_size)),
            ('KERNEL_RADIUS', str(self.kernel_radius)),
            ('KERNEL_SIZE', str(self.kernel_size)),
            ('STRIDE1', str(self.stride1)),
            ('STRIDE2', str(self.stride2)),
            ('MAX_DISPLACEMENT', str(self.max_displacement)),
            ('NEIGHBORHOOD_GRID_RADIUS', str(self.neighborhood_grid_radius)),
            ('NEIGHBORHOOD_GRID_WIDTH', str(self.neighborhood_grid_width))]

  def __eq__(self, other):
      return (type(self) == type(other) and
              self.pad_size == other.pad_size and
              self.kernel_size == other.kernel_size and
              self.stride1 == other.stride1 and
              self.stride2 == other.stride2 and
              self.max_displacement == other.max_displacement and
              self.bottom_shape == other.bottom_shape)

  def __hash__(self):
    return (hash(type(self)) ^
            hash(self.pad_size) ^
            hash(self.kernel_size) ^
            hash(self.stride1) ^
            hash(self.stride2) ^
            hash(self.max_displacement) ^
            hash(self.bottom_shape))

  def c_code_cache_version(self):
    return (1,)


class CorrelationOp(CorrelationBaseOp):
  func_name = "APPLY_SPECIFIC(Forward_gpu)"

  def __init__(self, bottom_shape, **kwargs):
    super(CorrelationOp, self).__init__(self.func_name, bottom_shape, **kwargs)

    self.kwargs = kwargs

  def make_node(self, bottom0, bottom1):
    bottom0 = as_cuda_ndarray_variable(bottom0)
    bottom1 = as_cuda_ndarray_variable(bottom1)

    assert bottom0.ndim == 4
    assert bottom1.ndim == 4

    return Apply(self, [bottom0, bottom1], [bottom0.type(), bottom0.type(), bottom0.type()])

  def infer_shape(self, node, in_shapes):
    bottom0_shape = T.shape(node.inputs[0])
    batch_size = bottom0_shape[0]
    bchannels = bottom0_shape[1]
    pb_height = bottom0_shape[2] + 2 * self.pad_size
    pb_width = bottom0_shape[3] + 2 * self.pad_size

    pb_shape = [batch_size, bchannels, pb_height, pb_width]
    out_shape = [batch_size, self.top_channels, self.top_height, self.top_width]
    return [pb_shape, pb_shape, out_shape]

  def grad(self, inp, output_grads):
    bottom0_padded, bottom1_padded, out = self(*inp)
    grad_op = CorrelationGradOp(self.bottom_shape, **self.kwargs)

    data_grads = grad_op(*(inp + [bottom0_padded, bottom0_padded, output_grads[2]]))

    return data_grads


class CorrelationGradOp(CorrelationBaseOp):
  func_name = "APPLY_SPECIFIC(Backward_gpu)"

  def __init__(self, bottom_shape, **kwargs):
    super(CorrelationGradOp, self).__init__(self.func_name, bottom_shape, **kwargs)

  def make_node(self, bottom0, bottom1, rbot0, rbot1, out_grad):
    bottom0 = as_cuda_ndarray_variable(bottom0)
    bottom1 = as_cuda_ndarray_variable(bottom1)
    rbot0 = as_cuda_ndarray_variable(rbot0)
    rbot1 = as_cuda_ndarray_variable(rbot1)
    out_grad = as_cuda_ndarray_variable(out_grad)

    assert bottom0.ndim == 4
    assert bottom1.ndim == 4
    assert rbot0.ndim == 4
    assert rbot1.ndim == 4
    assert out_grad.ndim == 4

    return Apply(self, [bottom0, bottom1, rbot0, rbot1, out_grad],
                 [bottom0.type(), bottom0.type()])

  def infer_shape(self, node, in_shapes):
    return [in_shapes[0], in_shapes[1]]

  def grad(self, inp, output_grads):
    return [grad_undefined(self, i, inp[i]) for i in xrange(len(inp))]
