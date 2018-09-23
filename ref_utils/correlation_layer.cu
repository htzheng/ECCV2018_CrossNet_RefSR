#section support_code_apply

#define ROUND_OFF 50000

#define WARPS_PER_BLOCK 1
#define THREADS_PER_WARP 32

#include <cfloat>

using std::max;
using std::min;

#define Dtype float

// The following chunks are borrowed from Caffe. -------------------------------

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      return 1; \
    } \
  } while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

// CUDA: use 512 threads per block
const int CAFFE_CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

// -----------------------------------------------------------------------------

// From the FlowNet caffe branch src/caffe/layers/correlation_layer.cu ---------

__global__ void APPLY_SPECIFIC(blob_rearrange_kernel2)(const Dtype* in, Dtype* out, int num, int channels, int width, int height, int widthheight, int padding, int pwidthheight)
{
    int xy = blockIdx.x*blockDim.x + threadIdx.x;
    if(xy>=widthheight)
        return;

    int ch = blockIdx.y;
    int n  = blockIdx.z;

    Dtype value=in[(n*channels+ch)*widthheight+xy];

    __syncthreads();

    int xpad  = (xy % width + padding);
    int ypad  = (xy / width + padding);
    int xypad = ypad * (width+2*padding) + xpad;

    out[(n*pwidthheight+xypad)*channels + ch] = value;
}


__global__ void APPLY_SPECIFIC(CorrelateData)(
  const int nthreads, int num, int topwidth, int topheight, int topchannels,
  int topcount, int max_displacement, int neighborhood_grid_radius,
  int neighborhood_grid_width, int kernel_radius, int kernel_size, int stride1,
  int stride2, int bottomwidth, int bottomheight, int bottomchannels,
  const Dtype *bottom0, const Dtype *bottom1, Dtype *top)
{
  extern __shared__ char patch_data_char[];

  Dtype *patch_data = (Dtype *)patch_data_char;

    // First (upper left) position of kernel upper-left corner in current center position of neighborhood in image 1
  int x1 = blockIdx.x*stride1 + max_displacement;
  int y1 = blockIdx.y*stride1 + max_displacement;
  int item = blockIdx.z;
  int ch_off = threadIdx.x;

  // Load 3D patch into shared shared memory
  for(int j = 0; j < kernel_size; j++) { // HEIGHT
    for(int i = 0; i < kernel_size; i++) { // WIDTH
      int ji_off = ((j * kernel_size) + i) * bottomchannels;
      for(int ch = ch_off; ch < bottomchannels; ch += (WARPS_PER_BLOCK*THREADS_PER_WARP)) { // CHANNELS
          int idx1 = ((item * bottomheight + y1+j) * bottomwidth + x1+i) * bottomchannels + ch;
          int idxPatchData = ji_off + ch;
          patch_data[idxPatchData] = bottom0[idx1];
      }
    }
  }

  __syncthreads();

  __shared__ Dtype sum[WARPS_PER_BLOCK*THREADS_PER_WARP];

  // Compute correlation
  for(int top_channel = 0; top_channel < topchannels; top_channel++) {
    sum[ch_off] = 0;

    int s2o = (top_channel % neighborhood_grid_width - neighborhood_grid_radius) * stride2;
    int s2p = (top_channel / neighborhood_grid_width - neighborhood_grid_radius) * stride2;

    for(int j = 0; j < kernel_size; j++) { // HEIGHT
      for(int i = 0; i < kernel_size; i++) { // WIDTH
        int ji_off = ((j * kernel_size) + i) * bottomchannels;
        for(int ch = ch_off; ch < bottomchannels; ch += (WARPS_PER_BLOCK*THREADS_PER_WARP)) { // CHANNELS
          int x2 = x1 + s2o;
          int y2 = y1 + s2p;

          int idxPatchData = ji_off + ch;
          int idx2 = ((item * bottomheight + y2+j) * bottomwidth + x2+i) * bottomchannels + ch;

          sum[ch_off] += patch_data[idxPatchData] * bottom1[idx2];
        }
      }
    }

    __syncthreads();

    if(ch_off == 0) {
        Dtype total_sum = 0;
        for(int idx = 0; idx < WARPS_PER_BLOCK*THREADS_PER_WARP; idx++) {
            total_sum += sum[idx];
        }
        const int sumelems = kernel_size*kernel_size*bottomchannels;
        const int index = ((top_channel*topheight + blockIdx.y)*topwidth)+blockIdx.x;
        top[index + item*topcount] = total_sum / (float)sumelems;
    }
  }

  // Aggregate
}

__global__ void APPLY_SPECIFIC(CorrelateDataBackward0)(
  const int nthreads, int num, int item, int topwidth, int topheight,
  int topchannels, int max_displacement, int neighborhood_grid_radius,
  int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
  int bottomwidth, int bottomheight, int pbottomwidth, int pbottomheight,
  int bottomchannels, int bottomcount, int pad_size,
  Dtype *bottom0diff, const Dtype *bottom1, const Dtype *topdiff)
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index % bottomchannels; //channels
    int l = (index / bottomchannels) % bottomwidth + pad_size; //w-pos
    int m = (index / bottomchannels / bottomwidth) % bottomheight + pad_size; //h-pos

    //Get X,Y ranges and clamp
    // round_off is a trick to enable integer division with ceil, even for negative numbers
    // We use a large offset, for the inner part not to become negative.
    const int round_off = ROUND_OFF;
    const int round_off_s1 = stride1 * round_off;

    // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
    int xmin = (l - 2*kernel_radius - max_displacement + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement) / stride1
    int ymin = (m - 2*kernel_radius - max_displacement + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement) / stride1

    // Same here:
    int xmax = (l - max_displacement + round_off_s1) / stride1 - round_off; // floor (l - max_displacement) / stride1
    int ymax = (m - max_displacement + round_off_s1) / stride1 - round_off; // floor (m - max_displacement) / stride1


    Dtype sum = 0;
    if(xmax>=0 && ymax>=0 && (xmin<=topwidth-1) && (ymin<=topheight-1))
    {
        xmin = max(0,xmin);
        xmax = min(topwidth-1,xmax);

        ymin = max(0,ymin);
        ymax = min(topheight-1,ymax);

        for(int p = -neighborhood_grid_radius; p <= neighborhood_grid_radius; p++) {
          for(int o = -neighborhood_grid_radius; o <= neighborhood_grid_radius; o++) {

            // Get bottom1 data:
            int s2o = stride2 * o;
            int s2p = stride2 * p;
            int idxbot1 = ((item * pbottomheight + (m+s2p)) * pbottomwidth + (l+s2o)) * bottomchannels + n;
            Dtype bot1tmp = bottom1[idxbot1]; // bottom1[l+s2o,m+s2p,n]

            // Index offset for topdiff in following loops:
            int op = (p+neighborhood_grid_radius) * neighborhood_grid_width + (o+neighborhood_grid_radius); // index [o,p]
            int idxopoffset = (item * topchannels + op);

            for(int y = ymin; y <= ymax; y++) {
              for(int x = xmin; x <= xmax; x++) {
                int idxtopdiff = (idxopoffset * topheight + y) * topwidth + x; // topdiff[x,y,o,p]
                sum += topdiff[idxtopdiff] * bot1tmp;
              }
            }
          }
        }
    }
    const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
    const int bot0index = ((n * bottomheight) + (m-pad_size)) * bottomwidth + (l-pad_size);
    bottom0diff[bot0index + item*bottomcount] = sum / (float)sumelems;
  }

}

__global__ void APPLY_SPECIFIC(CorrelateDataBackward1)(
  const int nthreads, int num, int item, int topwidth, int topheight,
  int topchannels, int max_displacement, int neighborhood_grid_radius,
  int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
  int bottomwidth, int bottomheight, int pbottomwidth, int pbottomheight,
  int bottomchannels, int bottomcount, int pad_size,
  const Dtype *bottom0, Dtype *bottom1diff, const Dtype *topdiff)
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    //int l = index % bottomwidth + pad_size; //w-pos
    //int m = (index / bottomwidth) % bottomheight + pad_size; //h-pos
    //int n = (index / bottomwidth / bottomheight) % bottomchannels; //channels
    int n = index % bottomchannels; //channels
    int l = (index / bottomchannels) % bottomwidth + pad_size; //w-pos
    int m = (index / bottomchannels / bottomwidth) % bottomheight + pad_size; //h-pos

    // round_off is a trick to enable integer division with ceil, even for negative numbers
    // We use a large offset, for the inner part not to become negative.
    const int round_off = ROUND_OFF;
    const int round_off_s1 = stride1 * round_off;

    Dtype sum = 0;
    for(int p = -neighborhood_grid_radius; p <= neighborhood_grid_radius; p++) {
      for(int o = -neighborhood_grid_radius; o <= neighborhood_grid_radius; o++) {

        int s2o = stride2 * o;
        int s2p = stride2 * p;

        //Get X,Y ranges and clamp
        // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
        int xmin = (l - 2*kernel_radius - max_displacement - s2o + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1
        int ymin = (m - 2*kernel_radius - max_displacement - s2p + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1

        // Same here:
        int xmax = (l - max_displacement - s2o + round_off_s1) / stride1 - round_off; // floor (l - max_displacement - s2o) / stride1
        int ymax = (m - max_displacement - s2p + round_off_s1) / stride1 - round_off; // floor (m - max_displacement - s2p) / stride1

        if(xmax>=0 && ymax>=0 && (xmin<=topwidth-1) && (ymin<=topheight-1))
        {
            xmin = max(0,xmin);
            xmax = min(topwidth-1,xmax);

            ymin = max(0,ymin);
            ymax = min(topheight-1,ymax);

            // Get bottom0 data:
            int idxbot0 = ((item * pbottomheight + (m-s2p)) * pbottomwidth + (l-s2o)) * bottomchannels + n;
            Dtype bot0tmp = bottom0[idxbot0]; // bottom1[l+s2o,m+s2p,n]

            // Index offset for topdiff in following loops:
            int op = (p+neighborhood_grid_radius) * neighborhood_grid_width + (o+neighborhood_grid_radius); // index [o,p]
            int idxOpOffset = (item * topchannels + op);

            for(int y = ymin; y <= ymax; y++) {
              for(int x = xmin; x <= xmax; x++) {
                int idxtopdiff = (idxOpOffset * topheight + y) * topwidth + x; // topdiff[x,y,o,p]
                sum += topdiff[idxtopdiff] * bot0tmp;
              }
            }
        }
      }
    }
    const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
    const int bot1index = ((n * bottomheight) + (m-pad_size)) * bottomwidth + (l-pad_size);
    bottom1diff[bot1index + item*bottomcount] = sum / (float)sumelems;
  }

}

// -----------------------------------------------------------------------------


// Theano specific -------------------------------------------------------------

int APPLY_SPECIFIC(Forward_gpu)(
  CudaNdarray* bottom0,
  CudaNdarray* bottom1,
  CudaNdarray** rbot0,
  CudaNdarray** rbot1,
  CudaNdarray** out)
{
  const int bnum = CudaNdarray_DIMS(bottom0)[0];
  const int bchannels = CudaNdarray_DIMS(bottom0)[1];
  const int bheight = CudaNdarray_DIMS(bottom0)[2];
  const int bwidth = CudaNdarray_DIMS(bottom0)[3];
  const int bwidthheight = bwidth * bheight;

  // Prepare outputs.
  int dims[] = {0, 0, 0, 0};
  dims[0] = bnum;
  dims[1] = TOP_CHANNELS;
  dims[2] = TOP_HEIGHT;
  dims[3] = TOP_WIDTH;

  CudaNdarray_prep_output(out, 4, dims);

  // Prepare rbot0, rbo1
  // https://github.com/liruoteng/FlowNet/blob/master/src/caffe/layers/correlation_layer.cpp#L77

  int paddedbottomheight = bheight+2*PAD_SIZE;
  int paddedbottomwidth = bwidth+2*PAD_SIZE;
  dims[1] = bchannels;
  dims[2] = paddedbottomheight;
  dims[3] = paddedbottomwidth;

  CudaNdarray_prep_output(rbot0, 4, dims);
  CudaNdarray_prep_output(rbot1, 4, dims);

  cudaMemset((*out)->devdata, Dtype(0.), CudaNdarray_SIZE(*out) * sizeof(Dtype));
  cudaMemset((*rbot0)->devdata, Dtype(0.), CudaNdarray_SIZE(*rbot0) * sizeof(Dtype));
  cudaMemset((*rbot1)->devdata, Dtype(0.), CudaNdarray_SIZE(*rbot1) * sizeof(Dtype));

  // Those capitalized parameters are taken from theano
  const int topcount = TOP_WIDTH * TOP_HEIGHT * TOP_CHANNELS;

  dim3 threadsPerBlock(THREADS_PER_WARP * WARPS_PER_BLOCK);

  int threads_per_block = 16;
  dim3 totalBlocksRearr((bwidthheight-1)/threads_per_block+1, bchannels, bnum);
  const int pwidthheight = (bwidth + 2 * PAD_SIZE) * (bheight + 2 * PAD_SIZE);

  APPLY_SPECIFIC(blob_rearrange_kernel2)<<<totalBlocksRearr,threads_per_block>>>
    (bottom0->devdata,(*rbot0)->devdata,bnum,bchannels,bwidth,bheight,bwidthheight,PAD_SIZE,pwidthheight);

  APPLY_SPECIFIC(blob_rearrange_kernel2)<<<totalBlocksRearr,threads_per_block>>>
    (bottom1->devdata,(*rbot1)->devdata,bnum,bchannels,bwidth,bheight,bwidthheight,PAD_SIZE,pwidthheight);

  const int num = bnum;
  const int channels = bchannels;
  const int height = bheight + 2*PAD_SIZE;
  const int width = bwidth + 2*PAD_SIZE;

  const int shared_memory_per_block = (KERNEL_SIZE*KERNEL_SIZE)*bchannels;

  int topThreadCount = topcount;

  dim3 totalBlocksCorr(TOP_WIDTH, TOP_HEIGHT, num);

  APPLY_SPECIFIC(CorrelateData)<<<totalBlocksCorr, threadsPerBlock, shared_memory_per_block * sizeof(Dtype)>>>(
    topThreadCount, num, TOP_WIDTH, TOP_HEIGHT, TOP_CHANNELS, topcount,
    MAX_DISPLACEMENT, NEIGHBORHOOD_GRID_RADIUS, NEIGHBORHOOD_GRID_WIDTH,
    KERNEL_RADIUS, KERNEL_SIZE, STRIDE1, STRIDE2, width, height, channels,
    (*rbot0)->devdata, (*rbot1)->devdata, (*out)->devdata);
  CUDA_POST_KERNEL_CHECK;

  return 0;
}

int APPLY_SPECIFIC(Backward_gpu)(
  CudaNdarray* bottom0,
  CudaNdarray* bottom1,
  CudaNdarray* rbot0,
  CudaNdarray* rbot1,
  CudaNdarray* out_grad,
  CudaNdarray** bottom0_grad,
  CudaNdarray** bottom1_grad)
{
  int count = CudaNdarray_SIZE(bottom0);

  CudaNdarray_prep_output(bottom0_grad, 4, CudaNdarray_DIMS(bottom0));
  CudaNdarray_prep_output(bottom1_grad, 4, CudaNdarray_DIMS(bottom1));

  cudaMemset((*bottom0_grad)->devdata, Dtype(0.), count * sizeof(Dtype));
  cudaMemset((*bottom1_grad)->devdata, Dtype(0.), count * sizeof(Dtype));

  // Get top diff, compute bottom diff
  const Dtype* top_diff = out_grad->devdata;

  const Dtype* bottom0_data = bottom0->devdata;
  const Dtype* bottom1_data = bottom1->devdata;

  const int num = CudaNdarray_DIMS(bottom0)[0];
  const int channels = CudaNdarray_DIMS(bottom0)[1];
  const int height = CudaNdarray_DIMS(bottom0)[2];
  const int width = CudaNdarray_DIMS(bottom0)[3];

  const int paddedheight = height + 2 * PAD_SIZE;
  const int paddedwidth = width + 2 * PAD_SIZE;

  const int bottomcount = channels * height * width;

  int botThreadCount = bottomcount;

  // CorrelationLayerBackward
  Dtype* bottom0_diff = (*bottom0_grad)->devdata;
  Dtype* bottom1_diff = (*bottom1_grad)->devdata;

  dim3 totalBlocksBackward0(width, height, channels * num); //First dim is fastest
  dim3 threadsPerBlockBackward0(THREADS_PER_WARP * WARPS_PER_BLOCK);
  const int buffer_size_backw0 = ((int)ceil((float)(2 * KERNEL_RADIUS) / (float)STRIDE1) + 1) * TOP_CHANNELS;

  // == Run kernel Backward 0
  for(int n = 0; n < num; n++) {
  //Bottom0:
  APPLY_SPECIFIC(CorrelateDataBackward0)<<<CAFFE_GET_BLOCKS(botThreadCount), CAFFE_CUDA_NUM_THREADS>>>(
      botThreadCount,
      num, n, TOP_WIDTH, TOP_HEIGHT, TOP_CHANNELS,
      MAX_DISPLACEMENT, NEIGHBORHOOD_GRID_RADIUS, NEIGHBORHOOD_GRID_WIDTH, KERNEL_RADIUS,
      STRIDE1, STRIDE2,
      width, height, paddedwidth, paddedheight, channels, bottomcount, PAD_SIZE,
      bottom0_diff, rbot1->devdata, top_diff
      );

  CUDA_POST_KERNEL_CHECK;
  }

  // == Run kernel Backward 1
  for(int n = 0; n < num; n++) {
  APPLY_SPECIFIC(CorrelateDataBackward1)<<<CAFFE_GET_BLOCKS(botThreadCount), CAFFE_CUDA_NUM_THREADS>>>(
      botThreadCount,
      num, n, TOP_WIDTH, TOP_HEIGHT, TOP_CHANNELS,
      MAX_DISPLACEMENT, NEIGHBORHOOD_GRID_RADIUS, NEIGHBORHOOD_GRID_WIDTH, KERNEL_RADIUS,
      STRIDE1, STRIDE2,
      width, height, paddedwidth, paddedheight, channels, bottomcount, PAD_SIZE,
      rbot0->devdata, bottom1_diff, top_diff
      );

  CUDA_POST_KERNEL_CHECK;
  }
  return 0;
}

// -----------------------------------------------------------------------------