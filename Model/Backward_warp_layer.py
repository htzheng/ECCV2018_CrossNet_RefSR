import torch
import torch.nn as nn
import torch.nn.functional as func

class Backward_warp(nn.Module):

    def __init__(self):
        super(Backward_warp,self).__init__()


    def _meshgrid(self,height,width):

        y_t = torch.linspace(0,height - 1, height).reshape(height,1) * torch.ones(1,width)
        x_t = torch.ones(height,1) * torch.linspace(0, width - 1, width).reshape(1,width)

        x_t_flat = x_t.reshape(1,1,height,width)
        y_t_flat = y_t.reshape(1,1,height,width)

        grid = torch.cat((x_t_flat,y_t_flat),1)

        return grid


    def _interpolate(self,img , x, y , out_height, out_width):

        num_batch,height,width,num_channel = img.size()
        height_f = float(height)
        width_f = float(width)

        x = torch.clamp(x,0,width - 1)
        y = torch.clamp(y,0,height - 1)

        x0_f = x.floor()
        y0_f = y.floor()
        x1_f = x0_f + 1.0
        y1_f = y0_f + 1.0

        x0 = torch.tensor(x0_f, dtype = torch.int64)
        y0 = torch.tensor(y0_f, dtype = torch.int64)
        x1 = torch.tensor(torch.clamp(x1_f, 0, width_f -1), dtype = torch.int64)
        y1 = torch.tensor(torch.clamp(y1_f, 0, height_f -1), dtype = torch.int64)
 
        dim1 = width * height
        dim2 = width
        base = torch.tensor((torch.arange(num_batch) * dim1),dtype = torch.int64)
        base = base.reshape(num_batch,1).repeat(1,out_height * out_width).view(-1)

        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2

        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        img_flat = img.reshape(-1,num_channel)

        Ia = img_flat[idx_a]
        Ib = img_flat[idx_b]
        Ic = img_flat[idx_c]
        Id = img_flat[idx_d]

        wa = ((x1_f-x) * (y1_f-y)).reshape(-1,1)
        wb = ((x1_f-x) * (y-y0_f)).reshape(-1,1)
        wc = ((x-x0_f) * (y1_f-y)).reshape(-1,1)
        wd = ((x-x0_f) * (y-y0_f)).reshape(-1,1)
        output = wa * Ia + wb * Ib + wc * Ic + wd *Id

        return output


    def _transform_flow(self,flow,input,downsample_factor):

        num_batch,num_channel,height,width = input.size()

        out_height = height
        out_width = width
        grid = self._meshgrid(height, width)
        if num_batch > 1:
            grid = grid.repeat(num_batch,1,1,1)

        control_point = grid.cuda() + flow
        input_t = input.permute(0,2,3,1)

        x_s_flat = control_point[:,0,:,:].contiguous().view(-1)
        y_s_flat = control_point[:,1,:,:].contiguous().view(-1)

        input_transformed = self._interpolate(input_t,x_s_flat,y_s_flat,out_height,out_width)

        input_transformed = input_transformed.reshape(num_batch,out_height,out_width,num_channel)

        output = input_transformed.permute(0,3,1,2)

        return output

    def forward(self,input,flow,downsample_factor = 1):

        return self._transform_flow(flow,input, downsample_factor)

