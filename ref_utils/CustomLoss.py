import torch
import torch.nn as nn
import torch.nn.functional as func


class CharbonnierLoss(nn.Module):

    def __init__(self):
        super(CharbonnierLoss,self).__init__()

    def forward(self,pre,gt):

        N = pre.shape[0]

        diff = torch.sum(torch.sqrt((pre - gt ).pow(2) + 0.001 **2)) / N

        return diff


class EuclideanLoss(nn.Module):

    def __init__(self):
        super(EuclideanLoss,self).__init__()

    def forward(self,pre,gt):

        N = pre.shape[0]        
        diff = torch.sum((pre - gt ).pow(2)) / (N * 2)

        return diff
