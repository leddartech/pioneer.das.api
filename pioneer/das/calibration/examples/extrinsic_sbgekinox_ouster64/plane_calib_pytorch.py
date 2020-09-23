"""
Created on 27/05/2020

@email: olivier.blondeau-fournier@leddartech.com

A script for the extrinsic calibration between the Sbgekinox IMU and a lidar (Ouster64)
based on planes and pytorch.

"""
import numpy as np
import math
import os
import torch
import torch.nn as nn
from pytorch3d.transforms import (
    so3_exponential_map,
    so3_log_map,
) # pip install pytorch3d


class PlaneCalib(nn.Module):
    def __init__(self, 
                 init_matrix=None, 
                 init_bias=None, 
                 use_bias=False,
                 use_svap_loss=False,
                 svap_weight=0,
                 outliers_ratio=0.95,
                ):
        super(PlaneCalib, self).__init__()
        self.init_matrix = init_matrix
        self.init_bias = init_bias
        self.use_bias = use_bias
        self.use_svap_loss = use_svap_loss
        self.svap_weight = svap_weight
        self.outliers_ratio = outliers_ratio

        self.set_init_transformation(init_matrix)
        self.bias=None
        if use_bias:
            self.set_init_bias(init_bias)
        else:
            print('The bias is not considered')
    
    def set_init_transformation(self, init_matrix):
        if init_matrix is None:
            log_rotation = torch.randn(3).float() 
            translation = torch.randn(3).float()
        else:
            log_rotation = so3_log_map(torch.from_numpy(init_matrix[:3, :3]).float().unsqueeze(0)).squeeze(0)
            translation = torch.from_numpy(init_matrix[:3, 3]).float()
        
        self.log_rotation = nn.Parameter(log_rotation.clone().detach())
        self.translation = nn.Parameter(translation.clone().detach())
        self.log_rotation.requires_grad = True
        self.translation.requires_grad = True

    def set_init_bias(self, init_bias):
        bias = torch.from_numpy(init_bias).float()
        self.bias = nn.Parameter(bias.clone().detach())

    def get_rotation_matrix(self):
        return so3_exponential_map(self.log_rotation.unsqueeze(0)).squeeze(0)

    def get_transformation_matrix(self):
        T = torch.eye(4)
        R = self.get_rotation_matrix()
        T[:3, :3] = R
        T[:3, 3] = self.translation
        return T

    def map_points(self, x, tr=None):
        '''Mapping the points using (rotation, translation, bias)
            x - tensor(bs,M,3)
            tr- optional - tensor(bs,M,4,4)
        '''
        if tr is not None:
            return torch.matmul(tr[:,:,:3,:3], x.unsqueeze(-1)).squeeze() + tr[:,:,:3,3]

        if self.use_bias:
            d = torch.norm(x, dim=-1, keepdim=True)
            unit_vec = x / d
            d = d + self.bias
            x = unit_vec * d
        
        R = self.get_rotation_matrix()
        x = x.permute(0,2,1).contiguous()
        y = torch.matmul(R, x).permute(0,2,1).contiguous() + self.translation
        return y

    def fit_plane(self, x):
        '''Fitting a plane on the point cloud x with svd.
        Recall that: n.(x-x0) = 0
        x - tensor(bs,M,3)
        '''
        center = torch.mean(x, dim=1, keepdim=True)
        a = x - center
        u, s, v = torch.svd(a) #this is crap, it returns v, not v^T as in np.linalg.svd
        return v[:,:,2], -torch.matmul(center, v[:,:,2].unsqueeze(-1)).squeeze(), s[:,2]
    
    def distance2_points_to_plane(self, x, n, d):
        ''' return squared distance to plane
            x - tensor(bs,M,3)
            n - tensor(bs,3)
            d - tensor(bs)
        '''
        xn = torch.bmm(x, n.unsqueeze(-1)).squeeze(-1)
        # return torch.abs(xn + d.unsqueeze(-1)) / torch.norm(n, dim=-1, keepdim=True)
        return torch.pow(xn + d.unsqueeze(-1), 2)
    
    def remove_outliers(self, x, keep_ratio):
        '''x = tensor(bs,M)
        '''
        _, m = x.size()
        n = int(keep_ratio*m)
        y,_ = torch.sort(x, dim=-1)
        return y[:,0:n]

    def loss_fn(self, l, s):
        '''return the scalar loss.
        '''
        if self.use_svap_loss:
            return torch.mean(l) + self.svap_weight * torch.mean(s)
        
        return torch.mean(l)

    def forward(self,x,y):
        ''' x - pts: tensor(bs,M,3)
            y - trajectories: tensor(bs,M,4,4)
        '''
        x = self.map_points(x)
        x = self.map_points(x, y)
        n,d,s = self.fit_plane(x)
        l = self.distance2_points_to_plane(x,n,d)
        l = self.remove_outliers(l, self.outliers_ratio)
        
        return self.loss_fn(l,s)
    




# if __name__ == '__main__':
    #testing

   
    


