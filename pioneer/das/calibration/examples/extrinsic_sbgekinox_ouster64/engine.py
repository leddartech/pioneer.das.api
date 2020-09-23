"""
Created on 02/06/2020

@email: olivier.blondeau-fournier@leddartech.com

A script for the optimization of the extrinsic calibration between the Sbgekinox IMU and a lidar (Ouster64)
based on planes and pytorch.

"""
import numpy as np
import math
import os
import torch
import torch.nn as nn
import visdom
import pickle
from typing import Iterable
import argparse
from plane_calib_pytorch import PlaneCalib
from convert_dataset import (PlaneCalibDataset, PlaneCalibDataloader)
import das.imu as DI
import time
vis = visdom.Visdom()

def evaluate(dataset, current_matrix, subfolder_, bias=0.0):
    if not os.path.exists(subfolder_):
        os.makedirs(subfolder_)
        
    for i in range(len(dataset)):
        start = time.time()
        pts, tr = dataset.get_points(i, current_matrix)
        
        norm = np.linalg.norm(pts.T[:3,:], axis=0)
        
        pts.T[:3,:] = (pts.T[:3,:]/norm)*(norm+bias)
    
        pts_h = np.vstack([pts.T,np.ones((1,pts.shape[0]))])
        pts_imu = current_matrix @ pts_h
        pts_plan = np.zeros_like(pts_imu)
        for j in range(pts.shape[0]):
            pts_plan[:,j] = tr[j,:,:] @ pts_imu[:,j] #pts_du_plan = tr @ T_calib @ pts

        np.savetxt(subfolder_+f"/plane_point{i}.txt", pts_plan.T[:,:3], fmt='%.3f')
        stop = time.time()
        print(pts.shape, tr.shape, stop-start)


def train_one_epoch(model: torch.nn.Module, 
                    data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    current_iteration: int,
                    visdom_window_loss: visdom.Visdom,
                    ):
    
    for pts, tr in data_loader:
        current_iteration+=1
        loss = model(pts, tr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.clone().detach().cpu().numpy()
        vis.line(X=np.array([current_iteration]),Y=np.array([loss_value]),update='append', win=visdom_window_loss)

    return loss_value, current_iteration

def create_plot_window(vis, xlabel, ylabel, title):
	return vis.line(X=np.array([1]), Y=np.array([np.nan]), opts=dict(xlabel=xlabel, ylabel=ylabel, title=title))

def save_checkpoint(state, file_name):
    torch.save(state, file_name+'.pt')

def main(args):
    #create folder if not exist for ckps:
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

    #load an initial calib:
    with open(args.dataset+'/extrinsics/sbgekinox_bcc-'+args.sensor+'.pkl', 'rb') as k:
        init_calib_matrix = DI.inverse_transform(pickle.load(k))
    print(f'_____The initial matrix from {args.sensor} to imu is:', init_calib_matrix)
    DI.print_transf_result(init_calib_matrix)
    #define model:
    
    print('_____Define the model')
    plane_calib_net = PlaneCalib(init_calib_matrix, 
                                    np.array([args.init_bias]), 
                                    args.use_bias,
                                    args.use_svap_loss,
                                    args.svap_weight,
                                    args.outliers_ratio)
    device = torch.device(args.device)
    plane_calib_net.to(device)
    
    #dataloader:
    print('_____Define the loader')
    dataset = PlaneCalibDataset(args.dataset,
                                args.planes_cfg,
                                args.sensor,
                                args.exported_package,
                                init_calib_matrix)
    loader = PlaneCalibDataloader(dataset, 
                                    args.num_max_points, 
                                    args.batch_size, 
                                    device, 
                                    args.shuffle)
    
    # optim:
    if args.set_optim is 'Adam':
        optimizer = torch.optim.AdamW(plane_calib_net.parameters(), 
                                    lr=args.lr,
                                    weight_decay=args.weight_decay)
    elif args.set_optim is 'SGD':
        optimizer = torch.optim.SGD(plane_calib_net.parameters(), 
                                    lr=args.lr, 
                                    momentum=args.momentum)
    else:
        print('No optimizer has been set')
        exit()
    
    print("_____Start training_____")
    plane_calib_net.train()
    vis_loss = create_plot_window(vis, '#Iterations', 'Loss', 'Training Loss')
    loss_epoch = []
    matrix_epoch = []
    bias_epoch = []
    n_iter = 0
    for epoch in range(args.num_epochs):
        l, n_iter = train_one_epoch(
                        plane_calib_net, 
                        loader, 
                        optimizer,
                        device,
                        n_iter,
                        vis_loss
                    )
        loss_epoch.append(l)
        #update data from the current calib matrix:
        current_tr_calib = plane_calib_net.get_transformation_matrix().detach().cpu().numpy()
        try:
            current_bias = plane_calib_net.bias.detach().cpu().numpy()
        except:
            current_bias = 0.0
        
        if args.epoch_use_new_calib:
            loader.update_calibration_matrix(current_tr_calib)
        else:
            loader.update_calibration_matrix(init_calib_matrix)
        
        matrix_epoch.append(current_tr_calib)
        bias_epoch.append(current_bias)

        print(f'Current calibration and bias at epoch {epoch+1}:')
        DI.print_transf_result(current_tr_calib)
        print(current_bias)

        print('Saving checkpoint')
        save_checkpoint(plane_calib_net.state_dict(), args.checkpoint+'planecalib_epoch_'+str(epoch+1))
        
        if epoch%args.eval_num_epochs==0:
            print('_____Evaluation_____')
            evaluate(dataset, current_tr_calib, args.checkpoint+'validation_'+str(epoch+1), current_bias)

        package = dict(loss=loss_epoch, calib=matrix_epoch, bias=bias_epoch)
        np.save(args.checkpoint+'results.npy', package)
    
    print("_____End training_____")
    return True



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IMU ouster/pixell calibration optimizer engine based on PyTorch, PyTorch3D')
    
    #dataset
    parser.add_argument('--dataset', default='/nas/pixset/exportedDataset/20200608_180144_rec_dataset_calib_imu_fast_brick05_exported', 
                            type=str, help='dataset platform folder')
    parser.add_argument('--planes_cfg', default='planes_ouster_20200519_201503.yml', type=str,
                            help='yml file for planes coordinates')
    parser.add_argument('--sensor', default='ouster64_bfc', type=str, help='sensor name')
    parser.add_argument('--exported_package', default='exported_dataset_brick05.pkl', type=str,
                            help='exported dataset pre-processed')
    parser.add_argument('--use_bias', action='store_true')
    parser.add_argument('--init_bias', default=-0.001, type=float, help='initial bias to start with')
    parser.add_argument('--epoch_use_new_calib', action='store_true', 
                            help='use the new current calibration matrix for new selection of points')


    #loader
    parser.add_argument('--num_max_points', default=5000, type=int, help='max number of returned points per plane')
    parser.add_argument('--batch_size', default=9, type=int, help='the batch size')
    parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
    parser.add_argument('--shuffle', action='store_true', help='if you want the batch to be shuffled')

    #model
    parser.add_argument('--use_svap_loss', action='store_true', help='Use the smallest svd eigenvalue in the loss function')
    parser.add_argument('--svap_weight', default=1e-3, type=float, help='if --use_svap_loss, then this is the weight assosiated to that term')
    parser.add_argument('--outliers_ratio', default=0.95, type=float, help='the ratio of keeped points when removing outliers')
    parser.add_argument('--set_optim', default='SGD', type=str, help='Optimizer, either SGD or Adam')
    parser.add_argument('--lr', default=5e-2, type=float)
    parser.add_argument('--momentum', default=0.7, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--num_epochs', default=500, type=int)
    parser.add_argument('--eval_num_epochs', default=10, type=int, help='evaluate (ie compute planes) after n epochs')
    parser.add_argument('--checkpoint', default='ckps/')

    args = parser.parse_args()
    
    #Fire:
    main(args)



    

   
    


