
'''this is the validation script, reproduced from das-rav4-dev-mapping
'''
from pioneer.common.colors import amplitudes_to_color
from pioneer.das.api import platform, interpolators

import pioneer.das.api.imu

import argparse
import collections 
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import yaml






from convert_dataset import SensorLidar
import os
import pickle

class GenerateVeloMap():

    def save_map_to_xyz(self, filename):
        map = np.vstack(self.pts_map)
        colors = amplitudes_to_color(map[:,3].astype('f4'))
        pts_map = np.hstack((map[:,0:3], colors))

        np.savetxt(filename, pts_map.astype('f8'))

    def new_pose_keeped(self, current_pose):
        diff = current_pose - self.last_pose

        distance = np.linalg.norm(diff[0:3])
        angle_diff = np.abs(current_pose[5] - self.last_pose[5])

        if(distance > self.min_distance): 
            self.last_pose = current_pose
            return True
        else:
            return False

    def __init__(self, platform_path, calibration, min_distance, sensor_name, fov_limitation, T_vlp16_to_leddar):
        self.pts_map = collections.deque()
        self.sensor_name = sensor_name
        self.sensor_type = sensor_name.split('_')[0]
        self.t_calib = calibration
        self.min_distance = min_distance
        self.pf = platform.Platform(platform_path, ignore=['radarTI_bfc'])


        self.fov_limitation = fov_limitation
        self.T_vlp16_to_leddar = T_vlp16_to_leddar
        if self.fov_limitation:
            assert self.sensor_type == 'vlp16', 'fov_limitation option is usable only from vlp16 sensor'
            assert self.fov_limitation in self.pf.sensors.keys(), '{} is not in platform sensors list'.format(self.fov_limitation)
            assert self.fov_limitation in self.T_vlp16_to_leddar.keys(), 'No calibration for vlp16 to {}'.format(self.fov_limitation)

        if self.sensor_type == 'lca2' or self.sensor_type=='pixell':
            self.sensor_lidar = SensorLidar(self.sensor_name, self.pf, np.array([0,0,0]), nb_point_per_interpolation_traj = 16)
        elif self.sensor_type == 'eagle':
            self.sensor_lidar = SensorLidar(self.sensor_name, self.pf, np.array([0,0,0]), nb_point_per_interpolation_traj = 16)
        elif self.sensor_type == 'vlp16':
            self.sensor_lidar = SensorLidar(self.sensor_name, self.pf, np.array([0,0,0]))
        elif self.sensor_type == 'ouster64':
            self.sensor_lidar = SensorLidar(self.sensor_name, self.pf, np.array([0,0,0]),nb_point_per_interpolation_traj = 25)
        else:
            assert('unknown sensor:{}'.format(self.sensor_name))

        #self.vlp = self.pf['vlp16_tfc_xyzit']
        self.nav = self.pf['sbgekinox_bcc_navposvel']
        self.euler = self.pf['sbgekinox_bcc_ekfeuler']
        self.sensor_lidar.print_nb_frame()
        print('nb frame nav:{}'.format(len(self.nav)))
        print('nb frame euler:{}'.format(len(self.euler)))
        
        self.last_pose = np.zeros(6)
        self.imu_accuracy = {}
        self.imu_accuracy['roll_acc'] = []
        self.imu_accuracy['pitch_acc'] = []
        self.imu_accuracy['yaw_acc'] = []
        self.imu_accuracy['latitude_acc'] = []
        self.imu_accuracy['longitude_acc'] = []
        self.imu_accuracy['altitude_acc'] = []

        nav_0 = self.nav[0].raw

        self.imu_offset= np.array([nav_0['latitude'],nav_0['longitude'],nav_0['altitude'], np.float64(0.0), np.float64(0.0), np.float64(0.0)])
        self.pose_offset = das.imu.imu_to_pose(self.imu_offset)
        self.pose_offset[5] = 0.0

        # pour test
        self.pose_offset = np.zeros(6)


    def display_points_map(self, current_scan):
        if len(self.pts_map) != 0:
            map = np.vstack(self.pts_map)
            colors = amplitudes_to_color(map[:,3].astype('f4'))
            self.pc.set_points(map[:, 0:3], colors)
            #map = np.vstack((map, current_scan))
        #else: 
        #    map = current_scan
        
    # def trajectory(self, vlp_frame):
    #     start = time.time()
    #     get_at_timestamp_time = []
    #     get_at_timestamp_time2 = []
        
    #     if vlp_frame['t'].shape[0]%100 == 0:
    #         end = int(vlp_frame['t'].shape[0]/100)
    #     else:
    #         end = int(vlp_frame['t'].shape[0]/100)+1
    #     traj = np.zeros((end,4,4), dtype = np.dtype('f8'))
    #     # on va en conserver que 100 par scan
        
    #     for i in range(end):
    #         ts = vlp_frame['t'][i*100]
    #         #start2 = time.time()
    #         nav = self.nav.get_at_timestamp(ts, interpolator=interpolators.euler_imu_linear_ndarray_interpolator).raw # latitude        longitude        altitude
    #         euler = self.euler.get_at_timestamp(ts, interpolator=interpolators.euler_imu_linear_ndarray_interpolator).raw # rool  pitch  yaw
    #         #get_at_timestamp_time.append(time.time()-start2)
    #         #print([nav['latitude'],nav['longitude'],euler['yaw']])
            
    #         self.imu_accuracy['roll_acc'].append(euler['roll_acc'])
    #         self.imu_accuracy['pitch_acc'].append(euler['pitch_acc'])
    #         self.imu_accuracy['yaw_acc'].append(euler['yaw_acc'])
    #         self.imu_accuracy['latitude_acc'].append(nav['latitude_acc'])
    #         self.imu_accuracy['longitude_acc'].append(nav['longitude_acc'])
    #         self.imu_accuracy['altitude_acc'].append(nav['altitude_acc'])


    #         imu = das.imu.get_imu_pose(nav, euler)
            
    #         pose = das.imu.imu_to_pose(imu)
    #         #start2 = time.time()
    #         Tr = das.imu.pose_to_transf(pose)
    #         #get_at_timestamp_time2.append(time.time()-start2)
    #         traj[i] = Tr
    #     #t_get_timestamp = np.array(get_at_timestamp_time)
    #     #t_get_timestamp2 = np.array(get_at_timestamp_time2)
    #     #print('trajectory :{}, get_at_timestamp:{}, struct array:{}'.format(time.time()-start,np.sum(t_get_timestamp), np.sum(t_get_timestamp2)))
    #     return traj

    def generate(self, base_path):
        file_number = 0
        for frame in range(len(self.sensor_lidar)):
            
            #vlp_frame = self.vlp[frame].raw
            ts =  self.sensor_lidar.get_frame_timestamp(frame)
            nav = self.nav.get_at_timestamp(ts, interpolator=interpolators.euler_imu_linear_ndarray_interpolator).raw # latitude        longitude        altitude
            euler = self.euler.get_at_timestamp(ts, interpolator=interpolators.euler_imu_linear_ndarray_interpolator).raw # rool  pitch  yaw
            #get_at_timestamp_time.append(time.time()-start2)
            #print([nav['latitude'],nav['longitude'],euler['yaw']])
            #print('euler {}'.format(euler.dtype))
            imu = das.imu.get_imu_pose(nav, euler)
            
            pose = das.imu.imu_to_pose(imu)
            
            if self.new_pose_keeped(pose):
                print('frames {}/{} keeped'.format(frame, len(self.sensor_lidar)))

                pts, pts_ts, intensity = self.sensor_lidar.get_point_cloud_ts_and_intensity(frame)
                if self.fov_limitation:
                    selection = self.sensor_lidar.vlp16_fov_limitation_to_leddar(pts, self.fov_limitation, self.T_vlp16_to_leddar[self.fov_limitation])
                    pts = pts[:, selection]
                    pts_ts = pts_ts[selection]
                    intensity = intensity[selection]
                    if not np.any(selection):
                        self.last_pose = np.zeros(6) # reinit new_pose_keeped
                        continue

                    print(np.any(selection))
                #print('pts.shape {}'.format(pts.shape))
                #traj = self.sensor_lidar.trajectory(pts_ts)
                ts =  self.sensor_lidar.get_frame_timestamp(frame)
                
                # pour moi ca sert à rien ca fait la meme chose dans les deux cas !
                # if self.sensor_type:
                #     distances = np.linalg.norm(pts.T[:,[0,2]], axis = 1)
                # else:
                #     distances = np.linalg.norm(pts.T[:,:2], axis = 1)
                distances = np.linalg.norm(pts.T[:,[0,2]], axis = 1)
                d_min = 0.2
                d_max = 50.0
                pts = pts[:,(distances>d_min) & (distances<d_max)]
                pts_ts = pts_ts[(distances>d_min) & (distances<d_max)]
                intensity = intensity[(distances>d_min) & (distances<d_max)]
                if pts.shape[1] == 0: # si pas de point, on passe au frame suivant
                    continue
                traj = self.sensor_lidar.trajectory(pts_ts)
                if traj is None:
                    continue
                pts_global = self.sensor_lidar.get_undistord_point_cloud(pts, traj, self.t_calib)

                if self.sensor_type in ['eagle']:
                    colors = amplitudes_to_color(intensity.astype('f8'),log_normalize=True)
                elif self.sensor_type in ['ouster64']:
                    #intensity = 255.0*((intensity - np.min(intensity))/(np.max(intensity)-np.min(intensity)))
                    intensity = np.clip(intensity,0,12000)
                    colors = amplitudes_to_color(intensity)#,log_normalize=True)
                else:
                    colors = amplitudes_to_color(intensity.astype('f8'))
                
                pts_keep = np.concatenate([pts_global.T[:, 0:3],
                                        colors], axis=1).astype('f8')
                self.pts_map.append(pts_keep)
                if len(self.pts_map) > 100:
                    filename = '{}_{}_{:04}.xyz'.format(base_path, self.sensor_name, file_number)
                    print('save: {}'.format(filename))
                    self.save_map_to_xyz(filename)
                    self.pts_map.clear()
                    file_number += 1
            else:
                print('frames {}/{}'.format(frame, len(self.sensor_lidar)))
        if len(self.pts_map) > 0:
            filename = '{}_{}_{:04}.xyz'.format(base_path, self.sensor_name, file_number)
            print('save: {}'.format(filename))
            self.save_map_to_xyz(filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IMU Lidar (VLP16, Ouster64, Pixell) map generation, the scan correction is integrated in the script.')
    parser.add_argument('dataset', help='dataset platform folder')
    parser.add_argument('sensor', help='sensor name to map, example ouster64_bfc')
    parser.add_argument('--distance', type=float, default = 10.0, help='minimum distance between two keeped scans')
    parser.add_argument('--fov_limitation', help='limit vlp16 fov to leddar name')
    parser.add_argument('--calibration_folder', help='folder with the pickle calibration files (sbgekinox_bcc-sensor_name.pkl)')
    parser.add_argument('--ignore_calibration_file', action='store_true', default=False, help='ignore the pickle calibration, and use the coded matrix in the script')
    #choices=[range(0.0, 50.0)]
    arg = parser.parse_args()
    T_calib = np.array([    [ 9.88468007e-01,  3.77341815e-02,  1.46653095e-01,  1.57051846e+00],
                            [-3.95875233e-02,  9.99168649e-01,  9.73855452e-03,  1.14856257e-03],
                            [-1.46163699e-01, -1.54318824e-02,  9.89140046e-01,  1.40000000e+00],
                            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])



    T_calib = np.array([    [ 0.98850244,  0.03559783,  0.14695485,  1.5749707 ],
                            [-0.03741884,  0.99925312,  0.00964495,  0.01203689],
                            [-0.14650176, -0.01503294,  0.98909618,  1.4       ],
                            [ 0.        ,  0.        ,  0.        ,  1.        ]])
  
    # square planes with interpolator
    T_calib = np.array([    [ 0.98848575,  0.03606495,  0.14695321,  1.57264353],
                            [-0.03790523,  0.99923386,  0.0097409 ,  0.01595938],
                            [-0.14648932, -0.01519904,  0.98909548,  1.4       ],
                            [ 0.        ,  0.        ,  0.        ,  1.        ]])

    # dataset 2 vlp16
    T_calib = np.array([   [ 0.98867498,  0.02897708,  0.14724847,  1.65755482],
                         [-0.03104416,  0.99944884,  0.01175886, -0.01139826],
                         [-0.14682658, -0.01619689,  0.98902963,  1.40075462],
                         [ 0.        ,  0.        ,  0.        ,  1.        ]])

    T_calib = np.array([[ 0.98847631,  0.03108675,  0.14814922,  1.67864856],
                        [-0.03307972,  0.99939211,  0.01100693, -0.00204176],
                        [-0.14771699, -0.01578083,  0.98890376,  1.40383142],
                        [ 0.        ,  0.        ,  0.        ,  1.        ]]
                        )
    # vlp sur dataset 2 (premier avec maxime a coté garage)
    T_calib_0_pose = np.array([ 1.67637268e+00, -2.39444640e-03,  1.40662774e+00, -9.17841145e-01,  8.48818228e+00, -1.92135757e+00])
    T_calib_0_pose[3:] = np.deg2rad(T_calib_0_pose[3:])
    T_calib = das.imu.pose_to_transf(T_calib_0_pose)




    # todo tracer les poses de l'imu qui sont utilisées voir l'ampleur du bruit

    # # no square planes with interpolator
    # T_calib = np.array(    [[ 0.98842765,  0.03915158,  0.14655351,  1.57063527],
    #                         [-0.0409916 ,  0.9991138 ,  0.00955514,  0.00664785],
    #                         [-0.14604954, -0.01545202,  0.98915659,  1.4       ],
    #                         [ 0.        ,  0.        ,  0.        ,  1.        ]])

    # # eagle : imu2vlp16 + vlp2eagle
    # T_calib = np.array([    [-0.02529548, -0.07908455,  0.99654693,  1.70488433],
    #                         [-0.99966361, -0.00371018, -0.02566903,  0.11137654],
    #                         [ 0.00572739, -0.99686101, -0.07896409,  1.09722738],
    #                         [ 0.        ,  0.        ,  0.        ,  1.        ]])

    # # eagle : imu2eagle by planes calib
    # T_calib = np.array([    [ 0.01421997, -0.19451284,  0.98079689,  2.11144763],
    #                         [-0.99988838, -0.00726339,  0.01305628, -0.16219299],
    #                         [ 0.0045843 , -0.98087308, -0.19459441,  1.1       ],
    #                         [ 0.        ,  0.        ,  0.        ,  1.        ]])

    # init de la convergence plane
    # T_calib_0_pose = np.array([1.80, 0.0, 1.1, -95.0, 0.0, -90.0]) 
    # T_calib_0_pose = np.array([1.79170706e+00, -1.53428598e-02, 1.10000000e+00, -9.44661596e+01, -1.81417602e+00, -9.16041198e+01])
    # T_calib_0_pose = np.array([1.78162278e+00, -5.49650606e-02, 1.10338279e+00, -9.50343155e+01, -1.20405701e+00, -1.03148226e+02])
    # T_calib_0_pose = np.array([1.80, 0.0, 1.1, -95.0, 0.0, -90.0])
    # first calibration dataset 2 ou 3
    # T_calib_0_pose = np.array([ 2.04097868e+00,  3.72052944e-02,  1.09514323e+00, -9.47315957e+01, -5.52035835e-02, -9.11844678e+01])
    # T_calib_0_pose = np.array([ 2.02454669e+00,  5.44563411e-02,  1.00000000e+00, -9.36536761e+01,  1.87726692e-01, -9.13110362e+01])
    # T_calib_0_pose[3:] = np.deg2rad(T_calib_0_pose[3:])
    # T_calib = das.imu.pose_to_transf(T_calib_0_pose)

    # # lca2 lca2_bfrl test d'init 
    # T_calib_0_pose = np.array([3, -1, 0, -90.0, 0, -120.0])
    # T_calib_0_pose[3:] = np.deg2rad(T_calib_0_pose[3:])
    # T_calib = das.imu.pose_to_transf(T_calib_0_pose)
    # # lca2 lca2_bfrl first calib sur dataset /data/exportedDataset0/20190613_163053_rec_dataset_calib_LCA2_imu1_exported
    # T_calib_0_pose = np.array([3.96958783e+00, -7.28560520e-01, 3.70230554e-02, -9.09408154e+01, 5.44158221e-01, -1.19931683e+02]) 
    # # avec les planes replacées
    # T_calib_0_pose = np.array([4.02941309e+00, -8.19540763e-01, 1.56014704e-02, -8.86186542e+01, -1.58790012e+00, -1.20079602e+02])
    # # avec les planes replacées and new timing banks
    # T_calib_0_pose = np.array([4.04219113e+00, -8.24039655e-01, 1.33788626e-02, -8.85738966e+01, -1.58535591e+00, -1.20201014e+02])
    # T_calib_0_pose[3:] = np.deg2rad(T_calib_0_pose[3:])
    # T_calib = das.imu.pose_to_transf(T_calib_0_pose)
    
    # lca2 lca2_bfrl first calib sur dataset /data/exportedDataset0/20190613_163434_rec_dataset_calib_LCA2_imu2_exported
    # T_calib_0_pose = np.array([3.81706150e+00, -7.40726218e-01, 5.45579317e-04, -9.86877140e+01, -6.09818413e+00, -1.20315956e+02]) 
    # T_calib_0_pose[3:] = np.deg2rad(T_calib_0_pose[3:])
    # T_calib = das.imu.pose_to_transf(T_calib_0_pose)

    # avec planes replacés sur dataset /data/exportedDataset0/20190613_163434_rec_dataset_calib_LCA2_imu2_exported
    # T_calib_0_pose = np.array([ 3.87043055e+00, -7.00000000e-01, 1.19393130e-02, -8.83438181e+01, -1.36076383e+00, -1.20018361e+02])

    # T_calib_0_pose = np.array([3.92370029, -0.01082097,  0.223879, -0.41173777,  1.32311163, -0.72022258])
    # T_calib_0_pose[3:] = np.deg2rad(T_calib_0_pose[3:])
    # T_calib = das.imu.pose_to_transf(T_calib_0_pose)

    T_calib = np.array([[ 0.99960005,  0.00626144,  0.02757868,  3.9000185 ],
                        [-0.00638757,  0.99996954,  0.00448801,  0.00433442],
                        [-0.02754974, -0.00466237,  0.9996096 ,  0.31463706],
                        [ 0.        ,  0.        ,  0.        ,  1.        ]],
                        dtype=np.float64)


    #Pixset: (Ouster)
    # (7a)
    # T_calib = np.array([[ 0.99957794,  0.00700371,  0.0281927 ,  3.894417  ],
    #                     [-0.00717837,  0.99995565,  0.00609865, -0.02152431],
    #                     [-0.02814874, -0.00629846,  0.9995839 ,  0.19157389],
    #                     [ 0.        ,  0.        ,  0.        ,  1.        ]],
    #                     dtype=np.float64)

    #(7b)
    # T_calib = np.array([[ 0.99958146,  0.00745099,  0.02795299,  3.8970003 ],
    #                     [-0.0075772 ,  0.99996156,  0.00441208, -0.01547982],
    #                     [-0.02791904, -0.00462204,  0.9995995 ,  0.21045506],
    #                     [ 0.        ,  0.        ,  0.        ,  1.        ]],
    #                     dtype=np.float64)

    #best:
    # T_calib = np.array([[ 0.99968183,  0.00936921,  0.02341823,  3.923609  ],
    #                     [-0.00949881,  0.99994016,  0.00542918, -0.01127126],
    #                     [-0.02336596, -0.0056499 ,  0.99971104,  0.22386882],
    #                     [ 0.        ,  0.        ,  0.        ,  1.        ]],
    #                     dtype=np.float64)


    # Pixset: (Pixell)
    #(9b)
    # T_calib = np.array([[ 0.9994204 ,  0.00716964,  0.03327836,  3.8925061 ],
    #                     [-0.00734876,  0.9999592 ,  0.00526327, -0.02603778],
    #                     [-0.03323926, -0.00550477,  0.99943227,  0.03098693],
    #                     [ 0.        ,  0.        ,  0.        ,  1.        ]],
    #                     dtype=np.float64)

    #(9a)
    # T_calib = np.array([[ 0.99961686,  0.00714235,  0.02673987,  3.9239588 ],
    #                     [-0.00727393,  0.9999619 ,  0.00482637, -0.02135073],
    #                     [-0.02670437, -0.00501904,  0.99963075,  0.0628574 ],
    #                     [ 0.        ,  0.        ,  0.        ,  1.        ]],
    #                     dtype=np.float64)

    # #best:
    T_calib = np.array([[ 9.9993527e-01,  1.1282927e-02,  1.4698396e-03,  3.9903121e+00],
                        [-1.1287323e-02,  9.9993175e-01,  3.0175715e-03, -2.2937423e-02],
                        [-1.4356882e-03, -3.0339684e-03,  9.9999434e-01,  7.0996813e-02],
                        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
                        dtype=np.float64)


    print('arg.ignore_calibration_file {}'.format(arg.ignore_calibration_file))
    if arg.ignore_calibration_file is False:
        if arg.calibration_folder:
            calibration_file = os.path.join(arg.calibration_folder, 'sbgekinox_bcc-{}.pkl'.format(arg.sensor))
        else:
            calibration_file = 'sbgekinox_bcc-{}.pkl'.format(arg.sensor)

        assert os.path.exists(calibration_file), 'calibration file {} does not exit'.format(calibration_file)
        assert os.path.isfile(calibration_file), '{} is not a file'.format(calibration_file)
        print('Load calibration file, {}'.format(calibration_file))
        with open(calibration_file, 'rb') as f:
            T_calib = pickle.load(f)
            print(T_calib)
            T_calib = das.imu.inverse_transform(T_calib)
    vlp16_to_leddar_tr = {}
    if arg.fov_limitation:
        if arg.calibration_folder:
            calibration_file = os.path.join(arg.calibration_folder, 'sbgekinox_bcc-{}.pkl'.format(arg.fov_limitation))
        else:
            calibration_file = 'sbgekinox_bcc-{}.pkl'.format(arg.fov_limitation)
        assert os.path.exists(calibration_file), 'calibration file {} does not exit'.format(calibration_file)
        assert os.path.isfile(calibration_file), '{} is not a file'.format(calibration_file)           
        with open(calibration_file, 'rb') as f:
            Tr_leddar_to_imu = pickle.load(f)
            vlp16_to_leddar_tr[arg.fov_limitation] = T_calib @ das.imu.inverse_transform(Tr_leddar_to_imu)


    velo_map = GenerateVeloMap(arg.dataset, T_calib, arg.distance, arg.sensor, arg.fov_limitation, vlp16_to_leddar_tr)
    map_filename = os.path.split(arg.dataset)[1]
    if os.path.split(arg.dataset)[1] == '':
        map_filename = os.path.split(os.path.split(arg.dataset)[0])[1]

    velo_map.generate(map_filename)


    # plt.figure()
    # plt.plot(np.rad2deg(velo_map.imu_accuracy['roll_acc']), label='roll_acc' )
    # plt.plot(np.rad2deg(velo_map.imu_accuracy['pitch_acc']), label='pitch_acc' )
    # plt.plot(np.rad2deg(velo_map.imu_accuracy['yaw_acc']), label='yaw_acc' )
    # plt.legend()
    # plt.figure()
    # plt.plot(velo_map.imu_accuracy['latitude_acc'], label='latitude_acc' )
    # plt.plot(velo_map.imu_accuracy['longitude_acc'], label='longitude_acc' )
    # plt.plot(velo_map.imu_accuracy['altitude_acc'], label='altitude_acc' )
    # plt.legend()
    # plt.show()