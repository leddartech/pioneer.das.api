import argparse
import numpy as np
import yaml
from das.api import platform, interpolators
import numpy as np
import math
import matplotlib.pyplot as plt
import collections 
import time
import das.imu
from yav import amplitudes_to_color
from planes_calib import SensorLidar
import os
import pickle

from yav.viewer import *
import time

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

    def get_position_offset(self):
        nav = self.nav[0].raw # latitude        longitude        altitude
        euler = self.euler[0].raw # rool  pitch  yaw
        imu = das.imu.get_imu_pose(nav, euler)
        pose = das.imu.imu_to_pose(imu)
        return pose[:3]

    def __init__(self, platform_path, calibration, min_distance, sensor_name, fov_limitation, T_vlp16_to_leddar):
        self.pts_map = collections.deque()
        self.sensor_name = sensor_name
        self.sensor_type = sensor_name.split('_')[0]
        self.t_calib = calibration
        self.min_distance = min_distance
        self.pf = platform.Platform(platform_path)


        self.fov_limitation = fov_limitation
        self.T_vlp16_to_leddar = T_vlp16_to_leddar
        if self.fov_limitation:
            assert self.sensor_type == 'vlp16', 'fov_limitation option is usable only from vlp16 sensor'
            assert self.fov_limitation in self.pf.sensors.keys(), '{} is not in platform sensors list'.format(self.fov_limitation)
            assert self.fov_limitation in self.T_vlp16_to_leddar.keys(), 'No calibration for vlp16 to {}'.format(self.fov_limitation)

        if self.sensor_type == 'lca2':
            self.sensor_lidar = SensorLidar(self.sensor_name, self.pf, np.array([0,0,0]), nb_point_per_interpolation_traj = 16)
        elif self.sensor_type == 'eagle':
            self.sensor_lidar = SensorLidar(self.sensor_name, self.pf, np.array([0,0,0]), nb_point_per_interpolation_traj = 16)
        elif self.sensor_type == 'vlp16':
            self.sensor_lidar = SensorLidar(self.sensor_name, self.pf, np.array([0,0,0]))
        elif self.sensor_type == 'ouster64':
            self.sensor_lidar = SensorLidar(self.sensor_name, self.pf, np.array([0,0,0]))
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
        #self.pose_offset = np.zeros(6)
        self.pose_offset = self.get_position_offset()


    def display_points_map(self, current_scan):
        if len(self.pts_map) != 0:
            map = np.vstack(self.pts_map)
            colors = amplitudes_to_color(map[:,3].astype('f4'))
            self.pc.set_points(map[:, 0:3], colors)

    def generate(self, frame):
        file_number = 0
            
        #vlp_frame = self.vlp[frame].raw
        ts =  self.sensor_lidar.get_frame_timestamp(frame)
        nav = self.nav.get_at_timestamp(ts, interpolator=interpolators.euler_imu_linear_ndarray_interpolator).raw # latitude        longitude        altitude
        euler = self.euler.get_at_timestamp(ts, interpolator=interpolators.euler_imu_linear_ndarray_interpolator).raw # rool  pitch  yaw
        #get_at_timestamp_time.append(time.time()-start2)
        #print([nav['latitude'],nav['longitude'],euler['yaw']])
        #print('euler {}'.format(euler.dtype))
        imu = das.imu.get_imu_pose(nav, euler)
        
        pose = das.imu.imu_to_pose(imu)
        
        print('frames {}/{} keeped'.format(frame, len(self.sensor_lidar)))

        pts, pts_ts, intensity = self.sensor_lidar.get_point_cloud_ts_and_intensity(frame)
        
        #print('pts.shape {}'.format(pts.shape))
        #traj = self.sensor_lidar.trajectory(pts_ts)
        ts =  self.sensor_lidar.get_frame_timestamp(frame)
        
        # pour moi ca sert à rien ca fait la meme chose dans les deux cas !
        # if self.sensor_type:
        #     distances = np.linalg.norm(pts.T[:,[0,2]], axis = 1)
        # else:
        #     distances = np.linalg.norm(pts.T[:,:2], axis = 1)
        distances = np.linalg.norm(pts.T[:,[0,2]], axis = 1)

        pts = pts[:,(distances>1.0) & (distances<40.0)]
        pts_ts = pts_ts[(distances>1.0) & (distances<40.0)]
        intensity = intensity[(distances>1.0) & (distances<40.0)]

        traj = self.sensor_lidar.trajectory(pts_ts)
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
        
        pts_keep[:,:3] = pts_keep[:,:3] - self.pose_offset

        return pts_keep
            #self.pts_map.append(pts_keep)

def on_frame_changed(viewer):

    i = viewer.get_frame()
    pcl = velo_map.generate(i)
    pcl2 = velo_map.generate((i+100)%n)

    colors =  np.zeros((pcl.shape[0],3)).astype('u1')
    colors[:,:] = [255,0,0]
    colors_ =  np.zeros((pcl2.shape[0],3)).astype('u1')
    colors_[:,:] = [0,255,0]
    pcd1.set_points(pcl[:,:3],colors=colors)#.astype('u1'))
    pcd11.set_points(pcl2[:,:3],colors=colors_)#.astype('u1'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IMU VLP16 map generation, the scan correction is integrated in the script.')
    parser.add_argument('dataset', help='dataset platform folder')
    parser.add_argument('sensor', help='sensor name to map, example vlp16_tfc')
    parser.add_argument('--distance', type=float, default = 10.0, help='minimum distance between two keeped scans')
    parser.add_argument('--fov_limitation', help='limite vlp16 fov to leddar name')
    parser.add_argument('--calibration_folder', help='folder with the pickle calibration files (sbgekinox_bcc-sensor_name.pkl)')
    parser.add_argument('--ignore_calibration_file', action='store_true', default=False, help='ignore the pickle calibration, and use the coded matrix in the script')
    #choices=[range(0.0, 50.0)]
    arg = parser.parse_args()

    # avec planes replacés sur dataset /data/exportedDataset0/20190613_163434_rec_dataset_calib_LCA2_imu2_exported
    T_calib_0_pose = np.array([ 3.87043055e+00, -7.00000000e-01, 1.19393130e-02, -8.83438181e+01, -1.36076383e+00, -1.20018361e+02])
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

    vlp16_to_leddar_tr = {}
    velo_map = GenerateVeloMap(arg.dataset, T_calib, arg.distance, arg.sensor, arg.fov_limitation, vlp16_to_leddar_tr)
    map_filename = os.path.split(arg.dataset)[1]

    if os.path.split(arg.dataset)[1] == '':
        map_filename = os.path.split(os.path.split(arg.dataset)[0])[1]

    #velo_map.generate(map_filename)

    n = len(velo_map.sensor_lidar)
    i = 0
    v = viewer(num=n)
    v.set_title('My Main Window')

    # create a second point cloud window
    pcd1 = v.create_point_cloud()
    pcd11 = v.create_point_cloud()

    v.add_frame_callback(on_frame_changed)
    v.set_frame(i) 

    v.run()