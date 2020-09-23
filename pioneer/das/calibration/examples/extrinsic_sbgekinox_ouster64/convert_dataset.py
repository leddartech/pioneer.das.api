from pioneer.das.api import platform, interpolators

import numpy as np
import yaml

import das.imu
import utm
import math 
import time
import transforms3d
import argparse

#from das.utils import fit_plane_svd, distance_to_plane
#from yav import viewer, amplitudes_to_color
import pickle
import torch
import random
import tqdm

    
class SensorLidar():

    '''sensor_name = eagle_tfc, vlp16_tfc, ouster64_bfc, pixell_bfc
    '''
    def __init__(self, 
                    sensor_name, 
                    platform, 
                    position_offset, 
                    nb_point_per_interpolation_traj=100
                ):
        self.sensor_name = sensor_name
        self.nb_point_per_interpolation_traj = nb_point_per_interpolation_traj
        self.sensor_type = sensor_name.split('_')[0]
        
        self.pf = platform
        self.position_offset = position_offset
        if self.sensor_type=='eagle':
            self.sensor = self.pf['{}_ech'.format(self.sensor_name)]
            self.sensor.sensor.config['reject_flags'] = []
        elif self.sensor_type=='vlp16':
            self.sensor = self.pf['{}_xyzit'.format(self.sensor_name)]
        elif self.sensor_type=='ouster64':
            self.sensor = self.pf['{}_xyzit'.format(self.sensor_name)]
        elif self.sensor_type=='lca2':
            self.sensor = self.pf['{}_ech'.format(self.sensor_name)]
            self.sensor.sensor.config['reject_flags'] = []
        elif self.sensor_type=='pixell':
            self.sensor = self.pf['{}_ech'.format(self.sensor_name)]
            self.sensor.sensor.config['reject_flags'] = []
        else:
            assert('unknown sensor:{}'.format(self.sensor_name))

        self.nav = self.pf['sbgekinox_bcc_navposvel']
        self.euler = self.pf['sbgekinox_bcc_ekfeuler']
    
    def get_point_cloud_ts_and_intensity(self, idx:int,
                                            threshold_distance:float=1.5):

        '''returns homogeneous points, timestamps and amplitudes
        '''
        
        data = self.sensor[idx]

        if self.sensor_type=='eagle' or self.sensor_type=='lca2':
            data._mask = None
            keep = data.raw['data']['flags'] == 1
            pts = data.point_cloud()[keep]
            pts = np.hstack([pts,  np.ones_like(pts[:,0].reshape(-1, 1))]).astype('f8')
            ts = self.get_frame_timestamp(idx) + data.raw['data']['timestamps'][keep] * self.sensor[idx].raw['timestamps_to_us_coeff']
            intensity = data.raw['data']['amplitudes'][keep]
            
        elif self.sensor_type=='pixell':
            # data._mask = None
            # keep = np.mod(data.raw['data']['flags'], 2).astype(bool)
            pts = data.point_cloud()##[keep]
            pts = np.hstack([pts,  np.ones_like(pts[:,0].reshape(-1, 1))]).astype('f8')
            ts = data.timestamps#[keep]
            intensity = data.amplitudes#raw['amplitudes'][keep]

        elif self.sensor_type=='vlp16' or self.sensor_type=='ouster64' :
            pts = np.concatenate([data.raw['x'].reshape(-1, 1),
                                    data.raw['y'].reshape(-1, 1),
                                    data.raw['z'].reshape(-1, 1),
                                    np.ones_like(data.raw['z'].reshape(-1, 1)),], axis=1).astype('f8')
            ts = data.raw['t']
            intensity = data.raw['i']
        
        else:
            assert('unknown sensor:{}'.format(self.sensor_name))
        
        keep_distances = np.linalg.norm(pts[:,[0,2]], axis = 1) > threshold_distance
        return pts[keep_distances,:].T, ts[keep_distances], intensity[keep_distances]

    def get_frame_timestamp(self, idx):
        return self.sensor[idx].timestamp

    def get_pts_timestamps(self, idx):
        if self.sensor_type=='eagle' or self.sensor_type=='lca2':
            return self.get_frame_timestamp(idx) + self.sensor[idx].raw['data']['timestamps'] * self.sensor[idx].raw['timstamps_to_us_coeff']
        if self.sensor_type=='pixell':
            return self.sensor[idx].raw['data']['timestamps']
        elif self.sensor_type=='vlp16':
            return self.sensor[idx].raw['t']
        else:
            assert('unknown sensor:{}'.format(self.sensor_name))

    def print_nb_frame(self):
        print(f'nb frame {self.sensor_name}:{len(self)}')

    def trajectory(self, pts_ts, position_tolerance=0.04):
        start = time.time()
        get_at_timestamp_time = []
        get_at_timestamp_time2 = []
        
        if pts_ts.shape[0]%self.nb_point_per_interpolation_traj == 0:
            end = int(pts_ts.shape[0]/self.nb_point_per_interpolation_traj)
        else:
            end = int(pts_ts.shape[0]/self.nb_point_per_interpolation_traj)+1
        traj = np.zeros((end,4,4), dtype = np.dtype('f8'))
        # on va en conserver que self.nb_point_per_interpolation_traj par scan
        
        for i in range(end):
            ts = pts_ts[i*self.nb_point_per_interpolation_traj]
            #start2 = time.time()
            nav = self.nav.get_at_timestamp(ts, interpolator=interpolators.euler_imu_linear_ndarray_interpolator).raw # latitude        longitude        altitude
            euler = self.euler.get_at_timestamp(ts, interpolator=interpolators.euler_imu_linear_ndarray_interpolator).raw # rool  pitch  yaw
            #get_at_timestamp_time.append(time.time()-start2)
            #print([nav['latitude'],nav['longitude'],euler['yaw']])
            if nav['latitude_acc']>position_tolerance or nav['longitude_acc']>position_tolerance or nav['altitude_acc']>position_tolerance:
                return None
            
            imu = das.imu.get_imu_pose(nav, euler)
            
            pose = das.imu.imu_to_pose(imu)
            pose[:3] = pose[:3] - self.position_offset
            #start2 = time.time()
            Tr = das.imu.pose_to_transf(pose)
            #get_at_timestamp_time2.append(time.time()-start2)
            traj[i] = Tr
        #t_get_timestamp = np.array(get_at_timestamp_time)
        #t_get_timestamp2 = np.array(get_at_timestamp_time2)
        #print('trajectory :{}, get_at_timestamp:{}, struct array:{}'.format(time.time()-start,np.sum(t_get_timestamp), np.sum(t_get_timestamp2)))
        return traj

    def __len__(self):
        return len(self.sensor)

    def get_undistord_point_cloud(self, pts_raw, Tr_list, t_calib):
        pts_corrected = []

        for i in range(Tr_list.shape[0]):
            Tr = Tr_list[i]
            pt = pts_raw[:, i * self.nb_point_per_interpolation_traj: min((i+1)*self.nb_point_per_interpolation_traj, pts_raw.shape[1])]
            if(pt.shape[1] != 0):
                pts_corrected.append(Tr.dot(t_calib.dot(pt)))
            

        assert len(pts_corrected) != 0, 'No point in this frame'
        pts = np.hstack(pts_corrected)
        return pts
    
    def vlp16_fov_limitation_to_leddar(self, pts_in, leddar_sensor_name, Tr_vlp_to_leddar):

        '''return a boolean vector, true if the vlp16 point is in the fov of the leddar
        '''
        leddar_sensor_type = leddar_sensor_name.split('_')[0]
        
        if leddar_sensor_type in ['eagle', 'lca2']:
            leddar_fov = np.array([self.pf[leddar_sensor_name].specs['h_fov'],self.pf[leddar_sensor_name].specs['v_fov']])
            if leddar_sensor_type == 'lca2':
                leddar_oriented_fov = np.abs(np.eye(2) @ leddar_fov)
            else:
                leddar_oriented_fov = np.abs(self.pf[leddar_sensor_name].orientation[:2,:2] @ leddar_fov)
        
        
        pf_in_leddar = Tr_vlp_to_leddar @ pts_in
        angles0, angles1 = self.pts_to_angles_leddar(pf_in_leddar)
        return (np.abs(angles0) < leddar_oriented_fov[0]/2) & (np.abs(angles1) < leddar_oriented_fov[1]/2)

    def pts_to_angles_vlp16(self, pts):
        x = pts[0,:]
        y = pts[1,:]
        z = pts[2,:]
        d = np.linalg.norm(pts[0:2, :], axis = 0)

        return(np.rad2deg(np.arctan2(y, x)), np.rad2deg(np.arctan2(z, d)))
    
    def pts_to_angles_leddar(self, pts):
        x = pts[0,:]
        y = pts[1,:]
        z = pts[2,:]
        d = np.linalg.norm(pts[[0,2], :], axis = 0)

        return(np.rad2deg(np.arctan2(x, z)), np.rad2deg(np.arctan2(-y, d)))


class DatasetExportation():
    def __init__(self, platform_path, scene_file, sensor_name):
        self.sensor_type = sensor_name.split('_')[0]
        self.sensor_name = sensor_name
        self.pf = platform.Platform(platform_path)
       
        self.nav = self.pf['sbgekinox_bcc_navposvel']
        self.euler = self.pf['sbgekinox_bcc_ekfeuler']
        
        print('nb frame nav:{}'.format(len(self.nav)))
        print('nb frame euler:{}'.format(len(self.euler)))

        with open(scene_file) as f:
            self.scene = yaml.load(f)   

        self.position_offset = np.array(self.scene['planes'][0]['center'])
        print('position offset:', self.position_offset)
        for i in range(len(self.scene['planes'])):
            self.scene['planes'][i]['center'] = self.scene['planes'][i]['center'] - self.position_offset
        
        if self.sensor_type == 'lca2':
            self.sensor_lidar = SensorLidar(self.sensor_name, self.pf, self.position_offset, nb_point_per_interpolation_traj = 16)
        elif self.sensor_type == 'eagle':
            self.sensor_lidar = SensorLidar(self.sensor_name, self.pf, self.position_offset, nb_point_per_interpolation_traj = 16)
        elif self.sensor_type =='pixell':
            self.sensor_lidar = SensorLidar(self.sensor_name, self.pf, self.position_offset, nb_point_per_interpolation_traj = 16)
        elif self.sensor_type == 'vlp16':
            self.sensor_lidar = SensorLidar(self.sensor_name, self.pf, self.position_offset)
        elif self.sensor_type == 'ouster64':
            self.sensor_lidar = SensorLidar(self.sensor_name, self.pf, self.position_offset)
        else:
            assert('unknown sensor:{}'.format(self.sensor_name))

        self.sensor_lidar.print_nb_frame()
        self.last_pose_planes = np.zeros(6)
        self.scans_planes = []

    # return horizontal and vertical angle
    def get_angles(self, pt):
        x = pt[0]
        y = pt[1]
        z = pt[2]
        d = np.linalg.norm([x,y,z])
        return(np.rad2deg(np.pi/2 - np.arctan2(z,x)), np.rad2deg(np.pi/2 - np.arccos(y/d)))

    def keep_scan_leddar(self, current_pose, object_list):
        angle_coef = 1.5 # if fov 20 deg, selection in 30 deg 
        TrImu = das.imu.pose_to_transf(current_pose)
        TrImu_inv = das.imu.inverse_transform(TrImu)
        T_calib_inv = das.imu.inverse_transform(self.T_calib_0_Tr)
       
        for i, obj_all in enumerate(object_list):
            obj=obj_all['center'].copy()
            #obj.append(1.0)
            obj = np.hstack((obj, 1.0))
            center_pt = np.array(obj).reshape((4,1))
            # plane center from global frame to leddar frame
            pt_in_leddar = T_calib_inv @ TrImu_inv @ center_pt
            # if the point is behind le leddar, useless to follow for this iteration
            if pt_in_leddar[2] < 0:
                #print(i, pt_in_leddar.T)
                continue
            angles  = self.get_angles(pt_in_leddar)
            #print(i, pt_in_leddar.T, angles)
            if(np.abs(angles[0]) < self.oriented_fov[0]*angle_coef/2) and (np.abs(angles[1]) < self.oriented_fov[1]*angle_coef/2):
                return True
        
        return False

    def keep_scan(self, current_pose, object_list):
        if self.sensor_type=='eagle' or self.sensor_type=='lca2':
            return self.keep_scan_leddar(current_pose, object_list)
        # pour le eagle
        distance_max_to_object = 60.0
        distance_max_between_pose = 1.0
        angle_max_between_pose = np.deg2rad(5.0)

        keep_distance = False
        for obj_all in object_list:
            obj=obj_all['center']
            dist = np.linalg.norm(np.array(obj[0:2])-current_pose[0:2])
            if dist < distance_max_to_object:
                keep_distance = True
        
        if not keep_distance:
            return False
        
        diff = current_pose - self.last_pose_planes

        distance = np.linalg.norm(diff[0:3])
        angle_diff = np.abs(diff[5])


        if(distance > distance_max_between_pose or angle_diff > angle_max_between_pose):
            self.last_pose_planes = current_pose
            return True
        else:
            return False
    
    def get_points_in_object(self, scan, T_calib, Tr_list, object_list, X, threshold_factor=1.0):
        pts = self.sensor_lidar.get_undistord_point_cloud(scan, Tr_list, T_calib)
        # np.savetxt('scan_corrected.xyz', pts[:3,:].T)
        # pts_no_corrected.append(Tr_list[0].dot(T_calib.dot(scan)))
        # pts = np.hstack(pts_no_corrected)
        # np.savetxt('scan_no_corrected.xyz', pts[:3,:].T)

        pt_list = []
        for obj_all in object_list:
            obj = obj_all['center']
            threshold = threshold_factor * obj_all['size']/2.0
            k = (pts[0,:] > obj[0]-threshold) & (pts[0,:] < obj[0]+threshold) & (pts[1,:] > obj[1]-threshold) & (pts[1,:] < obj[1]+threshold) & (pts[2,:] > obj[2]-threshold) & (pts[2,:] < obj[2]+threshold)
            pt_list.append(pts[0:3,k])

        return pt_list #pts[0:3,keep]
    
    def best_std(self, distances, percent):
        idx = np.argsort(np.abs(distances))
        idx_keep = idx[:int(idx.shape[0]*percent)]
        distances = distances[idx_keep]
        return distances

    def select_scans(self, _range=None):
        
        '''optional: _range = range(200, 1500, 2)
        '''
        #nb_pts_min_to_keep_scan = 5
        if _range is None:
            _range = range(len(self.sensor_lidar))

        for i in _range:
            #start = time.time()
            print('process scan:{}/{}'.format(i, len(self.sensor_lidar)))
            lidar_frame, pts_ts, lidar_intensity = self.sensor_lidar.get_point_cloud_ts_and_intensity(i)
            lidar_ts = self.sensor_lidar.get_frame_timestamp(i)

            nav = self.nav.get_at_timestamp(lidar_ts, interpolator=interpolators.euler_imu_linear_ndarray_interpolator).raw # latitude        longitude        altitude
            euler = self.euler.get_at_timestamp(lidar_ts, interpolator=interpolators.euler_imu_linear_ndarray_interpolator).raw # rool  pitch  yaw
            #print([nav['latitude'],nav['longitude'],euler['yaw']])
            imu = das.imu.get_imu_pose(nav, euler)
            pose = das.imu.imu_to_pose(imu)
            pose[:3] = pose[:3] - self.position_offset
            
            Tr = None
            if self.keep_scan(pose, self.scene['planes']):
                Tr = das.imu.pose_to_transf(pose)
                pts = lidar_frame


                traj = self.sensor_lidar.trajectory(pts_ts)
                if traj is not None:
                    self.scans_planes.append((lidar_frame, Tr, traj, lidar_intensity))
                    print('keeped')
                else:
                    print('imu acc rejection')


            #print('select_scans :{}'.format(time.time()-start))

        print('plane scans:{}'.format(len(self.scans_planes)))
        
        d = {}
        d['scans_planes'] = self.scans_planes
        d['position_offset'] = self.position_offset
        d['scene'] = self.scene
        d['nb_point_per_interpolation_traj'] = self.sensor_lidar.nb_point_per_interpolation_traj
        with open('exported_dataset.pkl', 'wb') as f:
            pickle.dump(d, f) #May cause Memory Error if size of d is too large


    def get_points_in_object(self, scan, T_calib, Tr_list, object_list, X, threshold_factor=1.0):
        pts = self.sensor_lidar.get_undistord_point_cloud(scan, Tr_list, T_calib)

        pt_list = []
        for obj_all in object_list:
            obj = obj_all['center']
            threshold = threshold_factor * obj_all['size']/2.0
            k = (pts[0,:] > obj[0]-threshold) & (pts[0,:] < obj[0]+threshold) & (pts[1,:] > obj[1]-threshold) & (pts[1,:] < obj[1]+threshold) & (pts[2,:] > obj[2]-threshold) & (pts[2,:] < obj[2]+threshold)
            pt_list.append(pts[0:3,k])
            

        return pt_list #pts[0:3,keep]

    def load_from_pickle_file(self, f_name='exported_dataset.pkl'):
        print(f'loading package from file: {f_name}')
        with open(f_name, 'rb') as f:
            d = pickle.load(f)
            self.scans_planes = d['scans_planes']
            self.position_offset = d['position_offset']
            self.scene = d['scene']
            self.sensor_lidar.nb_point_per_interpolation_traj = d['nb_point_per_interpolation_traj']
    
    def __len__(self):
        return len(self.scene['planes'])

    def get_points(self, index, T_calib):
        point_keeped = []
        tr_keeped = []
        
        # pour tout les scans, calculer ce qui correspond au plan index
        for s in self.scans_planes:
            pts_raw = s[0]
            tr_list = s[2]
            pts = self.sensor_lidar.get_undistord_point_cloud(pts_raw, tr_list, T_calib)
            obj = self.scene['planes'][index]['center']
            threshold = self.scene['planes'][index]['size']/2.0
            k = (pts[0,:] > obj[0]-threshold) & (pts[0,:] < obj[0]+threshold) & (pts[1,:] > obj[1]-threshold) & (pts[1,:] < obj[1]+threshold) & (pts[2,:] > obj[2]-threshold) & (pts[2,:] < obj[2]+threshold)
            if np.any(k):
                tr_list = np.repeat(tr_list, self.sensor_lidar.nb_point_per_interpolation_traj, axis=0)
                tr_list = tr_list[:pts_raw.shape[1], :, :]
                #print(tr_list.shape, pts.shape)
                point_keeped.append(pts_raw[0:3,k])
                tr_keeped.append(tr_list[k, :, :])
            
        return np.concatenate(point_keeped, axis = 1).T, np.concatenate(tr_keeped, axis = 0)
            #res = self.get_points_in_object(s[0],T_calib,s[2], self.scene['planes'], X, threshold_factor)


class PlaneCalibDataset(DatasetExportation):
    def __init__(self,
                    dataset='/nas/pixset/exportedDataset/20200519_201503_rec_dataset_calib_imu_brick02_exported',
                    planes_cfg='planes_ouster_20200519_201503.yml',
                    sensor='ouster64_bfc',
                    exported_package='exported_dataset.pkl',
                    calibration_matrix=np.eye(4)):
        super(PlaneCalibDataset, self).__init__(dataset, planes_cfg, sensor)

        self.load_from_pickle_file(exported_package)

        #calibration matrix: tf (4x4) from IMU to sensor
        self.calibration_matrix = calibration_matrix

    def get_calibration_matrix(self):
        return self._calibration_matrix
    
    def set_calibration_matrix(self, tf:np.ndarray):
        self._calibration_matrix = np.copy(tf)

    calibration_matrix = property(get_calibration_matrix, set_calibration_matrix)

    def __getitem__(self, mu):
        
        '''returns the mu-th plane points, using the current calibration matrix
        '''
        return self.get_points(index=mu, T_calib=self.calibration_matrix)

class PlaneCalibDataloader():
    def __init__(self, dataset:PlaneCalibDataset,
                    num_max_points:int,
                    batch_size:int,
                    device:torch.device,
                    shuffle:bool
                ):
        self.dataset = dataset
        self.num_max_points = num_max_points
        self.device = device
        self.batch_size = batch_size
        self.shuffle = shuffle

    def update_calibration_matrix(self, tf:np.ndarray):
        self.dataset.set_calibration_matrix(tf)
    
    def __iter__(self):
        l = len(self.dataset)
        self.__current = 0
        self.__batch_sequence = np.random.permutation(l) if self.shuffle else np.arange(l)
        return self

    def __next__(self):
        if self.__current >= len(self.dataset):
            raise StopIteration
        else:
            self.__current += self.batch_size
            ids_curren_batch = np.array(self.__batch_sequence[self.__current-self.batch_size: min(self.__current, len(self.dataset))])
            batched_pts = []
            bathced_tr = []
            
            for mu in ids_curren_batch:
                pts_, tr_ = self.dataset[mu]
                m_ = len(pts_)
                replace = False if m_>self.num_max_points else True
                ids = np.random.choice(m_, self.num_max_points, replace)
                batched_pts.append(pts_[ids])
                bathced_tr.append(tr_[ids])
            
            batched_pts = torch.from_numpy(np.stack(batched_pts, axis=0)).float()
            bathced_tr = torch.from_numpy(np.stack(bathced_tr,axis=0)).float()

            return batched_pts.to(self.device), bathced_tr.to(self.device)


if __name__ == '__main__':
    # example : 
    # python3 convert_dataset.py /nas/pixset/exportedDataset/20200608_180144_rec_dataset_calib_imu_fast_brick05_exported ./planes_ouster_20200519_201503.yml 
    #           ouster64_bfc --export_init --range 100 2200 2
    #
    #
    parser = argparse.ArgumentParser(description='IMU Lidar (Ouster64/Pixell) calibration dataset exportation.')
    parser.add_argument('dataset', help='dataset platform folder')
    parser.add_argument('planes', help='plane constraints yaml file description')
    parser.add_argument('sensor', help='sensor name to calibrate, ouster64_bfc for example')
    parser.add_argument('--export_init', action='store_true', help='At first, the dataset must be pre-pocessed and exported using this command')
    parser.add_argument('--range', default=None, nargs='+', type=int, help='--range start stop step')

    arg = parser.parse_args()
    
    exportation_dataset = DatasetExportation(arg.dataset, arg.planes, arg.sensor)
   
    if arg.export_init:
        _range = arg.range
        if _range is not None:
            _range = range(_range[0], _range[1], _range[2])
        exportation_dataset.select_scans(_range)
        exit()
    else:#evaluation part
        with open(arg.dataset+f'/extrinsics/sbgekinox_bcc-{arg.sensor}.pkl', 'rb') as k:
            init_matrix = das.imu.inverse_transform(pickle.load(k))
        exportation_dataset.load_from_pickle_file(f_name='exported_dataset.pkl')
        print('initial matrix:', init_matrix)
        print(f"len object:{len(exportation_dataset)}")
        print(f"nb_point_per_interpolation_traj:{exportation_dataset.sensor_lidar.nb_point_per_interpolation_traj}")
        
        for i in range(len(exportation_dataset)):
            start = time.time()
            pts, tr = exportation_dataset.get_points(i, init_matrix)
            
            pts_h = np.vstack([pts.T,np.ones((1,pts.shape[0]))])
            pts_imu = init_matrix @ pts_h
            pts_plan = np.zeros_like(pts_imu)
            for j in range(pts.shape[0]):
                pts_plan[:,j] = tr[j,:,:] @ pts_imu[:,j]

            # pts_du_plan = tr @ T_calib @ pts
            np.savetxt(f"plane_point{i}.xyz", pts_plan.T[:,:3], fmt='%.3f')
            stop = time.time()
            print(pts.shape, tr.shape, stop-start)