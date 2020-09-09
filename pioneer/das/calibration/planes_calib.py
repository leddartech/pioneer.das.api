import numpy as np
import yaml
from das.api import platform, interpolators
import utm
import math 
import time
import transforms3d
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import argparse
import das.imu
from das.utils import fit_plane_svd, distance_to_plane
from yav import viewer, amplitudes_to_color
import pickle

class ViewerEagleScanSelection():
    def __init__(self, plane_calib, position_offset):
        self.position_offset = position_offset
        self.plane_calib = plane_calib
        self.sensor_lidar = self.plane_calib.sensor_lidar
        self.v = viewer(num=len(self.sensor_lidar), axes='widget', camera=False)
        self.pc_map = self.v.create_point_cloud()
        self.pc_planes = self.v.create_point_cloud()
        self.v.add_frame_callback(self.update_viewer)
        self.add_planes(self.plane_calib.scene['planes'])
        self.axes_vehicle = self.v.create_axes()
        self.axes_leddar = self.v.create_axes()
        self.frustrum = self.v.create_frustrum_fov(self.plane_calib.oriented_fov[0],self.plane_calib.oriented_fov[1],50, np.array([0,0,0,0,0,0]), frustrum_orientation='eagle')
        first_ts = self.sensor_lidar.get_frame_timestamp(0)
        nav = self.plane_calib.nav.get_at_timestamp(first_ts).raw # latitude        longitude        altitude
        euler = self.plane_calib.euler.get_at_timestamp(first_ts).raw # rool  pitch  yaw
        imu = das.imu.get_imu_pose(nav, euler)
        pose = das.imu.imu_to_pose(imu)
        pose[:3] = pose[:3] - self.position_offset
        #self.position_offset  = pose[:3]
        self.v.run()


    def update_viewer(self, v):
        frame_idx = v.get_frame()
        
        pts, pts_ts, intensity = self.sensor_lidar.get_point_cloud_ts_and_intensity(frame_idx)
        traj = self.sensor_lidar.trajectory(pts_ts)
        pts_global = self.sensor_lidar.get_undistord_point_cloud(pts, traj, self.plane_calib.T_calib_0_Tr)
        ts = self.sensor_lidar.get_frame_timestamp(frame_idx)
        nav = self.plane_calib.nav.get_at_timestamp(ts).raw # latitude        longitude        altitude
        euler = self.plane_calib.euler.get_at_timestamp(ts).raw # rool  pitch  yaw

        
        imu = das.imu.get_imu_pose(nav, euler)
        
        pose = das.imu.imu_to_pose(imu)
        pose[:3] = pose[:3] - self.position_offset

        colors = amplitudes_to_color(intensity.astype('f8'))
        self.pc_map.set_points(pts_global.T[:, 0:3] , colors)


        m = das.imu.pose_to_transf(pose)
        self.axes_vehicle.set_matrix(m)

        self.frustrum.set_matrix(m @ self.plane_calib.T_calib_0_Tr)
        self.axes_leddar.set_matrix(m @ self.plane_calib.T_calib_0_Tr)
        print('keep scan {}:{}'.format(frame_idx, self.plane_calib.keep_scan_eagle(pose, self.plane_calib.scene['planes'])))


    def add_planes(self, planes):
        for i, obj_all in enumerate(planes):
            center=np.array(obj_all['center'])
            self.v.create_sphere(center=center, radius = 0.5, color = np.array([1.0, 0.0, 0.0]))
            self.v.create_caption2d(text = str(i), position=center)
        

    
class SensorLidar(): # sensor eagle_tfc, vlp16_tfc, lca2_xxxx
    def __init__(self, sensor_name, platform, position_offset, nb_point_per_interpolation_traj = 100):
        self.nb_point_per_interpolation_traj = nb_point_per_interpolation_traj
        self.sensor_type = sensor_name.split('_')[0]
        self.sensor_name = sensor_name
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
        else:
            assert('unknown sensor:{}'.format(self.sensor_name))

        self.nav = self.pf['sbgekinox_bcc_navposvel']
        self.euler = self.pf['sbgekinox_bcc_ekfeuler']
    
    def get_point_cloud_ts_and_intensity(self, idx):
        
        data = self.sensor[idx]

        if self.sensor_type=='eagle' or self.sensor_type=='lca2':
            data._mask = None
            pts = data.point_cloud()[self.sensor[idx].raw['data']['flags'] == 1,:]
            pts = np.hstack([pts,  np.ones_like(pts[:,0].reshape(-1, 1))]).astype('f8')
            ts = self.get_frame_timestamp(idx) + self.sensor[idx].raw['data']['timestamps'][self.sensor[idx].raw['data']['flags'] == 1] * self.sensor[idx].raw['timestamps_to_us_coeff']
            intensity = self.sensor[idx].raw['data']['amplitudes'][self.sensor[idx].raw['data']['flags'] == 1]
            distances = np.linalg.norm(pts[:,[0,2]], axis = 1)
        elif self.sensor_type=='vlp16' or self.sensor_type=='ouster64' :
            pts = np.concatenate([data.raw['x'].reshape(-1, 1),
                                    data.raw['y'].reshape(-1, 1),
                                    data.raw['z'].reshape(-1, 1),
                                    np.ones_like(data.raw['z'].reshape(-1, 1)),], axis=1).astype('f8')
            ts = data.raw['t']
            intensity = data.raw['i']
        
            distances = np.linalg.norm(pts[:,:2], axis = 1)
        
        else:
            assert('unknown sensor:{}'.format(self.sensor_name))

        return pts[distances > 2.0,:].T, ts[distances > 2.0], intensity[distances > 2.0]

    def get_frame_timestamp(self, idx):
        return self.sensor[idx].timestamp

    def get_pts_timestamps(self, idx):
        if self.sensor_type=='eagle' or self.sensor_type=='lca2':
            return self.get_frame_timestamp(idx) + self.sensor[idx].raw['data']['timestamps'] * self.sensor[idx].raw['timstamps_to_us_coeff']
        elif self.sensor_type=='vlp16':
            return self.sensor[idx].raw['t']
        else:
            assert('unknown sensor:{}'.format(self.sensor_name))


    def print_nb_frame(self):
        if self.sensor_type=='eagle' or self.sensor_type=='lca2':
            print('nb frame {}:{}'.format(self.sensor_name, len(self)))
        elif self.sensor_type=='vlp16' or self.sensor_type=='ouster64':
            print('nb frame {}:{}'.format(self.sensor_name, len(self)))
        else:
            assert('unknown sensor:{}'.format(self.sensor_name))


    def trajectory(self, pts_ts):
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
        return(len(self.sensor))


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
    # return a boolean vector, true if the vlp16 point if in the fov of the leddar
    def vlp16_fov_limitation_to_leddar(self, pts_in, leddar_sensor_name, Tr_vlp_to_leddar):
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

class RecordSteps():
    def __init__(self, platform_path, scene_file, sensor_name):
        self.data = {}
        self.data['dataset'] = platform_path
        self.data['scene_file'] = scene_file
        self.data['sensor_name'] = sensor_name
        self.data['x_steps'] = []

    def add_step(self,x_current):
        self.data['x_steps'].append(x_current)


    def save_file(self, ouput_file = 'steps.pkl'):
        with open(ouput_file, 'wb') as f:
            pickle.dump(self.data, f)

class PlaneCalib(): # todo corriger le eagle ici aussi
    def __init__(self, platform_path, scene_file, sensor_name):
        self.sensor_type = sensor_name.split('_')[0]
        self.sensor_name = sensor_name
        self.record_steps = RecordSteps(platform_path, scene_file, sensor_name)
        self.pf = platform.Platform(platform_path)
        if self.sensor_type in ['eagle', 'lca2']:
            self.fov = np.array([self.pf[self.sensor_name].specs['h_fov'],self.pf[self.sensor_name].specs['v_fov']])
            if self.sensor_type == 'lca2':
                self.oriented_fov = np.abs(np.eye(2) @ self.fov)
            else:
                self.oriented_fov = np.abs(self.pf[self.sensor_name].orientation[:2,:2] @ self.fov)

        
        #self.vlp = self.pf['vlp16_tfc_xyzit']
        self.nav = self.pf['sbgekinox_bcc_navposvel']
        self.euler = self.pf['sbgekinox_bcc_ekfeuler']
        #print('nb frame lidar:{}'.format(len(self.vlp)))
        
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
        elif self.sensor_type == 'vlp16':
            self.sensor_lidar = SensorLidar(self.sensor_name, self.pf, self.position_offset)
        elif self.sensor_type == 'ouster64':
            self.sensor_lidar = SensorLidar(self.sensor_name, self.pf, self.position_offset)
        else:
            assert('unknown sensor:{}'.format(self.sensor_name))

        self.sensor_lidar.print_nb_frame()
        self.last_pose_planes = np.zeros(6)
        self.scans_planes = []


        self.x0 = np.zeros(6)
        self.verbose = True


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
    
    def get_points_in_object(self, scan, T_calib, Tr_list, object_list, threshold_factor=1.0):
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

    def eval_cost(self, X, output_planes_parameter = False ):
        output_param = []
        
        print('Eval x:{}'.format(X))
        T_calib = das.imu.x_to_mat(X)
        
        nb_pts = 0
        pts_planes = []
        for i in range(len(self.scene['planes'])):
            pts_planes.append([])


        

        
        threshold_factor = 1.0
        for s in self.scans_planes:
            res = self.get_points_in_object(s[0],T_calib,s[2], self.scene['planes'], threshold_factor)
            for i in range(len(self.scene['planes'])):
                pts_planes[i].append(res[i])

        for i in range(len(self.scene['planes'])):
            threshold_factor = 2.0
            pts = np.hstack(pts_planes[i])
            while (pts.T.shape[0] < 4 and threshold_factor <= 4.0):
                for s in self.scans_planes:
                    res = self.get_points_in_object(s[0],T_calib,s[2], [self.scene['planes'][i]], threshold_factor)
                    pts_planes[i].append(res[0])

                print('threshold_factor:{}'.format(threshold_factor))
                threshold_factor *= 2.0
                pts = np.hstack(pts_planes[i])
        

        distance_planes = []
      
        for i in range(len(self.scene['planes'])):
            dic_param = {}
            pts = np.hstack(pts_planes[i])
            dic_param['pts'] = pts
            np.savetxt('planes_{}.xyz'.format(i),pts.astype('f8').T)
            # compute planes
            if (pts.T.ndim != 2 or pts.T.shape[0] < 4):
                distance_planes.append(math.nan)#0.15) # TODO : plutot doubler la zone de recherche
                print('- plane:{}, dst:{}'.format([0,0,0,0], distance_planes[-1]))
            else:
                params = fit_plane_svd(pts.T, full_matrices = False)
                
                dst = distance_to_plane(pts.T, params)
                dist = self.best_std(dst,0.99)
                distance_planes.append(np.mean(dist * dist))
                print('- plane:{}, dst:{}'.format(params, np.sqrt(distance_planes[-1])))
                dic_param['planes_param'] = params
                dic_param['distance'] = np.sqrt(distance_planes[-1])
            # les inliers distances, on conserve que 

            nb_pts += pts.shape[1]
            output_param.append(dic_param)
        
        print('* Result:{} nb pts:{}'.format(np.sum(np.sqrt(distance_planes)), nb_pts))
        if (output_planes_parameter):
            return (np.sum(distance_planes), output_param)
        else:
            return (np.sum(distance_planes))

    def update_calibration(self, vlp_sensor_name, xHat):
        mechanical = np.array(self.pf.yml[vlp_sensor_name]['export']['mechanical'])
        print('Old mechanical transformation:{}'.format(mechanical))
        mechanical[3:6] = np.deg2rad(mechanical[3:6]) 
        Tr_old = das.imu.pose_to_transf(mechanical)
        Tr_new = das.imu.x_to_mat(xHat)
        Tr = Tr_new.dot(Tr_old)
        print('New transformation:\n{}'.format(Tr))
        print('solution Euler Angle (deg):{}'.format(np.rad2deg(das.imu.get_euler_angleZYX(Tr))))
        new_mechanical = np.concatenate((Tr[0:3,3], np.rad2deg(das.imu.get_euler_angleZYX(Tr))))
        print('New mechanical transformation:{}'.format(new_mechanical))
        return(new_mechanical)


  

    def select_scans(self):
        #nb_pts_min_to_keep_scan = 5
        for i in range(len(self.sensor_lidar)):#300,600):#
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
                self.scans_planes.append((lidar_frame, Tr, traj, lidar_intensity))


            #print('select_scans :{}'.format(time.time()-start))

        print('plane scans:{}'.format(len(self.scans_planes)))



    def optimize(self):
        def constraint_Z_sup(X):
            return self.x0[2] - X[2]
        def constraint_Z_inf(X):
            return X[2] - self.x0[2]
        bounds = (  (self.x0[0]-2.0, self.x0[0]+2.0),
                    (self.x0[1]-0.3, self.x0[1]+0.3),
                    (self.x0[2]-0.1, self.x0[2]+0.1),
                    (self.x0[3]-10.0/57, self.x0[3]+10.0/57),
                    (self.x0[4]-10.0/57, self.x0[4]+10.0/57),
                    (self.x0[5]-10.0/57, self.x0[5]+10.0/57),
                )
  
        cons = [{'type':'ineq', 'fun': constraint_Z_sup},
                {'type':'ineq', 'fun': constraint_Z_inf}]


        self.record_steps.add_step(self.x0)


        self.solution = minimize(self.eval_cost, self.x0,
                            method='SLSQP', options={'disp': True, 'maxiter': 200, 'ftol': 1e-08}, bounds=bounds, callback=self.record_steps.add_step) # constraints=cons) # , 'eps':1e-3 # SLSQP
        
        if(self.verbose):
            print(self.solution)

        self.record_steps.add_step(self.solution.x)
        self.record_steps.save_file()
        return(self.solution.x)

    def set_x0(self, pose):
        pose[3:6] = np.deg2rad(pose[3:6])
        self.T_calib_0_Tr = das.imu.pose_to_transf(pose)
        self.x0 = das.imu.axangle_X0_init(self.T_calib_0_Tr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IMU VLP16/Eagle calibration, with planes constraints.')
    parser.add_argument('dataset', help='dataset platform folder')
    parser.add_argument('planes', help='plane constraints yaml file description')
    parser.add_argument('sensor', help='sensor name to calibrate, vlp16_tfc for example')
    parser.add_argument('-v', '--view', action='store_true', default=False, help='view 3d point cloud and plane selection')

    arg = parser.parse_args()
    #pole_calib = PoleCalib('/data/exportedDataset0/20190306_173607_rec_dataset_VLP16_poteaux_exported/', './mapping_dev/poteaux_20190306_173607.yml')
    plane_calib = PlaneCalib(arg.dataset, arg.planes, arg.sensor)
    


    # TODO: T_calib_0_pose have to be also in argument 
    # TODO: arguments select scan indices 
    # vlp16
    T_calib_0_pose = np.array([1.72, 0.0, 1.4, 0.0, 10.0, 0.0]) # c'est bien 10 degres qu'il faut, les scans n'ont pas l'air d'etre terrible à verifier
    # T_calib_0_pose = np.array([0.0, 0.0, 1.4, 0.0, 0.0, 0.0]) # c'est bien 10 degres qu'il faut, les scans n'ont pas l'air d'etre terrible à verifier
    
    # eagle
    #T_calib_0_pose = np.array([0.0, -1.4, 1.72, -90.0, 90.0, 0.0])
    # T_calib_0_pose = np.array([1.72, 0.0, 1.2, -90.0, 0.0, -90.0]) 
    #T_calib_0_pose = np.array([1.80, 0.0, 1.1, -95.0, 0.0, -90.0]) 
    #T_calib_0_pose = np.array([1.50, 0.0, 1.1, -95.0, 0.0, -90.0]) 


    # # lca2
    # T_calib_0_pose = np.array([3.8, -1, 0, -90.0, 0, -120.0])  

    plane_calib.set_x0(T_calib_0_pose)

    if arg.view:
        viewer_eagle_scan_selection =  ViewerEagleScanSelection(plane_calib, plane_calib.position_offset)

    t = time.time()

    plane_calib.select_scans()
    print('scan select duration {}s'.format(time.time()-t))
    t = time.time()
    #pt = pole_calib.eval_cost(np.zeros(6))
    xHat = plane_calib.optimize()
    print('optimization duration {}s'.format(time.time()-t))
    print('solution x:{}'.format(xHat))
    print('solution Ax Angle:{}'.format(das.imu.axangle3_to_4(xHat[3:6])))
    Tr = das.imu.x_to_mat(xHat)
    print('solution Euler Angle (deg):{}'.format(np.rad2deg(das.imu.get_euler_angleZYX(Tr))))
    print('solution Tr:\n{}'.format(Tr))

    file_out = 'sbgekinox_bcc-{}.pkl'.format(arg.sensor)
    with open(file_out, 'wb') as f:
        pickle.dump(Tr, f)

    plane_calib.update_calibration('vlp16_tfc', xHat)


    


    pass
