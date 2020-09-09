import collections
import numbers
import os
import re
import sys

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

from das.api import chessboard, platform, sensors
from das.calib import intrinsics
from das.utils import ransac_fit_plane_ols, normal_from_parameters, ray_plane_intersection
from leddar_utils import clouds, images
from sklearn.neighbors import KDTree

try:
    from yav import viewer, amplitudes_to_color
except ImportError:
    print("You don't have YAV. Please clone and install the YAV repository from"
          "http://svleddar-gitlab/Advanced_Engineering/SDK/yav.",
          file=sys.stderr)
    raise

class Sensor_(object):

    def __init__(self, sensor, datasource_name, config):
        self._sensor = sensor
        self._datasource_name = datasource_name
        self.config = config
        self.view = None

    @property
    def name(self):
        return self._sensor.name

    @property
    def full_datasource_name(self):
        return self._sensor.datasources[self._datasource_name].label

    def detect_chessboard(self, data):
        raise NotImplementedError()

    def first_detected_chessboard_frame(self):
        return None

    def create_view(self, v):
        raise NotImplementedError()

    def update_view(self, data, found, corners):
        raise NotImplementedError()

    @property
    def name(self):
        return self._sensor.name

class Camera_(Sensor_):

    def __init__(self, sensor, datasource_name, config):
        super(Camera_, self).__init__(sensor, datasource_name, config)
        self.chessboard = None
        self.chessboards = collections.OrderedDict()
        self._init_chessboards()
        self.matrix = self._sensor.camera_matrix
        self.distortion = self._sensor.distortion_coeffs

    def _init_chessboards(self):
        chessboard_name = 'chessboards'
        if chessboard_name in self._sensor.datasources:
            source = self._sensor.datasources[chessboard_name]
            pkl = source[0]
            corners_dict = pkl.raw['corners']
            frames = sorted(corners_dict.keys())
            for key in frames:
                value = corners_dict[key]
                try:
                    corners = value['corners']
                    found = value['found']
                    self.chessboards[key] = (found, corners)
                except:
                    continue
            self.chessboard = pkl.raw.get('chessboard', None)
            print('Loaded {} precomputed chessboards'.format(len(self.chessboards)))

        self.chessboard = None
        if self.chessboard is None:
            self.chessboard = \
                chessboard.load_chessboard_specifications(self._sensor.platform)
        assert self.chessboard is not None, \
               'Please specify the chessboard in the platform.yml file'

    @property
    def pattern_size(self):
        return self.chessboard['nx'], self.chessboard['ny']

    def detect_chessboard(self, data):
        image = data.raw
        precomputed = self.chessboards.get(data.index, None)
        if precomputed:
            found, points2d = precomputed
        else:
            found, points2d = intrinsics._get_corners(image, self.pattern_size)

        points3d = None
        if self.config.get('reflective_spheres', False):
            spheres = self.chessboard['reflective_spheres']
            assert spheres.size != 0, 'No reflective spheres defined in YAML'

            if found:
                points = self.chessboard['points']
                ret, R, T = cv2.solvePnP(points, points2d,
                                         self.matrix, self.distortion)
                assert ret, 'solvePnP failed'
                # project the 3d sphere point in the image
                points2d, _ = cv2.projectPoints(spheres, R, T,
                                                self.matrix, self.distortion)
                # transform the object frame to camera frame
                R, _ = cv2.Rodrigues(R)
                points3d = (R.dot(spheres.T) + T).T
        else:
            if found:
                points = self.chessboard['points']
                ret, R, T = cv2.solvePnP(points, points2d,
                                         self.matrix, self.distortion)
                assert ret, 'solvePnP failed'
                R, _ = cv2.Rodrigues(R)
                points3d = (R.dot(points.T) + T).T

        return found, dict(points2d=points2d, points3d=points3d)


    def first_detected_chessboard_frame(self):
        if self.chessboards:
            for frame in self.chessboards.keys():
                break
            return frame
        return None

    def create_view(self, v):
        self.view = v.create_image_window(title=self.name)

    def update_view(self, data, found, corners):
        image = data.raw.copy()
        points2d = corners['points2d']
        if self.config.get('reflective_spheres', False):
            del self.view.ax.lines[:]
            self.view.update(image[..., ::-1])
            if found:
                self.view.ax.plot(points2d[:, 0, 0], points2d[:, 0, 1], 'ro')
        else:
            if found:
                image = cv2.drawChessboardCorners(image, self.pattern_size,
                    points2d, found)
            self.view.update(image[..., ::-1])

        self.view.draw()

class LCAx_(Sensor_):

    def __init__(self, sensor, datasource_name, config, use_angular_calibration):
        super(LCAx_, self).__init__(sensor, datasource_name, config)
        # allow overriding specs
        self.pcd = None
        self.spheres = []
        self.amplitude_image_view = None
        self.chessboard = chessboard.load_chessboard_specifications(self._sensor.platform)
        pattern_size =  self.chessboard['nx'], self.chessboard['ny']
        self.debug = None
        self.debug_pcd = None
        self.use_angular_calibration = use_angular_calibration

    def _valid(self, data):
        # remove bad echoes
        mask = data.flags != 3
        if mask.all():
            mask = data.distances < 200
        return mask

    def _point_cloud(self, data):
        pc = data.point_cloud()
        return pc

    def _apply_plane_constraint(self, pt, pts):
        """Apply a plane constraint to the found point

        Arguments:
            pt {[type]} -- [description]
            pts {[type]} -- [description]

        Returns:
            [type] -- [description]
        """

        kdtree = KDTree(pts[:, :2])
        # search in a radius twice the size of a chessboard square
        radius = self.chessboard['dx']*2
        ind, = kdtree.query_radius(pt[:, :2], radius)
        center = np.median(pts[ind, :], axis=0)

        # remove any point that is further than 1.5 meter from the median
        # point in the point cloud
        kdtree = KDTree(pts[ind, :])
        ind2, = kdtree.query_radius(center[None, :], 1.5)
        ind = ind[ind2]

        # estimate a plane
        # we use only the neighbors (computed using x-y coordinates)
        # of the provided point
        mask = np.zeros(pts.shape[0], dtype=np.bool)
        mask[ind] = True

        new_pt = pt.copy()

        pts_masked = pts[mask, :]
        # we try to use more samples than the minimum 3 to save on ransac
        # iterations. Usually more than half of the points will be inliers.
        min_n =  max(3, min(100, pts_masked.shape[0]//2))
        if pts_masked.shape[0] >= min_n:
            params, in_plane, _ = ransac_fit_plane_ols(pts_masked,
                residual_threshold=0.1, min_samples=min_n, ransac_iter=100)
            if params is not None:
                in_plane_indices = np.arange(pts.shape[0])
                in_plane_indices = in_plane_indices[mask]
                in_plane_indices = in_plane_indices[in_plane]
                inliers = np.zeros(pts.shape[0], np.bool)
                inliers[in_plane_indices] = True

                normal = normal_from_parameters(params)
                if np.abs(np.dot(normal, [0, 0, 1])) < 0.75:
                    # force the found plane to be roughly perpendicular
                    # to the z-axis the axis of the sensor
                    params = in_plane = inliers = None
            else:
                params = in_plane = inliers = None
        else:
            params = in_plane = inliers = None

        colors = np.ones_like(pts, dtype='u1')*255

        if inliers is not None:
            colors[mask, :] = [0, 255, 0]
            colors[inliers, :] = [255, 0, 0]

            if self.debug is not None:
                self.debug_pcd.set_points(pts, colors)
                self.debug.render()

        new_pt = pt.copy()
        if inliers is not None:
            new_pt = ray_plane_intersection(pt, params)

        found = inliers is not None
        return found, new_pt

    def _detect_reflective_sphere(self, data):
        mask = self._valid(data)
        pc = self._point_cloud(data)

        # get the maximum amplitude point
        indices = np.arange(len(data.indices))
        amps = data.amplitudes[mask]
        index = amps.argmax()

        true_index = indices[mask][index:index+1]

        sphere_pt = pc[true_index, :]

        found, new_pt = self._apply_plane_constraint(sphere_pt, pc[mask, :])


        corners = dict(indices=true_index,
                       points3d=new_pt,
                       points2d=None,
                       amplitudes=data.amplitudes[true_index])
        return found, corners

    def detect_chessboard(self, data):
        if self.config.get('reflective_spheres', False):
            return self._detect_reflective_sphere(data)
        else:
            raise ValueError('CNN based chessboard detector is not supported anymore')

    def create_view(self, v):
        self.view = v.create_point_cloud_window(title=self.name)
        self.pcd = self.view.create_point_cloud()
        self.image_view = v.create_image_window(title=self.name + '_amplitude')
        self.spheres = []
        # Debug stuff
        self.debug = v.create_point_cloud_window(title='debug')
        self.debug_pcd = self.debug.create_point_cloud()

    def sphere(self, index):
        while index >= len(self.spheres):
            sphere = self.view.create_sphere(radius=0.01, color=[128, 0, 0])
            sphere.hide()
            self.spheres.append(sphere)
        return self.spheres[index]

    def update_view(self, data, found, corners):
        mask = self._valid(data)
        pc = self._point_cloud(data)

        amplitudes = data.amplitudes
        masked_amps = data.amplitudes[mask]
        norm = matplotlib.colors.LogNorm(masked_amps.min(), masked_amps.max())
        colors = amplitudes_to_color(norm(amplitudes), mask=mask)

        if found:
            points = corners['points3d']
            for i, pt in enumerate(points):
                sphere = self.sphere(i)
                sphere.set_center(pt)
                sphere.show()
        else:
            for s in self.spheres:
                s.hide()

        pc = pc[mask, :]
        colors = colors[mask]

        self.pcd.set_points(pc, colors)
        self.view.render()

        vis_mask = images.echoes_visibility_mask(data.raw['data'][data.mask]) & mask
        specs = data.specs
        shape = (specs['v'], specs['h'])
        image = images.echoes_to_image(shape, data.indices, data.amplitudes,
            mask=vis_mask)
        image = data.transform_image(image)
        # image = data.amplitude_img(options='min_distance')
        masked_amps = data.amplitudes[vis_mask]
        norm = matplotlib.colors.LogNorm(masked_amps.min(), masked_amps.max())
        del self.image_view.ax.lines[:]
        self.image_view.imshow(image)
        self.image_view._image.set_norm(norm)
        if found:
            points2d = corners['points2d']
            if points2d is not None:
                self.image_view.ax.plot(points2d[:, 0], points2d[:, 1], 'r.')
        self.image_view.draw()


def create_sensor(sensor, config, use_angular_calibration):
    if isinstance(sensor, sensors.LCAx):
        name = 'ech'
        return LCAx_(sensor, name, config, use_angular_calibration)
    elif isinstance(sensor, sensors.Sensor):
        name = 'img'
        if name in sensor.datasources:
            return Camera_(sensor, name, config)
    raise ValueError('Unsupported sensor type. Must be LCAx or Camera')

def sort_sensors(sensors):
    def camera_first(s):
        if isinstance(s, Camera_):
            return 0
        return 1
    sensors = sorted(sensors, key=camera_first)
    return sensors

def get_sources(sensors):
    sources = [s.full_datasource_name for s in sensors]
    return sources

def get_frame_homologs(frame, synched, sensors, sources):
    sensor1, sensor2 = sensors
    src1, src2 = sources

    data = synched[frame]
    data1 = data[src1]

    found1, corners1 = sensor1.detect_chessboard(data1)

    found2 = False
    corners2 = dict(points2d=None, points3d=None)

    if found1:
        data2 = data[src2]
        found2, corners2 = sensor2.detect_chessboard(data2)

    return found1, corners1, found2, corners2

def create_homologs_frame_callback(synched, sensors, sources):
    def callback(v):
        frame = v.get_frame()
        data = synched[frame]

        sensor1, sensor2 = sensors
        src1, src2 = sources

        found1, corners1, found2, corners2 = get_frame_homologs(frame, synched, sensors, sources)

        data1 = data[src1]
        sensor1.update_view(data1, found1, corners1)

        data2 = data[src2]
        sensor2.update_view(data2, found2, corners2)

    return callback

def compute_homologs(config, directory, view=False, view_homologs=False, use_angular_calibration = False):
    dataset = config['dataset']
    plat = platform.Platform(dataset)

    sensors_list = plat.sensors
    names = list(config['sensors'].keys())
    assert len(names) == 2, \
        'sensors dictionnary must contain 2 sensors. {} != 2'.format(len(names))

    # create thin wrapper classes on top of existing sensors
    sensors_list = [create_sensor(sensors_list[name], config['sensors'][name].get('config', {}), use_angular_calibration)
        for name in names]

    # sort the datasources so camera is always first
    # this way the stronger detector of the camera is used to filter frames
    # where there is no chessboard
    sensors_list = sort_sensors(sensors_list)
    names = [s.name for s in sensors_list]
    sources = get_sources(sensors_list)
    sensor1, sensor2 = sensors_list

    if isinstance(sensor1, Camera_) and isinstance(sensor2, Camera_):
        sensor1.config['reflective_spheres'] = False
        sensor2.config['reflective_spheres'] = False

    # get a synchronized platform
    synched = plat.synchronized(sources)

    # get the start and end of the range of frames we are interested in
    n = len(synched)

    # get the first detected chessboard frame
    first_frame = sensor1.first_detected_chessboard_frame()

    min_frame = config.get('min_frame', 0)
    max_frame = config.get('max_frame', n-1)

    min_frame = max(0, min_frame)
    max_frame = min(max_frame, n-1)

    src1, src2 = sources

    hfile = '{}-{}-homologs.csv'.format(sensor1.name, sensor2.name)
    error_file = '{}-{}-error.csv'.format(sensor1.name, sensor2.name)
    hpath = os.path.join(directory, hfile)
    epath = os.path.join(directory, error_file)
    frames = list(range(min_frame, max_frame+1))
    if view:
        v = viewer(num=len(frames), title='Homologs')
        for s in sensors_list:
            s.create_view(v)
        v.add_frame_callback(create_homologs_frame_callback(synched, sensors_list, sources))
        v.run()
    else:
        homologs = []
        error = []
        for i in tqdm(frames):
            data = synched[i]
            data1 = data[src1]
            if first_frame is not None and data1.index < first_frame:
                continue

            found1, corners1, found2, corners2 = \
                get_frame_homologs(i, synched, sensors_list, sources)

            if found1 and found2:
                pts1 = corners1['points3d']
                pts2 = corners2['points3d']
                if np.any(pts2) and not np.isnan(pts2).any():
                    n = pts1.shape[0]
                    homologs.append(np.concatenate([np.ones((n, 1))*i, pts1, pts2], axis=1))
                else :
                    error.append(i)
        if len(homologs) !=0:
            homologs = np.concatenate(homologs, axis=0)

            np.savetxt(hpath,
                    homologs)
            np.savetxt(epath,
                    error)

        if view_homologs:
            def update(v):
                i = v.get_frame()
                pts1 = homologs[:i+1, 1:4]
                pts2 = homologs[:i+1, 4:7]
                if pts1.size != 0:
                    c1 = np.zeros_like(pts1, dtype='u1')
                    c1[:, 0] = 255
                    c2 = np.zeros_like(pts2, dtype='u1')
                    c2[:, 1] = 255
                    pcd1.set_points(pts1, colors=c1)
                    pcd2.set_points(pts2, colors=c2)

            n = len(homologs)
            v = viewer(num=n)
            pcd1 = v.create_point_cloud()
            pcd2 = v.create_point_cloud()

            v.add_frame_callback(update)
            v.run()

    return sensor1.name, sensor2.name, hfile


if __name__ == '__main__':
    import argparse
    import yaml
    import copy

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='YAML config file')
    parser.add_argument('--angularCalib', action='store_true', help='activate the projection with angular calibration compensation')

    args = parser.parse_args()

    use_angular_calibration = args.angularCalib
    directory = os.path.split(os.path.realpath(args.config))[0]

    with open(args.config, 'r') as f:
        global_config = yaml.load(f)

    dataset = global_config['dataset']
    if not os.path.isabs(dataset):
        dataset = os.path.join(directory, dataset)

    view = global_config.get('view', False)
    view_homologs = global_config.get('view_homologs', False)
    sensors_dict = global_config['sensors']
    n_sensors = len(sensors_dict)
    names = list(sensors_dict.keys())
    if view and n_sensors > 2:
        # Don't know why but the script segfaults when we do this
        print('Cannot use view when the number of sensors is more than two',
              file=sys.stderr)
        sys.exit(1)
    transformations = {}
    for i in range(n_sensors):
        for j in range(i+1, n_sensors):
            sensor1 = names[i]
            sensor2 = names[j]
            # create a configuration for each pair of sensors and call
            # the compute homologs function
            config = dict(dataset=dataset,
                          sensors={
                            # copy the configuration since we will override
                            # the reflective_spheres usage when we have two
                            # cameras.
                            sensor1: copy.deepcopy(sensors_dict[sensor1]),
                            sensor2: copy.deepcopy(sensors_dict[sensor2])})
            print('Computing homologs {}-{}'.format(sensor1, sensor2))
            sensor1, sensor2, homolog_file = \
                compute_homologs(config, directory, view=view,
                                 view_homologs=view_homologs, use_angular_calibration=use_angular_calibration)
            transformations['{}-{}'.format(sensor1, sensor2)] = homolog_file


    if not view:
        loopfile = os.path.join(directory, 'loop_config.yml')
        index = 1
        while os.path.exists(loopfile):
            loopfile = os.path.join(directory, 'loop_config{}.yml'.format(index))
            index += 1

        loop = dict(
            transformations=transformations,
            loops=dict(loop1='{sensor1}-{sensor2}->-{sensor1}-{sensor2}'.format(
                sensor1=sensor1, sensor2=sensor2
            )),
            plot=None,
            output='loop_all.pkl',
            ransac=dict(
                nb_iterations=5000,
                distance_threshold=0.15
            )
        )

        with open(loopfile, 'w') as f:
            yaml.dump(loop, f, default_flow_style=False)





