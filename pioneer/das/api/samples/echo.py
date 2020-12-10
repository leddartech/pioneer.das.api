from pioneer.common import banks, clouds, images, plane, platform
from pioneer.common.logging_manager import LoggingManager
from pioneer.das.api.samples.sample import Sample

import copy
import numpy as np


class Echo(Sample):
    """Sample from a single data package provided by a LCAx sensor. 
        See pioneer.common.clouds.to_echo_package() to create a similar data package from scratch.
    """

    def __init__(self, index, datasource, virtual_raw=None, virtual_ts=None):
        super(Echo, self).__init__(index, datasource, virtual_raw, virtual_ts)
        self._mask = None

    @property
    def raw(self):
        if self._raw is None:
            r = super(Echo, self).raw
            r['das.sample'] = self
            if r['data']['timestamps'][-1] == 0:
                try:
                    cfg = self.datasource.sensor['cfg'].get_at_timestamp(
                        self.timestamp).raw
                    if self.datasource.sensor.specs is None:
                        self.datasource.sensor.specs = banks.extract_specs(
                            lambda n: cfg[n])

                    try:
                        if self.datasource.sensor.modules_angles is None:
                            self.datasource.sensor.modules_angles = banks.extract_intrinsics_modules_angles(
                                lambda n: cfg[n])
                    except:
                        LoggingManager.instance().warning(
                            "Sensor {} has no modules angles, or can not read modules angles from the sensor intrinsic calibratrion".format(self.datasource.sensor.name))

                    banks.add_timestamp_offsets(r, self.datasource.sensor.name, self.datasource.sensor.specs,
                                                int(cfg['ID_ACCUMULATION_EXP']), int(cfg['ID_OVERSAMPLING_EXP']), int(cfg['ID_BASE_POINT_COUNT']))
                except:
                    pass

            self._raw = r

        return self._raw

    @property
    def specs(self):
        # override the sensor specs if they are present in the YAML config file
        sensor_specs = self.datasource.sensor.specs
        if sensor_specs is not None:
            return sensor_specs
        return {k: self.raw[k] for k in ['v', 'h', 'v_fov', 'h_fov']}

    @property
    def v(self):
        return self.raw['v']

    @property
    def h(self):
        return self.raw['h']

    @property
    def mask(self):
        if self._mask is None:
            data = self.raw['data']
            config = self.datasource.sensor.config

            vmask = np.bitwise_and(data['flags'], 0x01).astype(
                np.bool)  # keep valid echoes only
            fmask = np.isin(
                data['flags'], config['reject_flags'], invert=True) & vmask

            def get_mask(slices, values):
                dmask = np.full_like(fmask, True)
                for s, e in slices:
                    dmask = dmask & ~((values >= s) & (values <= e))
                return dmask

            self._mask = fmask & get_mask(config['dist_reject_intervals'], data['distances']) & get_mask(
                config['amp_reject_intervals'], data['amplitudes'])

        return self._mask

    @property
    def masked(self):
        r = dict(self.raw)
        r['data'] = self.data
        return r

    @property
    def data(self):
        return self.raw['data'][self.mask]

    @property
    def indices(self):
        return self.data['indices']

    @property
    def timestamps(self):
        try:  # if relative timestamps are converted to global, there will be a key 'timestamps' in the raw
            t_ = self.raw['timestamps'][self.mask]
        except:  # otherwise global timestamps are considered to be directly in the data
            t_ = self.raw['data']['timestamps'][self.mask]
        return t_

    @property
    def distances(self):
        d = self.data['distances']
        s = self.datasource.sensor
        d = s.apply_distance_corrections(self.timestamp, self.indices, d)
        return d

    @property
    def amplitudes(self):
        return self.data['amplitudes']

    @property
    def flags(self):
        return self.data['flags']

    def cache(self):
        return self.datasource.sensor.cache(self.specs)

    def get_cloud(self, referential: str = None, ignore_orientation: bool = False, undistort: bool = False, reference_ts: int = -1, dtype: np.dtype = np.float64):
        points, amplitudes, triangles = self.quad_cloud(
            referential, ignore_orientation, undistort, reference_ts, dtype)
        return points, amplitudes, triangles.reshape(-1, 3)

    def point_cloud(self, referential: str = None, ignore_orientation: bool = False, undistort: bool = False, reference_ts: int = -1,
                    dtype: np.dtype = np.float64):
        """Compute a 3D point cloud from raw data

        Args:
            referential: The target sensor referential or full datasource name
            ignore_orientation: Ignore the source sensor orientation (default: {False})
            undistort: Apply motion compensation to 3d points.
            reference_ts: (only used if referential == 'world' and/or undistort == True), refer to compute_transform()'s documentation
            dtype: the output numpy data type
        """

        pts_Local = self.datasource.sensor.get_corrected_cloud(
            self.timestamp, self.cache(), 'point_cloud', self.indices, self.distances, None, dtype)

        if undistort:
            to_world = referential == 'world'
            self.undistort_points(
                [pts_Local], self.timestamps, reference_ts, to_world, dtype=dtype)
            if to_world:
                return pts_Local  # note that in that case, orientation has to be ignored

        return self.transform(pts_Local, referential, ignore_orientation, reference_ts, dtype=dtype)

    def quad_cloud(self, referential=None, ignore_orientation=False, undistort=False, reference_ts=-1, dtype=np.float64):
        """Compute a 3d surface cloud from raw data (quads made of 2 triangles)

        Args:
            referential: The target sensor referential or full datasource name
            ignore_orientation: Ignore the source sensor orientation (default: {False})
            undistort: Apply motion compensation to 3d points.
            reference_ts: (only used if referential == 'world' and/or undistort == True), 
                          refer to compute_transform()'s documentation
            dtype: the output numpy data type
        """

        pts_Local, quad_amplitudes, quad_indices = self.datasource.sensor.get_corrected_cloud(
            self.timestamp, self.cache(), 'quad_cloud', self.indices, self.distances, self.amplitudes, dtype)

        if undistort:
            sn = self.indices.shape[0]

            # four points per quad, 1 different direction per point, same distance for each
            to_world = referential == 'world'
            self.undistort_points([pts_Local[0:sn], pts_Local[sn:2*sn], pts_Local[2*sn:3*sn],
                                   pts_Local[3*sn:]], self.timestamps, reference_ts, to_world, dtype=dtype)
            if to_world:
                # note that in that case, orientation has to be ignored
                return pts_Local, quad_amplitudes, quad_indices

        pts_Ref = self.transform(
            pts_Local, referential, ignore_orientation, reference_ts, dtype=dtype)
        return pts_Ref, quad_amplitudes, quad_indices

    def amplitude_img(self, options='max_amplitude', dtype=np.float32, extrema=None):

        if extrema is None:
            extrema = self.datasource.sensor.config['extrema_amp']

        if options == 'max_amplitude':
            img = images.extrema_image(
                self.v, self.h, self.data, sort_field='amplitudes', sort_direction=-1, dtype=dtype, extrema=extrema)
        elif options == 'min_distance':
            _, others = images.extrema_image(self.v, self.h, self.data, sort_field='distances', sort_direction=1,
                                             other_fields=['amplitudes'], dtype=dtype, extrema=extrema)
            img = others['amplitudes']

        elif options == 'amplitudes_sum':
            img = images.accumulation_image(
                self.v, self.h, indices=self.data['indices'], weights=self.data['amplitudes'], dtype=dtype)
        else:
            raise ValueError('Invalid options: {}'.format(options))

        return self.transform_image(img)

    def distance_img(self, options='min_distance', dtype=np.float32, extrema=None):

        if extrema is None:
            extrema = self.datasource.sensor.config['extrema_dist']

        if options == 'max_amplitude':
            _, others = images.extrema_image(self.v, self.h, self.data, sort_field='amplitudes',
                                             sort_direction=-1, other_fields=['distances'], dtype=dtype, extrema=extrema)
            img = others['distances']

        elif options == 'min_distance':
            img = images.extrema_image(
                self.v, self.h, self.data, sort_field='distances', sort_direction=1, dtype=dtype, extrema=extrema)
        elif options == 'distances_sum':
            img = images.accumulation_image(
                self.v, self.h, indices=self.data['indices'], weights=self.data['distances'], dtype=dtype)
        else:
            raise ValueError('Invalid options: {}'.format(options))

        return self.transform_image(img)

    def other_field_img(self, field, dtype=np.float32):
        amp, others = images.extrema_image(self.v, self.h, self.data, sort_field='amplitudes',
                                           sort_direction=-1, other_fields=[field], dtype=dtype, extrema=None)
        img = others[field]

        return self.transform_image(img)

    @property
    def coords_img_tf(self):
        if not hasattr(self, '_coords_img_tf'):
            vv, hh = np.mgrid[0:self.v, 0:self.h]
            coords_img = np.stack(
                (vv, hh, np.arange(0, self.v*self.h).reshape(self.v, self.h)), axis=2)
            self._coords_img_tf = self.transform_image(coords_img)
        return self._coords_img_tf

    def image_coord_to_channel_index(self, row, col):
        return self.coords_img_tf[row, col, 2]

    def channel_index_to_image_coord(self, index):
        return np.argwhere(self.coords_img_tf[..., 2] == index)[0]

    def image_stack(self, field: str, number_images: int = 3, options='min_distance', dtype=np.float32):
        """Returns multiple images to account for the multiple echoes per channel"""
        image_stack = np.zeros((number_images, self.v, self.h))
        remaining_data = copy.deepcopy(self.data)
        for i in range(number_images):
            if remaining_data.size == 0:
                break
            if options == 'min_distance':
                mask = images.echoes_visibility_mask(remaining_data)
            elif options == 'max_amplitude':
                mask = images.maximum_amplitude_mask(remaining_data)

            image_stack[i] = self.transform_image(images.extrema_image(self.v, self.h, remaining_data[mask],
                                                                       sort_direction=1, other_fields=[field], dtype=dtype)[1][field])
            remaining_data = remaining_data[~mask]
        return image_stack

    def clip_to_fov_mask(self, pts: np.ndarray) -> np.ndarray:

        lcax = self.datasource.sensor
        specs = self.specs

        if lcax.angle_chart:
            cache = self.cache()
            correct_v_angles = lcax.get_corrected_projection_data(
                self.timestamp, cache, 'angles')

            v_cell_size, h_cell_size = clouds.v_h_cell_size_rad(specs)

            planes = clouds.frustrum_planes(clouds.custom_frustrum_directions(
                correct_v_angles, v_cell_size, h_cell_size, dtype=np.float64))
        else:
            planes = clouds.frustrum_planes(clouds.frustrum_directions(
                specs['v_fov'], specs['h_fov'], dtype=np.float64))

        if self.orientation is not None:
            planes[:, :3] = (self.orientation @ (planes[:, :3].T)).T

        return plane.plane_test(planes[0], pts)\
            & plane.plane_test(planes[1], pts)\
            & plane.plane_test(planes[2], pts)\
            & plane.plane_test(planes[3], pts)

    def get_rgb_from_camera_projection(self, camera: str, undistort: bool = False, return_mask: bool = False):
        """Returns the rgb data for each point from its projected position in camera.

            Args:
                camera: (str) name of the camera datasource (ex: 'flir_bbfc_flimg')
                undistort: (bool) if True, motion compensation is applied to the points before the projection (default is False)
                return_mask: (bool) if True, also returns the mask that only includes points inside the camera fov.

            Returns:
                rgb: A Nx3 array, where N is the number of points in the point cloud. RGB data is in the range [0,255]
                mask (optional): a Nx1 array of booleans. Values are True where points are inside the camera fov. False elsewhere.
        """

        image_sample = self.datasource.sensor.pf[camera].get_at_timestamp(
            self.timestamp)

        pcloud = self.point_cloud(
            referential=platform.extract_sensor_id(camera), undistort=undistort)
        projection, mask = image_sample.project_pts(
            pcloud, mask_fov=True, output_mask=True)
        projection = projection.astype(int)

        rgb = np.zeros((pcloud.shape[0], 3))
        image = image_sample.raw_image()
        rgb[mask, :] = image[projection[:, 1], projection[:, 0]]

        if return_mask:
            return rgb, mask
        return rgb

    def get_pulses(self, trace_ds_type: str, pulse_sample_size: int = 10, trace_processing=None, return_distance_scaling: bool = False):
        """Returns the corresponding pulses for every echoes from a trace datasource.

            Args:
                trace_ds_type: (str) the type of trace datasource (ex: 'trr' or 'ftrr')
                pulse_sample_size: (int) The number of data points to gather before and after each echo in the waveforms.
                    For example, with pulse_sample_size=10, the pulses will be 21 points, because the highest point is
                    taken in addition to the 10 points before and the 10 points after.
                trace_processing: (callable) function applied to the traces.
                return_distance_scaling: (bool) If True, also return the distance (meters) between trace data points.
        """
        trace_sample = self.datasource.sensor[trace_ds_type].get_at_timestamp(
            self.timestamp)
        traces = trace_sample.raw if trace_processing is None else trace_sample.processed(
            trace_processing)
        if 'high' in traces:
            traces = traces['high']

        full_traces = traces['data'][self.indices]

        if isinstance(traces['time_base_delays'], float):
            time_base_delays = traces['time_base_delays'] 
        else: 
            time_base_delays = traces['time_base_delays'][self.indices]

        echoes_positions_in_traces = (
            (self.distances - time_base_delays)/traces['distance_scaling']).astype(int)
        ind = np.indices(echoes_positions_in_traces.shape)
        padded_traces = np.pad(
            full_traces, ((0, 0), (pulse_sample_size, pulse_sample_size+1)))

        pulses = np.vstack([padded_traces[ind, echoes_positions_in_traces+ind_pulse+pulse_sample_size]
                            for ind_pulse in np.arange(-pulse_sample_size, pulse_sample_size+1, dtype=int)]).T

        if return_distance_scaling:
            return pulses, traces['distance_scaling']
        return pulses

    def get_signal_to_noise(self, trace_ds_type: str):
        traces = self.datasource.sensor[trace_ds_type].get_at_timestamp(
            self.timestamp).raw
        if 'high' in traces:
            traces = traces['high']
        traces_zeroed = traces['data'] - \
            np.mean(traces['data'], axis=1)[:, None]
        noise = np.std(traces_zeroed, axis=1)[self.indices]
        return np.log10(self.amplitudes**2/noise)+1
