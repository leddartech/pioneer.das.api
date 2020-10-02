from pioneer.common import clouds, banks, images, plane
from pioneer.common.logging_manager import LoggingManager
from pioneer.das.api.samples.sample import Sample

import numpy as np

class Echo(Sample):

    def __init__(self, index, datasource, virtual_raw = None, virtual_ts = None):
        super(Echo, self).__init__(index, datasource, virtual_raw, virtual_ts)
        self._mask = None

    @property
    def raw(self):
        if self._raw is None:
            r = super(Echo, self).raw
            r['das.sample'] = self
            if 'indices' in r: # old package conversion
                if not hasattr(self.datasource, '_warned_ech'):
                    LoggingManager.instance().warning("Deprecated 'ech' format detected, please convert/re-export dataset {}".format(self.datasource.sensor.platform.dataset))
                    self.datasource._warned_ech = True
                try:
                    if self.datasource.sensor.specs is None:
                        cfg = self.datasource.sensor['cfg'][0].raw
                        self.datasource.sensor.specs = banks.extract_specs(lambda n: cfg[n])
                finally:
                    r = clouds.convert_echo_package(r, specs = self.datasource.sensor.specs)
            if r['data']['timestamps'][-1] == 0:
                try:
                    cfg = self.datasource.sensor['cfg'].get_at_timestamp(self.timestamp).raw
                    if self.datasource.sensor.specs is None:
                        self.datasource.sensor.specs = banks.extract_specs(lambda n: cfg[n])
                    
                    try:
                        if self.datasource.sensor.modules_angles is None:
                            self.datasource.sensor.modules_angles = banks.extract_intrinsics_modules_angles(lambda n: cfg[n])
                    except:
                        LoggingManager.instance().warning("Sensor {} has no modules angles, or can not read modules angles from the sensor intrinsic calibratrion".format(self.datasource.sensor.name))

                    banks.add_timestamp_offsets(r, self.datasource.sensor.name, self.datasource.sensor.specs, \
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

            vmask = np.bitwise_and(data['flags'], 0x01).astype(np.bool) #keep valid echoes only
        
            fmask = np.isin(data['flags'], config['reject_flags'], invert = True) & vmask

            def get_mask(slices, values):
                dmask = np.full_like(fmask, True)
                for s,e in slices:
                    dmask = dmask & ~((values >= s) & (values <= e))
                return dmask
            
            self._mask = fmask & get_mask(config['dist_reject_intervals'], data['distances']) & get_mask(config['amp_reject_intervals'], data['amplitudes'])

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
        try:#if relative timestamps are converted to global, there will be a key 'timestamps' in the raw
            t_ = self.raw['timestamps'][self.mask]
        except: #otherwise global timestamps are considered to be directly in the data
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

    def get_cloud(self, referential:str = None, ignore_orientation:bool=False, undistort:bool=False, reference_ts:int=-1, dtype:np.dtype = np.float64):
        points, amplitudes, triangles = self.quad_cloud(referential, ignore_orientation, undistort, reference_ts, dtype)
        return points, amplitudes, triangles.reshape(-1, 3)

    def point_cloud(self, referential:str=None, ignore_orientation:bool=False, undistort:bool=False, reference_ts:int=-1, 
                                dtype:np.dtype=np.float64):
        """Compute a 3D point cloud from raw data
        
        Args:
            referential: The target sensor referential or full datasource name
            ignore_orientation: Ignore the source sensor orientation (default: {False})
            undistort: Apply motion compensation to 3d points.
            reference_ts: (only used if referential == 'world' and/or undistort == True), refer to compute_transform()'s documentation
            dtype: the output numpy data type
        """
        pts_Local = self.datasource.sensor.get_corrected_cloud(self.timestamp, self.cache(), 'point_cloud', self.indices, self.distances, None, dtype)
        
        if undistort:
            to_world = referential == 'world'
            self.undistort_points([pts_Local], self.timestamps, reference_ts, to_world, dtype = dtype)
            if to_world:
                return pts_Local # note that in that case, orientation has to be ignored

        return self.transform(pts_Local, referential, ignore_orientation, reference_ts, dtype = dtype)

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
            self.undistort_points([pts_Local[0:sn], pts_Local[sn:2*sn], pts_Local[2*sn:3*sn], pts_Local[3*sn:]]
            , self.timestamps, reference_ts, to_world, dtype = dtype)
            if to_world:
                return pts_Local, quad_amplitudes, quad_indices # note that in that case, orientation has to be ignored

        pts_Ref = self.transform(pts_Local, referential, ignore_orientation, reference_ts, dtype = dtype)
        return pts_Ref, quad_amplitudes, quad_indices

    def amplitude_img(self, options='max_amplitude', dtype = np.float32, extrema = None):

        if extrema is None:
            extrema = self.datasource.sensor.config['extrema_amp']

        if options == 'max_amplitude':
            img = images.extrema_image(self.v, self.h, self.data
                                                     , sort_field = 'amplitudes'
                                                     , sort_direction = -1
                                                     , dtype=dtype
                                                     , extrema = extrema)
        elif options == 'min_distance':
            _, others = images.extrema_image(self.v, self.h
                                                     , self.data
                                                     , sort_field = 'distances'
                                                     , sort_direction = 1
                                                     , other_fields=['amplitudes']
                                                     , dtype=dtype
                                                     , extrema = extrema)
            img = others['amplitudes']

        elif options == 'amplitudes_sum':
            img = images.accumulation_image(self.v, self.h
                                                          , indices = self.data['indices']
                                                          , weights = self.data['amplitudes']
                                                          , dtype=dtype)
        else:
            raise ValueError('Invalid options: {}'.format(options))

        return self.transform_image(img)

    def distance_img(self, options='min_distance', dtype = np.float32, extrema = None):

        if extrema is None:
            extrema = self.datasource.sensor.config['extrema_dist']

        if options == 'max_amplitude':
            _, others = images.extrema_image(self.v, self.h
                                                     , self.data
                                                     , sort_field = 'amplitudes'
                                                     , sort_direction = -1
                                                     , other_fields = ['distances']
                                                     , dtype=dtype
                                                     , extrema = extrema)
            img = others['distances']

        elif options == 'min_distance':
            img = images.extrema_image(self.v, self.h
                                                     , self.data
                                                     , sort_field = 'distances'
                                                     , sort_direction = 1
                                                     , dtype = dtype
                                                     , extrema = extrema)
        elif options == 'distances_sum':
            img = images.accumulation_image(self.v, self.h
                                                          , indices = self.data['indices']
                                                          , weights = self.data['distances']
                                                          , dtype = dtype)
        else:
            raise ValueError('Invalid options: {}'.format(options))

        return self.transform_image(img)

    def other_field_img(self, field, dtype = np.float32):

        amp, others = images.extrema_image(self.v, self.h
                                                    , self.data
                                                    , sort_field = 'amplitudes'
                                                    , sort_direction = -1
                                                    , other_fields = [field]
                                                    , dtype=dtype
                                                    , extrema = None)
        img = others[field]

        return self.transform_image(img)

    def image_coord_to_channel_index(self, row, col):
        #FIXME: this whould be cached
        vv, hh = np.mgrid[0:self.v, 0:self.h]
        coords_img = np.stack((vv,hh, np.arange(0, self.v*self.h).reshape(self.v, self.h)), axis=2)
        coords_img_tf = self.transform_image(coords_img)

        return coords_img_tf[row, col, 2]

    def channel_index_to_image_coord(self, index):
        #FIXME: this should be cached
        vv, hh = np.mgrid[0:self.v, 0:self.h]
        coords_img = np.stack((vv,hh, np.arange(0, self.v*self.h).reshape(self.v, self.h)), axis=2)
        coords_img_tf = self.transform_image(coords_img)

        return np.argwhere(coords_img_tf[...,2] == index)[0]

    def clip_to_fov_mask(self, pts:np.ndarray) -> np.ndarray:
                
        lcax = self.datasource.sensor
        specs = self.specs

        if lcax.angle_chart:
            cache = self.cache()
            correct_v_angles = lcax.get_corrected_projection_data(self.timestamp, cache, 'angles')

            v_cell_size, h_cell_size = clouds.v_h_cell_size_rad(specs)

            planes = clouds.frustrum_planes(clouds.custom_frustrum_directions(correct_v_angles, v_cell_size, h_cell_size, dtype = np.float64))
        else:
            planes = clouds.frustrum_planes(clouds.frustrum_directions(specs['v_fov'], specs['h_fov'], dtype = np.float64))

        if self.orientation is not None:
            planes[:, :3] = (self.orientation @ (planes[:, :3].T)).T

        return plane.plane_test(planes[0], pts)\
                & plane.plane_test(planes[1], pts)\
                & plane.plane_test(planes[2], pts)\
                & plane.plane_test(planes[3], pts)


