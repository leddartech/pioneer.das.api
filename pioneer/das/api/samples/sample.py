from pioneer.common import platform, linalg
from pioneer.common.logging_manager import LoggingManager

import numpy as np
from numpy import rot90, flipud, fliplr #for use with Sample.LUT
import pprint
import six

def warn_if_less_than_64bit(dtype:np.dtype):
    if np.finfo(dtype).bits < 64:
        LoggingManager.warning(f"{dtype} precision detected, performing computations in that context with less than 64bits precision is ill-advised")


class Sample(object):

    LUT = {}

    @staticmethod
    def __goc_lut():
        if not Sample.LUT:
            # Basic transformation on xy-plane
            rotations = {
            # Clockwise rotation sends x to -y and y to x
            'rot90({},k=-1)': np.array([[0, -1,  0],
                 [1,  0,  0],
                 [0,  0,  1]], 'f4'),
            # Counter-clockwise rotation sends x to y and y to -x
            'rot90({},k=1)': np.array([[0,  1,  0],
                 [-1, 0,  0],
                 [0,  0,  1]], 'f4'),
            'rot90({},k=2)': np.array([[-1, 0,  0],
                 [0, -1,  0],
                 [0,  0,  1]], 'f4'),
            '{}': np.eye(3, dtype='f4')}

            reflexions = {
            'fliplr({})': np.array([[-1, 0,  0],
                 [0,  1,  0],
                 [0,  0,  1]], 'f4'),
            'flipud({})': np.array([[1,  0,  0],
                 [0, -1,  0],
                 [0,  0,  1]], 'f4'),
            '{}': np.eye(3, dtype='f4')}

            # Other orientations than are not defined on xy plane:
            special_transformations = {
                np.array([[1,  0,  0],
                 [0, 1,  0],
                 [0,  0,  -1]], 'f4').tostring(): 'flipud({})',
                np.array([[0, 0, 1],
                 [-1, 0, 0],
                 [0, 1, 0]], 'f4').tostring(): '{}',
                np.array([[0, 0, 1],
                 [-1, 0, 0],
                 [0, -1, 0]], 'f4').tostring(): 'fliplr({})',

            }

            for rot_op, rot in rotations.items():
                for refl_op, refl in reflexions.items():
                    comb = np.matmul(rot, refl)
                    inv_com = np.matmul(refl, rot)
                    Sample.LUT[comb.tostring()] = rot_op.format(refl_op)
                    Sample.LUT[inv_com.tostring()] = refl_op.format(rot_op)

            Sample.LUT.update(special_transformations)

        return Sample.LUT

    def __init__(self, index, datasource, virtual_raw = None, virtual_ts = None):
        '''
            virtual_raw and virtual_ts are meant for samples which dont exist in
            a physical datasource. For example interpolated samples, or samples
            derived from other samples.
        '''
        self.index = index
        self.datasource = datasource
        self.virtual_raw = virtual_raw
        self.virtual_ts = virtual_ts
        self._raw = None

    @property
    def orientation(self):
        return self.datasource.sensor.orientation

    @property
    def intrinsics(self):
        return self.datasource.sensor.intrinsics

    @property
    def extrinsics(self):
        return self.datasource.sensor.extrinsics

    @property
    def raw(self):
        if self.virtual_raw is None:
            if self._raw is None:
                self._raw = self.datasource.ds[self.index]
            return self._raw
        return self.virtual_raw

    @property
    def timestamp(self):
        if self.virtual_ts is None:
            return self.datasource.timestamps[self.index]
        return self.virtual_ts

    @property
    def time_of_issues(self):
        if self.virtual_ts is None:
            return self.datasource.time_of_issues[self.index]
        return self.virtual_ts

    @property
    def label(self):
        return self.datasource.label

    @property
    def sensor_type(self):
        return platform.parse_datasource_name(self.datasource.label)[0]

    def compute_transform(self, referential_or_ds:str=None, ignore_orientation:bool=False, reference_ts:int = -1, dtype = np.float64) -> np.ndarray:
        """Compute the transform from this Sample sensor to another sensor
        referential.

        By default the transform between the two sensors reference frames include
        the contribution from the orientation member. If ones wishes to ignore this
        contribution it can set the ignore_orientation parameter to True.

        Args:
            referential_or_ds: The other sensor name or full datasource name, 
                                e.g. flir_tfc or flir_tfc_img (default: {None})
            ignore_orientation: Ignore the source sensor orientation (default: {False})
            reference_ts:   (only used if referential_or_ds == 'world'), the timestamp at which we want to 'jump' from 
                            EgomotionProvider's referential (e.g. the IMU) to the to 'world' referential. Also note that 
                            the 'world' referential is actually EgomotionProdiver's referential at some reference time 't' 
                            (refer to your actual EgomotionProvider's configuration, by default 't' is EgomotionProvider's
                            initial timestamp). 'world' referential is thus commonly refered to as the 'EgoZero' referential

            dtype: the output numpy data type

        Returns:
            The [4, 4] transformation matrix
        """

        tf_TargetRef_from_Local = np.eye(4, dtype = dtype)
       
        try:
            if isinstance(referential_or_ds, six.string_types) and self.datasource.sensor.name != referential_or_ds:
                
                referential = platform.referential_name(referential_or_ds)

                if referential == 'world':

                    warn_if_less_than_64bit(dtype)

                    if reference_ts < 0:
                        reference_ts = self.timestamp

                    provider = self.datasource.sensor.platform.egomotion_provider
                    tf_Ego_from_Local = self.datasource.sensor.map_to(provider.referential_name)
                    tf_TargetRef_from_Local = provider.compute_tf_EgoZero_from_Sensor(tf_Ego_from_Local, reference_ts)

                else:
                    tf_TargetRef_from_Local = np.copy(self.datasource.sensor.map_to(referential))
        except:
            LoggingManager.instance().warning(f'Transformation matrix between {self.datasource.sensor.name} and {referential_or_ds} could not be computed.')
                    
        if self.orientation is not None and not ignore_orientation:
            tf_TargetRef_from_Local[:3, :3] = tf_TargetRef_from_Local[:3, :3] @ self.orientation
        
        return tf_TargetRef_from_Local

    def transform(self, pts:np.ndarray, referential_or_ds:str, ignore_orientation:bool=False, reference_ts:int = -1, reverse:bool=False, dtype = np.float64) -> np.ndarray:
        """Transform 3D points from this Sample sensor to another sensor
        referential.

        Arguments:
            pts: The [N, 3] points to be transformed
            referential_or_ds: The target sensor referential or full datasource name
            ignore_orientation: Ignore the source sensor orientation (default: {False})
            reference_ts: refer to compute_transform()'s doc (only used if referential_or_ds == 'world')
            reverse: apply the reverse transformation
            dtype: the output numpy data type

        Returns:
            The transformed points
        """
        r = self.compute_transform(referential_or_ds, ignore_orientation=ignore_orientation, reference_ts=reference_ts, dtype=dtype)

        if reverse:
            r = linalg.tf_inv(r)

        if r is not None:
            return Sample.transform_pts(r, pts)

        return pts

    def undistort_points(self, pts_Local_batches:list, timestamps:np.ndarray, reference_ts:int = -1, to_world:bool = False, dtype = np.float64):
        """Transform 3D points that have not been sampled simultaneously to their 'correct' place
        referential.

        Args:
            pts_Local_batches: a list of arrays of [N, 3] points to be transformed
            timestamps: the N timestamps (common for all point batches)
            to_world:   If 'True', leave undistorted points in 'world' referential, otherwise
                        project them back to local referential
            reference_ts:   only used if to_world == False, let the use chose at what time
                            undistorted points are projected back to the local referential
                            (useful to compare points from different sensors in a common local referential)
            dtype: the output numpy data type

        Returns:
            The transformed points
        """

        warn_if_less_than_64bit(dtype)

        provider = self.datasource.sensor.platform.egomotion_provider        
        tf_Ego_from_Local = self.compute_transform(provider.referential_name, False, dtype = dtype)
        traj_EgoZero_from_Ego = provider.compute_trajectory(timestamps, provider.tf_Global_from_EgoZero, dtype = dtype)

        for pts_Local in pts_Local_batches:

            pts_Ego = linalg.map_points(tf_Ego_from_Local, pts_Local)

            pts_EgoZero = provider.apply_trajectory(traj_EgoZero_from_Ego, pts_Ego)

            if to_world:
                pts_Local[:] = pts_EgoZero
            else:
                if reference_ts < 0:
                    reference_ts = self.timestamp

                tf_Global_from_Ego = provider.get_Global_from_Ego_at(reference_ts, dtype = dtype)
                tf_Local_from_EgoZero = linalg.tf_inv(tf_Ego_from_Local) @ linalg.tf_inv(tf_Global_from_Ego) @ provider.tf_Global_from_EgoZero
                pts_Local[:] = linalg.map_points(tf_Local_from_EgoZero, pts_EgoZero)

    def _get_orientation_lut(self):
        if self.orientation is not None:
            orientation_f4 = self.orientation.astype('f4')
            lut = Sample.__goc_lut()
            if orientation_f4.tostring() in lut:
                return lut[orientation_f4.tostring()]

            LoggingManager.instance().warning('orientation {} could not be mapped to any image transform'.format(self.orientation))
        return None

    def transform_image(self, image):
        # Transform the bottom left origin of LCAx to top-left camera origin
        image = np.flipud(image)

        operation = self._get_orientation_lut()
        if operation is not None:
            return np.copy(eval(operation.format('image')))

        return np.copy(image)

    @staticmethod
    def transform_pts(matrix4x4, ptsNx3):
        return linalg.map_points(matrix4x4, ptsNx3)

    def pretty_print(self):
        return pprint.pformat(self.raw)
