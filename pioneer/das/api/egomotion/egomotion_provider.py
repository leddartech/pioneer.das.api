from pioneer.common import linalg

import numpy as np

class EgomotionProvider(object):
    """Computes the position and orientation of the ego vehicle. A Platform must have
        an ego motion provider in order to use the 'world' referential.

        A sub-class should be created and adapted to a motion sensing device. The sub-class
        must override the get_Global_from_Ego_at() method, with gathers the motion sensing
        device's raw data at a given timestamp, then computes and returns a 4x4 affine 
        transformation matrix corresponding to the position and orientation of the ego vehicle.

        The ego motion provider can be automatically added to a Platform if the motion sensing
        device has a create_egomotion_provider() method.
    """

    def __init__(self, referential_name:str, subsampling:int=100):
        self.subsampling = subsampling
        self._referential_name = referential_name
        self._tf_Global_from_EgoZero = linalg.tf_eye(dtype = np.float64)
  
    @property 
    def tf_Global_from_EgoZero(self) -> np.ndarray:
        return self._tf_Global_from_EgoZero

    @property 
    def referential_name(self) -> str:
        return self._referential_name

    def get_Global_from_Ego_at(self, ts:int, dtype = np.float64) -> np.ndarray:
        """Transform to map a point from world coords to EgomotionProdiver's coords"""
        raise RuntimeError("Not Implemented")
    
    def compute_tf_EgoZero_from_Sensor(self, tf_Ego_from_Sensor:np.ndarray, reference_ts:int)->np.ndarray:
        tf_Global_from_Ego = self.get_Global_from_Ego_at(reference_ts) 
        return linalg.tf_inv(self.tf_Global_from_EgoZero) @ tf_Global_from_Ego @ tf_Ego_from_Sensor 


    def compute_trajectory(self, sorted_ts:np.ndarray, tf_Global_from_EgoZero:np.ndarray, dtype = np.float64) -> np.ndarray:
        """Trajectory of tf_EgoZero_from_Ego transforms """
        end = int(np.ceil(sorted_ts.shape[0]/self.subsampling))
        # use double precision for trajectory computation, due to utm big number
        trajectory_EgoZero_from_Ego = np.empty((end,4,4), dtype = dtype)

        tf_EgoZero_from_Global = linalg.tf_inv(tf_Global_from_EgoZero)
       
        for i in range(end):
            tf_Global_from_Ego = self.get_Global_from_Ego_at(sorted_ts[i*self.subsampling])
            trajectory_EgoZero_from_Ego[i] = tf_EgoZero_from_Global @ tf_Global_from_Ego

        return trajectory_EgoZero_from_Ego        

    def apply_trajectory(self, trajectory:np.ndarray, points:np.ndarray) -> np.ndarray:
        corrected = np.empty_like(points)
        n_points = points.shape[0]
        for i in range(trajectory.shape[0]):
            s, e = i * self.subsampling, min((i+1) * self.subsampling, n_points)
            corrected[s:e] = linalg.map_points(trajectory[i], points[s:e])
        return corrected
