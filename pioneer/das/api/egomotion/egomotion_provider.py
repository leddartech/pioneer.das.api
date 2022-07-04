from pioneer.common import linalg
from typing import Optional
from abc import ABC, abstractclassmethod

import numpy as np
import warnings
warnings.simplefilter('once', DeprecationWarning)



class EgomotionProvider(ABC):
    """Computes the position and orientation of the ego vehicle. A Platform must have
        an ego motion provider in order to use the 'world' referential.
    """

    def __init__(self, referential_name:str):
        self.referential_name = referential_name

    @abstractclassmethod
    def get_transform(self, timestamp:int) -> np.ndarray:
        """Returns the transform to the world from the Egomotion Provider's referential at a given time."""

    @abstractclassmethod
    def get_timestamps(self) -> np.ndarray:
        """Returns all timestamps in an array."""

    def get_inverse_transform(self, timestamp:int) -> np.ndarray:
        """Returns the transform to the Egomotion Provider's referential from the world at a given time."""
        return linalg.tf_inv(self.get_transform(timestamp))

    def get_first_transform(self) -> np.ndarray:
        """Returns the transform at the first available timestamp"""
        return self.get_transform(self.get_timestamps()[0])

    def get_first_inverse_transform(self) -> np.ndarray:
        """Returns the inverse transform at the first available timestamp"""
        return self.get_inverse_transform(self.get_timestamps()[0])

    def get_trajectory(self, timestamps:Optional[np.ndarray]=None, start_at_origin:bool=False, subsampling:int=100) -> np.ndarray:
        """Returns all transforms."""
        timestamps = self.get_timestamps() if timestamps is None else timestamps
        end = int(np.ceil(timestamps.shape[0]/subsampling))
        trajectory = np.empty((end,4,4))
        first_inverse_transform = self.get_first_inverse_transform() if not start_at_origin else np.eye(4)
        for i in range(end):
            to_world = self.get_transform(timestamps[i*subsampling])
            trajectory[i] = first_inverse_transform @ to_world
        return trajectory






    ### Legacy section ###

    def get_Global_from_Ego_at(self, ts:int, dtype = np.float64) -> np.ndarray:
        warnings.warn("EgomotionProvider.get_Global_from_Ego_at() is deprecated. Use EgomotionProvider.get_transform() instead.", DeprecationWarning)
        return self.get_transform(timestamp = ts).astype(dtype)

    def get_timestamps_range(self) -> np.ndarray:
        warnings.warn("EgomotionProvider.get_timestamps_range() is deprecated. Use EgomotionProvider.get_timestamps() instead (first and last elements).", DeprecationWarning)
        _ts = self.get_timestamps()
        return np.array((_ts[0], _ts[-1]))

    @property 
    def tf_Global_from_EgoZero(self) -> np.ndarray:
        warnings.warn("EgomotionProvider.tf_Global_from_EgoZero is deprecated. Use EgomotionProvider.get_first_transform() instead.", DeprecationWarning)
        return self.get_first_transform()

    def compute_tf_EgoZero_from_Sensor(self, tf_Ego_from_Sensor:np.ndarray, reference_ts:int) -> np.ndarray:
        warnings.warn("EgomotionProvider.compute_tf_EgoZero_from_Sensor(tf_Ego_from_Sensor, reference_ts) is deprecated. \
            Use: EgomotionProvider.get_first_inverse_transform() @ EgomotionProvider.get_transform(reference_ts) @ tf_Ego_from_Sensor ", 
        DeprecationWarning)
        return self.get_first_inverse_transform() @ self.get_transform(reference_ts) @ tf_Ego_from_Sensor

    def compute_trajectory(self, sorted_ts:np.ndarray, tf_Global_from_EgoZero:np.ndarray=np.eye(4), dtype = np.float64, subsampling:int=100) -> np.ndarray:
        warnings.warn("EgomotionProvider.compute_trajectory() is deprecated. Use EgomotionProvider.get_trajectory() instead.", DeprecationWarning)
        return self.get_trajectory(sorted_ts, subsampling=subsampling)

    def apply_trajectory(self, trajectory:np.ndarray, points:np.ndarray, subsampling:int=100) -> np.ndarray:
        warnings.warn("EgomotionProvider.apply_trajectory() is deprecated.", DeprecationWarning)
        corrected = np.empty_like(points)
        n_points = points.shape[0]
        for i in range(trajectory.shape[0]):
            s, e = i * subsampling, min((i+1) * subsampling, n_points)
            corrected[s:e] = linalg.map_points(trajectory[i], points[s:e])
        return corrected
