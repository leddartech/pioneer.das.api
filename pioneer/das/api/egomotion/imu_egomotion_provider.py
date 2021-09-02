from pioneer.common                                import linalg
from pioneer.das.api.datasources                   import AbstractDatasource
from pioneer.das.api.egomotion.egomotion_provider  import EgomotionProvider

import numpy as np
from scipy.interpolate import splprep, splev

class IMUEgomotionProvider(EgomotionProvider):
    """Ego Motion provider for datasets from the RAV-4 platform"""

    def __init__(self, referential_name:str, navposvel:AbstractDatasource, ekfeuler:AbstractDatasource, subsampling:int=100):
        super(IMUEgomotionProvider, self).__init__(referential_name, subsampling)
        self.navposvel = navposvel
        self.ekfeuler = ekfeuler
        self._tf_Global_from_EgoZero = self.get_Global_from_Ego_at(self.navposvel.timestamps[0])

    def get_Global_from_Ego_at(self, ts:int, dtype = np.float64) -> np.ndarray:
        """override"""
        nav = self.navposvel.get_at_timestamp(ts).raw  # latitude, longitude, altitude
        euler = self.ekfeuler.get_at_timestamp(ts).raw # roll, pitch, yaw
        return linalg.tf_from_poseENU(linalg.imu_to_utm(nav, euler), dtype = dtype) 

    def get_timestamps_range(self) -> np.ndarray:
        """override"""
        ts = self.navposvel.timestamps
        return np.array([ts[0], ts[-1]])

    def trace_sampled_trajectory(self, sampling_distance:float=1.0, N:int=None) -> np.ndarray:
        xyz = np.empty((0,3))
        for ts in self.navposvel.timestamps:
            xyz = np.vstack([xyz, self.get_Global_from_Ego_at(ts)[:3,3]])

        distances = ((xyz[1:,0] - xyz[:-1,0])**2 + (xyz[1:,1] - xyz[:-1,1])**2)**0.5
        distances = np.insert(distances, 0, 0)
        total_distance = np.sum(distances)

        N = int(total_distance / sampling_distance) + 1 if N is None else N
        
        # To prevent a crash from scipy's splprep, we have to remove any identical consecutive vertices
        keep = np.where(~((xyz[:-1,0] == xyz[1:,0]) & (xyz[:-1,1] == xyz[1:,1])))[0] + 1
        keep = np.insert(keep, 0, 0)

        tck, u = splprep([xyz[keep,0], xyz[keep,1]])
        u = np.linspace(0,1,N)
        return np.array(splev(u, tck)).T