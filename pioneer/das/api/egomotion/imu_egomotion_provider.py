from pioneer.common                                import linalg
from pioneer.das.api.datasources                   import DatasourceWrapper
from pioneer.das.api.egomotion.egomotion_provider  import EgomotionProvider

import numpy as np
from scipy.interpolate import splprep, splev
import warnings
warnings.simplefilter('once', DeprecationWarning)



class IMUEgomotionProvider(EgomotionProvider):
    """Ego Motion provider for datasets from the RAV-4 platform"""

    def __init__(self, referential_name:str, navposvel:DatasourceWrapper, ekfeuler:DatasourceWrapper):
        super().__init__(referential_name)
        self.navposvel = navposvel
        self.ekfeuler = ekfeuler

    def get_transform(self, timestamp:int) -> np.ndarray:
        nav = self.navposvel.get_at_timestamp(timestamp).raw  # latitude, longitude, altitude
        euler = self.ekfeuler.get_at_timestamp(timestamp).raw # roll, pitch, yaw
        return linalg.tf_from_poseENU(linalg.imu_to_utm(nav, euler), dtype = np.float64) 

    def get_timestamps(self) -> np.ndarray:
        return self.navposvel.timestamps


    ### Legacy section ###

    def trace_sampled_trajectory(self, sampling_distance:float=1.0, N:int=None) -> np.ndarray:
        warnings.warn("IMUEgomotionProvider.trace_sampled_trajectory() is deprecated.", DeprecationWarning)
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
