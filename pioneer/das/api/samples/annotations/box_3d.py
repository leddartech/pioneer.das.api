from pioneer.common import platform, linalg, IoU3d
from pioneer.common.logging_manager import LoggingManager
from pioneer.common import platform as platform_utils
from pioneer.das.api import categories
from pioneer.das.api.samples.sample import Sample

from transforms3d import euler
import numpy as np


class Box3d(Sample):

    def __init__(self, index, datasource, virtual_raw = None, virtual_ts = None):
        super(Box3d, self).__init__(index, datasource, virtual_raw, virtual_ts)

    def label_names(self):
        '''Converts the category numbers in their corresponding names (e.g. 0 -> 'pedestrian') and returns the list of names for all boxes in the sample'''
        label_source_name = categories.get_source(platform_utils.parse_datasource_name(self.datasource.label)[2])
        try:
            return [categories.CATEGORIES[label_source_name][str(category_number)]['name'] for category_number in self.raw['data']['classes']]
        except:
            LoggingManager.instance().warning(f"Can not find the CATEGORIES and NAMES of {label_source_name}.")

    def _mapto(self, tf=None):
        """Maps the box3d to the new referential, given a 4x4 matrix transformation"""

        bbox = np.copy(self.raw['data'])
        if tf is None:
            return bbox
        
        for i in range(len(bbox)):
            tf_Localds_from_Box = np.eye(4)
            tf_Localds_from_Box[:3, :3] = euler.euler2mat(bbox['r'][i,0], bbox['r'][i,1], bbox['r'][i,2])
            tf_Localds_from_Box[:3, 3] = bbox['c'][i,:]
            tf_Referential_from_Box = tf @ tf_Localds_from_Box
            bbox['c'][i,:] = tf_Referential_from_Box[:3, 3]
            bbox['r'][i,:] = euler.mat2euler(tf_Referential_from_Box)

        return bbox
    
    def mapto(self, referential_or_ds:str=None, ignore_orientation:bool=False, reference_ts:int=-1, dtype=np.float64):
        """ Will map each box in another referential. """

        referential = platform.referential_name(referential_or_ds)
        if self.datasource.sensor.name == referential:
            tf_Referential_from_Localds = None
        else:
            tf_Referential_from_Localds = self.compute_transform(referential_or_ds, ignore_orientation, reference_ts, dtype)

        return self._mapto(tf_Referential_from_Localds)

    def attributes(self):
        if 'attributes' in self.raw:
            return self.raw['attributes']
        LoggingManager.instance().warning(f"There are no 'attributes' for that sample {self.datasource.label}.")
        return None
    
    def confidences(self):
        if 'confidence' in self.raw:
            return self.raw['confidence']
        LoggingManager.instance().warning(f"There are no 'confidences' for that sample {self.datasource.label}.")
        return None

    def num_pts_in(self, pt_cloud, margin=0):
        """ Returns, for each box, the mask of those points from pt_cloud that are inside the box.
            Args:
                pt_cloud - (M,3)
                margin (optional) - positive float- increases the size of box

            Returns:
                mask - boolean (n_boxe,M)
        """
        bbox = self.raw['data']
        nbpts = np.zeros((len(bbox), len(pt_cloud)), dtype=bool)
        for i in range(len(bbox)):
            tf_Localds_from_Box = np.eye(4)
            tf_Localds_from_Box[:3, :3] = euler.euler2mat(bbox['r'][i,0], bbox['r'][i,1], bbox['r'][i,2])
            tf_Localds_from_Box[:3, 3] = bbox['c'][i,:]
            aabb = np.vstack([-(bbox['d'][i,:]+margin) / 2.0,(bbox['d'][i,:]+margin) / 2.0])
            nbpts[i,:] = linalg.points_inside_box_mask(pt_cloud, aabb, linalg.tf_inv(tf_Localds_from_Box))
        return nbpts

    def set_angle_to_domain(self, domain=[0,2*np.pi]):
        """Will set the angles to a given domain"""

        bbox = np.copy(self.raw['data'])
        for i in range(len(bbox)):
            bbox['r'][i,:] = [linalg.map_angle_to_domain(bbox['r'][i,j], domain=domain) for j in range(3)]

        return bbox
    
    def compute_iou(self, box, return_max=False, map2yaw=None):
        """Compute the iou score between all the elements of self and of box.
            
            Return a matrix len(self), len(box) when row,col are indexed in the same order as self, box.

            If return_max=True: return only a single number for each element of self (the max value).

            Important note: By default the computation is performed in the sbg ref where only one angle (yaw) is not zero, unless
            map2yaw is provided (a callable) which brings all the boxes in a referential where only one rotation (yaw).

        """
        if map2yaw is not None:
            box0 = map2yaw(self)
            box1 = map2yaw(box)
        else:
            try: #must find either sbg or ENU.
                referential = platform.referential_name('sbgekinox_bcc')
                tf_TargetRef_from_Local = np.copy(self.datasource.sensor.map_to(referential))
                tf_TargetRef_from_Local[:3, :3] = tf_TargetRef_from_Local[:3, :3] @ self.orientation

            except:
                LoggingManager.instance().warning('IoU computation, falling back to ENU system transformation.')
                tf_TargetRef_from_Local = np.eye(4)
                tf_TargetRef_from_Local[:3, :3] = np.array([[0, 0, 1],
						                                    [-1, 0, 0],
						                                    [0, -1, 0]],
						                                    dtype=np.float).T
            box0 = self._mapto(tf=tf_TargetRef_from_Local)
            box1 = box._mapto(tf=tf_TargetRef_from_Local)
        
        Z0 = [box0['c'], box0['d'], 'z', box0['r'][:,2]]
        Z1 = [box1['c'], box1['d'], 'z', box1['r'][:,2]]
        matiou = IoU3d.matrixIoU(Z0=Z0, Z1=Z1)
        if return_max:
            return np.max(matiou, axis=1)
        else:
            return matiou