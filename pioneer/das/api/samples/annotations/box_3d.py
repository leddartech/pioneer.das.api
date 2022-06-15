from pioneer.common import platform, linalg, IoU3d
from pioneer.common.logging_manager import LoggingManager
from pioneer.common import platform as platform_utils
from pioneer.das.api import categories
from pioneer.das.api.samples.sample import Sample

from transforms3d import euler
from typing import Iterable, Optional

import copy
import numpy as np
import warnings
warnings.simplefilter('once', DeprecationWarning)


class Box3d(Sample):

    def __init__(self, index, datasource, virtual_raw = None, virtual_ts = None):
        super(Box3d, self).__init__(index, datasource, virtual_raw, virtual_ts)
        self._label_source = categories.get_source(platform_utils.parse_datasource_name(self.datasource.label)[2])

    def get_centers(self) -> np.ndarray:
        return self.raw['data']['c']

    def get_dimensions(self) -> np.ndarray:
        return self.raw['data']['d']

    def get_rotations(self) -> np.ndarray:
        return self.raw['data']['r']

    def get_confidences(self) -> Iterable[Optional[float]]:
        return self.raw.get('confidence', [None for _ in range(len(self))])

    def get_category_numbers(self) -> Iterable[int]:
        return self.raw['data']['classes']

    def get_categories(self) -> Iterable[str]:
        categorie_labels = []
        for category_number in self.get_category_numbers():
            labels = categories.CATEGORIES[self._label_source]
            label = labels.get(str(category_number), {'name':'?'})['name']
            categorie_labels.append(label)
        return categorie_labels

    def get_ids(self) -> Iterable[Optional[int]]:
        return self.raw['data']['id']

    def __len__(self) -> int:
        return self.get_centers().shape[0]

    def set_transform(self, transform:np.ndarray) -> 'Box3d':
        if transform is None: return self

        centers = self.get_centers()
        rotations = self.get_rotations()

        raw = copy.deepcopy(self.raw)

        for i in range(len(self)):
            box_transform = np.eye(4)
            box_transform[:3, :3] = euler.euler2mat(*rotations[i])
            box_transform[:3, 3] = centers[i,:]
            box_to_transform = transform @ box_transform
            raw['data']['c'][i,:] = box_to_transform[:3, 3]
            raw['data']['r'][i,:] = euler.mat2euler(box_to_transform)

        return Box3d(self.index, self.datasource, raw, self.timestamp)

    def set_referential(self, referential:str, ignore_orientation:bool=False, reference_ts:int=-1, dtype=np.float64) -> 'Box3d':
        referential = platform.referential_name(referential)
        if self.datasource.sensor.name == referential: return self
        transform = self.compute_transform(referential, ignore_orientation, reference_ts, dtype)
        return self.set_transform(transform)
    

    ### Legacy section ###

    def attributes(self):
        warnings.warn("Box3d.attributes() is deprecated.", DeprecationWarning)
        if 'attributes' in self.raw:
            return self.raw['attributes']
        LoggingManager.instance().warning(f"There are no 'attributes' for that sample {self.datasource.label}.")
        return None
    
    def dynamics(self):
        warnings.warn("Box3d.dynamics() is deprecated.", DeprecationWarning)
        if 'dynamics' in self.raw:
            return self.raw['dynamics']
        LoggingManager.instance().warning(f"There are no 'dynamics' for that sample {self.datasource.label}.")
        return None

    def confidences(self):
        warnings.warn("Box3d.confidences() is deprecated, use Box3d.get_confidences() instead.", DeprecationWarning)
        if 'confidence' not in self.raw: return
        return self.get_confidences()

    def label_names(self):
        warnings.warn("Box3d.label_names() is deprecated, use Box3d.get_categories() instead.", DeprecationWarning)
        return self.get_categories()

    def _mapto(self, tf=None):
        """Maps the box3d to the new referential, given a 4x4 matrix transformation"""
        warnings.warn("Box3d._mapto() is deprecated, use Box3d.set_transform() instead.", DeprecationWarning)
        return self.set_transform(tf).raw['data']
    
    def mapto(self, referential_or_ds:str=None, ignore_orientation:bool=False, reference_ts:int=-1, dtype=np.float64):
        """ Will map each box in another referential. """
        warnings.warn("Box3d.mapto() is deprecated, use Box3d.set_referential() instead.", DeprecationWarning)
        return self.set_referential(referential_or_ds, ignore_orientation, reference_ts, dtype).raw['data']

    def num_pts_in(self, pt_cloud, margin=0):
        """ Returns, for each box, the mask of those points from pt_cloud that are inside the box.
            Args:
                pt_cloud - (M,3)
                margin (optional) - positive float- increases the size of box

            Returns:
                mask - boolean (n_boxe,M)
        """
        warnings.warn("Box3d.num_pts_in() is deprecated.", DeprecationWarning)
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
        warnings.warn("Box3d.set_angle_to_domain() is deprecated.", DeprecationWarning)

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
        warnings.warn("Box3d.compute_iou() is deprecated.", DeprecationWarning)
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