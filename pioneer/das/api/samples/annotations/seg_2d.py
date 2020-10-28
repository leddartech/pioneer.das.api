from pioneer.das.api import datatypes
from pioneer.das.api.samples.sample import Sample

import copy
import cv2
import numpy as np

class Seg2d(Sample):

    def __init__(self, index, datasource, virtual_raw = None, virtual_ts = None):
        super(Seg2d, self).__init__(index, datasource, virtual_raw, virtual_ts)

    def poly2d(self, confidence_threshold=0.5):

        raw = copy.deepcopy(self.raw)

        poly2d = np.empty(0, dtype=datatypes.poly2d())
        for seg_data in raw['data']:
            mask = (seg_data['confidences']>confidence_threshold).astype(np.uint8)
            if mask.max()==0:
                continue
            polys,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polys = np.array(polys)
            for poly in polys:
                poly = poly[:,0,:]
                poly2d = np.append(poly2d, np.array([(poly,seg_data['classes'],0,0)], dtype=datatypes.poly2d()))

        return poly2d
