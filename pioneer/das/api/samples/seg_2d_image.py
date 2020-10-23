from pioneer.das.api                import datatypes
from pioneer.das.api.samples.sample import Sample

import copy
import cv2
import numpy as np

class Seg2dImage(Sample):

    def __init__(self, index, datasource, virtual_raw = None, virtual_ts = None):
        super(Seg2dImage, self).__init__(index, datasource, virtual_raw, virtual_ts)

    def seg2d(self):
        category_img = copy.deepcopy(self.raw)

        seg2d = {'data': np.empty(0, dtype=datatypes.seg2d())}

        for category_number in np.unique(category_img):

            category_array = np.zeros_like(category_img)
            mask = np.where(category_img == category_number)
            category_array[mask] = 1 # confidence 1 is forced here

            seg2d['data'] = np.append(seg2d['data'], np.array([(category_array, category_number)], dtype=datatypes.seg2d()))

        return seg2d


    def poly2d(self, confidence_threshold=0.5):

        seg2d = self.seg2d()

        poly2d = np.empty(0, dtype=datatypes.poly2d())
        for seg_data in seg2d['data']:
            mask = (seg_data['confidences']>0).astype(np.uint8)
            if mask.max()==0:
                continue
            polys,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polys = np.array(polys)
            for poly in polys:
                poly = poly[:,0,:]
                poly2d = np.append(poly2d, np.array([(poly, seg_data['classes'],0,0)], dtype=datatypes.poly2d()))

        return poly2d
