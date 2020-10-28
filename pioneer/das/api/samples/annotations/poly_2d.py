from pioneer.common import platform
from pioneer.das.api import categories
from pioneer.das.api.samples.sample import Sample

import cv2
import numpy as np

class Poly2d(Sample):

    def __init__(self, index, datasource, virtual_raw = None, virtual_ts = None):
        super(Poly2d, self).__init__(index, datasource, virtual_raw, virtual_ts)

    def colored_image(self, resolution:tuple=None):
        polygons = self.raw
        _,_,ds_type = platform.parse_datasource_name(self.datasource.label)
        poly_source = categories.get_source(ds_type)
        image = np.zeros((polygons['resolution'][0],polygons['resolution'][1],3), dtype=np.uint8)
        
        for poly in polygons['data']:
            name, color = categories.get_name_color(poly_source,poly['classes'])
            color = np.array(color)/255
            cv2.fillPoly(image, [poly['polygon']], color)

        return self.resize_mask(image, resolution) if resolution is not None else image

    def mask_category(self, category:str, resolution:tuple=None, confidence_threshold:float=0.5):
        polygons = self.raw
        _,_,ds_type = platform.parse_datasource_name(self.datasource.label)
        poly_source = categories.get_source(ds_type)
        mask = np.zeros((polygons['resolution'][0],polygons['resolution'][1]), dtype=np.uint8)
        
        for i, poly in enumerate(polygons['data']):

            if 'confidence' in polygons:
                if polygons['confidence'][i] < confidence_threshold:
                    continue

            name, color = categories.get_name_color(poly_source,poly['classes'])
            if name == category:
                cv2.fillPoly(mask, [poly['polygon']], 1)

        return self.resize_mask(mask, resolution) if resolution is not None else mask

    def areas(self):
        areas = np.empty(0)
        for poly in self.raw['data']:
            areas = np.append(areas, cv2.contourArea(poly['polygon']))
        return areas

    @staticmethod
    def resize_mask(mask, resolution):
        return cv2.resize(mask, (resolution[1],resolution[0]), interpolation = cv2.INTER_NEAREST)
