from pioneer.common                  import platform, clouds
from pioneer.das.api                 import categories
from pioneer.das.api.samples.sample  import Sample

import numpy as np

class Seg3d(Sample):

    def __init__(self, index, datasource, virtual_raw = None, virtual_ts = None):
        super(Seg3d, self).__init__(index, datasource, virtual_raw, virtual_ts)
            
    def colors(self, mode=None):

        _,_,ds_type = platform.parse_datasource_name(self.datasource.label)
        seg_source = categories.get_source(ds_type)
        classes = self.raw['data']['classes']

        colors = np.zeros((len(classes),4))

        for c in np.unique(classes):
            name, color = categories.get_name_color(seg_source, c)
            color = np.array(color)/255. # color channels between 0 and 1
            color = np.append(color, 1) # alpha (opacity) = 1
            ind = np.where(classes == c)[0]
            colors[ind] = color

        if mode=='quad_cloud':
            colors = clouds.quad_stack(colors)

        return colors

    def mask_category(self, category):
        classes = self.raw['data']['classes']

        if type(category) is int:
            return classes == category

        elif type(category) is str:
            _,_,ds_type = platform.parse_datasource_name(self.datasource.label)
            seg_source = categories.get_source(ds_type)
            try:
                category_number = categories.get_category_number(seg_source, category)
            except:
                category_number = -1
            return classes == category_number
        
        else:
            raise ValueError('The category must be either an integer or a string.')
            
          