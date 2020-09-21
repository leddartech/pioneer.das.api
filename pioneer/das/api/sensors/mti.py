from pioneer.das.api.interpolators import linear_dict_of_float_interpolator, linear_ndarray_interpolator
from pioneer.das.api.samples import Sample
from pioneer.das.api.sensors.sensor import Sensor


class MTi(Sensor):
    def __init__(self, name, platform):
        if not platform.is_live():
            factories = {'ago':(Sample, linear_ndarray_interpolator)}
        else:
            factories = {'ago':(Sample, linear_dict_of_float_interpolator)}
        super(MTi, self).__init__(name, platform, factories)