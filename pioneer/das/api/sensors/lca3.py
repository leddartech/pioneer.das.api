from pioneer.das.api.sensors.lcax import LCAx

import numpy as np
        
class LCA3(LCAx):
    def __init__(self, name, platform):
        super(LCA3, self).__init__(name, platform)
        self.scan_direction = 'horizontal'
        #90 deg around 'z' axis
        self.orientation = np.array([[ 0,  1,  0],
                                        [-1,  0,  0],
                                        [ 0,  0,  1]], 'f8')

    def get_trace_smoothing_kernel(self):
        """Get the convolution kernel for trace processing for Eagle"""
        # Temporary fix for binning while we can't read the binning parameter anywhere
        kernel = np.array([1,9,18,23,26,23,8,-7,-14,-17,-17,-15,-13,-11,-10,-8,-8,-6,-5,-4,-3,-3,-3,-2,-2])[::-1]/256
        if self.binning == 2:
            kernel = np.array([1,18,26,8,-14,-17,-13,-10,-8,-5,-3,-3,-2])[::-1]/128
        return kernel
