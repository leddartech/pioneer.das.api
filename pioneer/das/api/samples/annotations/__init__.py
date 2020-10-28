from pioneer.das.api.interpolators import nearest_interpolator
from pioneer.das.api.samples.annotations.box_3d import Box3d
from pioneer.das.api.samples.annotations.box_2d import Box2d
from pioneer.das.api.samples.annotations.poly_2d import Poly2d
from pioneer.das.api.samples.annotations.seg_2d import Seg2d
from pioneer.das.api.samples.annotations.seg_2d_image import Seg2dImage
from pioneer.das.api.samples.annotations.seg_3d import Seg3d
from pioneer.das.api.samples.annotations.lane import Lane


ANNOTATIONS_FACTORY = {
    'box3d': (Box3d, nearest_interpolator),
    'box2d': (Box2d, nearest_interpolator),
    'poly2d': (Poly2d, nearest_interpolator),
    'seg2d': (Seg2d, nearest_interpolator),
    'seg2dimg': (Seg2dImage, nearest_interpolator),
    'seg3d': (Seg3d, nearest_interpolator),
    'lane': (Lane, nearest_interpolator),
}