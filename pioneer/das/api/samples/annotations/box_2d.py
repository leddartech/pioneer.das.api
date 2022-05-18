from pioneer.das.api.samples.sample import Sample
from pioneer.das.api import categories
from pioneer.common import platform as platform_utils

from typing import Iterable, Optional

import numpy as np
import warnings
warnings.simplefilter('once', DeprecationWarning)


class Box2d(Sample):
    def __init__(self, index, datasource, virtual_raw = None, virtual_ts = None):
        super(Box2d, self).__init__(index, datasource, virtual_raw, virtual_ts)

    def get_centers(self) -> np.ndarray:
        return np.vstack((self.raw['data']['x'], self.raw['data']['y'])).T

    def get_dimensions(self) -> np.ndarray:
        return np.vstack((self.raw['data']['h'], self.raw['data']['w'])).T

    def get_confidences(self) -> Iterable[Optional[float]]:
        return self.raw.get('confidence', [None for _ in range(len(self))])

    def get_category_numbers(self) -> Iterable[int]:
        return self.raw['data']['classes']

    def get_categories(self) -> Iterable[str]:
        label_source_name = categories.get_source(platform_utils.parse_datasource_name(self.datasource.label)[2])
        return [categories.CATEGORIES[label_source_name][str(category_number)]['name'] for category_number in self.get_category_numbers()]

    def get_ids(self) -> Iterable[Optional[int]]:
        return self.raw['data']['id']

    def __len__(self) -> int:
        return self.get_centers().shape[0]

    def label_names(self):
        warnings.warn("Box2d.label_names() is deprecated, use Box2d.get_categories() instead.", DeprecationWarning)
        return self.get_categories()
