from pioneer.das.api.samples.sample import Sample
from pioneer.das.api import categories
from pioneer.common import platform as platform_utils
from pioneer.common.logging_manager import LoggingManager


class Box2d(Sample):
    def __init__(self, index, datasource, virtual_raw = None, virtual_ts = None):
        super(Box2d, self).__init__(index, datasource, virtual_raw, virtual_ts)

    def label_names(self):
        '''Converts the category numbers in their corresponding names (e.g. 0 -> 'pedestrian') and returns the list of names for all boxes in the sample'''
        label_source_name = categories.get_source(platform_utils.parse_datasource_name(self.datasource.label)[2])
        try:
            return [categories.CATEGORIES[label_source_name][str(category_number)]['name'] for category_number in self.raw['data']['classes']]
        except:
            LoggingManager.instance().warning(f"Can not find the CATEGORIES and NAMES of {label_source_name}.")
