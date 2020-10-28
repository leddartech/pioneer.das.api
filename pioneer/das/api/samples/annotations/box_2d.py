from pioneer.das.api.samples.sample import Sample

class Box2d(Sample):

    def __init__(self, index, datasource, virtual_raw = None, virtual_ts = None):
        super(Box2d, self).__init__(index, datasource, virtual_raw, virtual_ts)
