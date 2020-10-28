from pioneer.das.api.samples.sample import Sample


class Lane(Sample):
    def __init__(self, index, datasource, virtual_raw = None, virtual_ts = None):
        super(Lane, self).__init__(index, datasource, virtual_raw, virtual_ts)  