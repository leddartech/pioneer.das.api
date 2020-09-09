class DataSource(object):

    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()
