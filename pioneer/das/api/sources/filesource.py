from pioneer.das.api.datasources import DataSource
from pioneer.das.api.loaders import pickle_loader, txt_loader, image_loader

from six import string_types

def _endswith(s, patterns):
    assert isinstance(s, string_types)
    if isinstance(patterns, string_types):
        patterns = [patterns,]
    return any([s.endswith(p) for p in patterns])

class FileSource(DataSource):

    """Base class for loading files from a dataset. Will be used to
    load a list of echoes from pickle files contained in a folder or a tar
    archive. Subclases must be used.
    """

    def __init__(self, path, pattern = None, sort=True, loader=None):
        self.path = path
        self.pattern = pattern
        self.sort = sort
        self._loader = loader
        self.files = []

    @property
    def loader(self):
        if self._loader is None:
            if _endswith(self.pattern, '.pkl'):
                self._loader = pickle_loader
            elif _endswith(self.pattern, ['.png', '.jpg']):
                self._loader = image_loader
            elif _endswith(self.pattern, '.txt'):
                self._loader = txt_loader
            else:
                raise RuntimeError('could not guess loader from pattern extension')
        return self._loader

    def get_timestamps(self):
        raise NotImplementedError()
