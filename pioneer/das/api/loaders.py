import cv2
import os
import numpy as np
import pickle
import re
import six


def load_files_from_folder(folder, pattern, sort=False, return_keys=False):
    """Load files from a folder and sort them according to a regex pattern"""
    if return_keys:
        assert sort, 'sort must be True when return_keys is True'

    files = os.listdir(folder)
    regex = re.compile(pattern)
    output = []
    for f in files:
        match = regex.match(f)
        if match:
            groups = match.groups()
            if sort and not groups:
                raise ValueError('no groups')
            if sort and groups:
                sample = (int(groups[0]), os.path.join(folder, f))
            else:
                sample = os.path.join(folder, f)

            output.append(sample)
    if sort:
        output.sort()
        keys = [s[0] for s in output]
        output = [s[1] for s in output]
    if return_keys and sort:
        return output, keys
    else:
        return output


def txt_loader(fileobj):
    """Bytes data file loader. To be used with FileSource.

    Arguments:
        fileobj {string,file} -- A file object or a string

    Returns:
        bytes
    """
    if isinstance(fileobj, bytes):
        data = fileobj.decode('utf-8')
    elif isinstance(fileobj, six.string_types):
        with open(fileobj, 'rb') as f:
            data = f.read().decode('utf-8')
    elif hasattr(fileobj, 'read'):
        data = fileobj.read().decode('utf-8')
    else:
        raise ValueError('fileobj is not a filename or a file object')
    return data

def pickle_loader(fileobj):
    """Pickle data file loader. To be used with FileSource.

    Arguments:
        fileobj {string,file} -- A file object or a string

    Returns:
        object -- Unpickled object
    """
    if isinstance(fileobj, bytes):
        data = pickle.loads(fileobj, encoding="latin1")
    elif isinstance(fileobj, six.string_types):
        with open(fileobj, 'rb') as f:
            data = pickle.load(f, encoding="latin1")
    elif hasattr(fileobj, 'read'):
        data = pickle.load(fileobj, encoding="latin1")
    else:
        raise ValueError('fileobj is not a filename or a file object')
    return data

def image_loader(fileobj):
    """Image loader to be used with FileSource

    Arguments:
        fileobj {string} -- Image filename

    Returns:
        np.ndarray -- Image
    """
    if isinstance(fileobj, six.string_types):
        return cv2.imread(fileobj, cv2.IMREAD_COLOR)[..., ::-1] #bgr->rgb
    elif isinstance(fileobj, bytes):
        byte_arr = bytearray(fileobj)
    else:
        byte_arr = bytearray(fileobj.read())
    
    return cv2.imdecode(np.asarray(byte_arr, dtype=np.uint8), cv2.IMREAD_COLOR)[..., ::-1] #bgr->rgb
