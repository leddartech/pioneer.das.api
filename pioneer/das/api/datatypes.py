import numpy as np

def box2d():
    '''Fields are: 
        'x' (float): the top coordinate of the box in pixels
        'y' (float): the left coordinate of the box in pixels
        'h' (float): box height in pixels
        'w' (float): box width in pixels
        'r' (float): box rotation angle (anti-clockwise)
        'classes' (int): the object category number
        'id' (int): the object's instance unique id
        'flags' (int): miscellaneous infos
    '''
    return np.dtype([
        ('x', 'f4'),
        ('y', 'f4'),
        ('h', 'f4'),
        ('w', 'f4'),
        ('r','f4'),
        ('classes', 'u2'),
        ('id', 'u2'),
        ('flags', 'u2')
    ])

def box3d():
    '''Fields are: 
        'c' (float, float, float): center coordinates
        'd' (float, float, float): the box dimensions
        'r' (float, float, float): the Euler angles (rotations)
        'classes' (int): the object category number
        'id' (int): the object's instance unique id 
        'flags' (int): miscellaneous infos
    Coordinate system: +x is forward, +y is left and +z is up. 
    '''
    return np.dtype([
        ('c', 'f4', (3)),
        ('d', 'f4', (3)),
        ('r','f4', (3)),
        ('classes', 'u2'),
        ('id', 'u2'),
        ('flags', 'u2'),
    ])

def attributes():
    return np.dtype([
        ('occlusions', 'u2'),
        ('truncations', 'u2'),
        ('on the road','?'),
        ('vehicle activities', 'O'),
        ('human activities', 'O')
    ])

def seg3d():
    '''Fields are: 
        'classes' (int): the object category number
        'id' (int): the object's instance unique id 
    Note: A 3D segmentation array must have the same length and ordering as the corresponding point cloud array.
    '''
    return np.dtype([('classes', 'u2'), ('id', 'u2')])

def poly2d():
    '''Fields are: 
        'polygon': an (N,2) array with N vertices and 2 x,y coordinates (in pixel units)
        'classes' (int): the object category number
        'id' (int): the object's instance unique id
        'flags' (int): miscellaneous infos
    '''
    return np.dtype([
        ('polygon','O'),
        ('classes','u2'),
        ('id','u2'),
        ('flags','u2'),
    ])

def seg2d():
    '''Fields are: 
        'confidences': 2D array of values between 0 and 1
        'classes' (int): the object category number
    Note: the size of the 2D arrays should be the same as the sensor's resolution
    '''
    return np.dtype([
        ('confidences','O'),
        ('classes','u2'),
    ])

def datasource_xyzit():
    '''Fields are: 
        'x' (float): x coordinate (forward)
        'y' (float): y coordinate (left)
        'z' (float): z coordinate (up)
        'i' (int): intensity
        't' (int): timestamp
    '''
    return np.dtype([('x','f4'),('y','f4'),('z','f4'),('i','u2'),('t','u8')])
    
def datasource_xyzit_float_intensity():
    '''Fields are: 
        'x' (float): x coordinate (forward)
        'y' (float): y coordinate (left)
        'z' (float): z coordinate (up)
        'i' (float): intensity
        't' (int): timestamp
    '''
    return np.dtype([('x','f4'),('y','f4'),('z','f4'),('i','f4'),('t','u8')])

def lane():
    """Fields are: 
        'vertices': (N,3) numpy array of the 3D coordinates
        'type' (int): number to identify the type of lane (see das/lane_type.py)
    """
    return np.dtype([('vertices','O'),('type','u4')])