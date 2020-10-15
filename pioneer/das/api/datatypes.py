import numpy as np

def box2d():
    return np.dtype([('x', 'f4') #vertical coordinate from top to bottom
                    , ('y', 'f4') #horizontal coordinate from left to right
                    , ('h', 'f4') #height (vertical)
                    , ('w', 'f4') #width (horizontal)
                    , ('r','f4') #rotation (anti-clockwise)
                    , ('classes', 'u2')
                    , ('id', 'u2')
                    , ('flags', 'u2')])

def box3d():
    ''' Here are the standard coordinates for 3d bounding boxes

    y_ENU
    ^                 x_Robot (forward, maps bbox lenght)
    |                 /
    |    y_Robot (sideways, maps bbox width)
    |         '     /   
    |           '  /   
    |             o   
    |               
    |               
    | 
    o----------->x_ENU
    z_ENU
    '''
    return np.dtype([('c', 'f4', (3)) #center (x,y,z)
                    , ('d', 'f4', (3)) #dimensions (lx, ly, lz) typically mapped to (length, width, height)
                    , ('r','f4', (3)) #euler angles (rx, ry, rz)
                    , ('classes', 'u2')
                    , ('id', 'u2')
                    , ('flags', 'u2')])

def seg3d():
    return np.dtype([('classes', 'u2'), ('id', 'u2')])

def poly2d():
    return np.dtype([('polygon','O') #(N,2) array with N vertices and 2 x,y coordinates (in pixel units)
                    ,('classes','u2')
                    ,('id','u2')
                    ,('flags','u2')])

def seg2d():
    return np.dtype([('confidences','O') #Array of values between 0 and 1. Dimensions = reference sensor's resolution. 
                    ,('classes','u2')])


def datasource_xyzit():
    return np.dtype([('x','f4'),('y','f4'),('z','f4'),('i','u2'),('t','u8')])
    
def datasource_xyzit_float_intensity():
    return np.dtype([('x','f4'),('y','f4'),('z','f4'),('i','f4'),('t','u8')])

def rad():
    return np.dtype([('x','f4'),('y','f4'),('v','f4'),('i','f4'),('t','u8')])

def lane():
    """
    vertices: (N,3) numpy array of the 3D coordinates
    type: number to identify the type of lane (see das/lane_type.py)
    """
    return np.dtype([('vertices','O'),('type','u4')])