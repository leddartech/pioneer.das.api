import numpy as np

def from_spherical_to_cartesian(distance, 
                                    elevation, 
                                    azimut):
    l = distance * np.cos(elevation)
    return (
        l * np.cos(azimut), 
        l * np.sin(azimut),
        distance * np.sin(elevation) 
    )

def referential_transform(distance,
                            elevation,
                            azimut,
                            a,
                            b,
                            c):
    '''Return mappeds (distance, elevation, azimut) in a new referential.
        The parameters (a,b,c) are some constants.
    '''
    x_, y_, z_ = from_spherical_to_cartesian(distance, elevation, azimut)
    x_ += a
    y_ += b
    z_ += c
    return (
        (x_**2 + y_**2 + z_**2)**0.5,
        np.arctan2(z_, (x_**2 + y_**2)**0.5),
        np.arctan2(y_, x_)
    )

def from_calibration_jig_to_sensor(pitch, 
                                    yaw, 
                                    pivot):
    '''Input data from the calibration jig are converted to sensor data at 
        universal point referential (elevation, azimut)
    '''
    return (
        np.arctan2(np.sin(pitch) * np.cos(yaw - pivot) , (np.sin(yaw - pivot)**2 + np.cos(pitch)**2 * np.cos(yaw - pivot)**2)**0.5),
        np.arctan2(np.tan(yaw) , np.cos(pitch))
    )

def from_sensor_to_lcas(distance, 
                            elevation, 
                            azimut, 
                            dz_elevation, 
                            dx_azimut, 
                            dy_azimut):
    '''Convert a coordinate (distance, elevation, azimut) from universal
        sensor point to lcas sub-module.
        
        Constants:
            dz_elevation: is the distance on the z-axis (or along elevation) between sensor point and lcas point
            dx_azimut: distance on the x-axis between referentials
            dy_azimut: distance on the y-axis between referentials
    '''
    return referential_transform(distance,
                            elevation,
                            azimut,
                            -dx_azimut,
                            -dy_azimut,
                            -dz_elevation)

def from_lcas_to_sensor(distance, 
                            elevation, 
                            azimut, 
                            dz_elevation, 
                            dx_azimut, 
                            dy_azimut):
    '''Convert a coordinate (distance, elevation, azimut) from submodule
        lcas point to unviversal sensor point.

        Constans: (see: from_sensor_to_lcas)
    '''
    return referential_transform(distance,
                            elevation,
                            azimut,
                            dx_azimut,
                            dy_azimut,
                            dz_elevation)

def sensor_angles_from_lcas_angles(distance,
                                    elevation,
                                    azimut,
                                    dx_azimut,
                                    dy_azimut):
    '''Compute (elevation, azimut) in sensor referential from
            distance: from sensor referential
            elevation: from lcas referential
            azimut: from lcas referential
        and
        Constants: (see from_sensor_to_lcas)
    '''
    gamma_ = np.cos(elevation) * (dy_azimut * np.sin(azimut) + dx_azimut * np.cos(azimut))
    distance_ = - gamma_ + (gamma_**2  + distance**2 - dx_azimut**2 - dy_azimut**2)**0.5
    _, elevation_, azimut_ = from_lcas_to_sensor(distance_, 
                                                    elevation, 
                                                    azimut, 
                                                    0, 
                                                    dx_azimut, 
                                                    dy_azimut)
    return elevation_, azimut_