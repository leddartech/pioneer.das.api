import numbers
import numpy as np

def from_float_index(float_index):
    i = int(np.floor(float_index))
    t = float_index - i
    return i, t

def _linear_interpolate_helper(t, from_val, to_val):
    return t * to_val + (1 - t) * from_val

def _angles_linear_interpolate_helper(t, from_val, to_val):
    if from_val-to_val>np.pi:
        return t * (to_val+2.0*np.pi) + (1 - t) * from_val
    elif from_val-to_val<-np.pi:
        return t * to_val + (1 - t) * (from_val+2.0*np.pi)
    else:
        return t * to_val + (1 - t) * from_val


def linear_ndarray_interpolator(datasource, float_index):
    i, t = from_float_index(float_index)
    if i+1 != len(datasource):
        raw_from, raw_to = datasource[i].raw, datasource[i+1].raw
    else:
        raw_from, raw_to = datasource[i].raw, datasource[i].raw
    if raw_from.dtype.names is None:
        return _linear_interpolate_helper(t, raw_from, raw_to)
    
    result = np.zeros_like(raw_from)
    for name in raw_from.dtype.names:
        result[name] = _linear_interpolate_helper(t, raw_from[name], raw_to[name])
    return result


def euler_imu_linear_ndarray_interpolator(datasource, float_index):
    i, t = from_float_index(float_index)
    if i+1 != len(datasource):
        raw_from, raw_to = datasource[i].raw, datasource[i+1].raw
    else:
        raw_from, raw_to = datasource[i].raw, datasource[i].raw

    if raw_from.dtype.names is None:
        return _linear_interpolate_helper(t, raw_from, raw_to)
    
    result = np.zeros_like(raw_from)
    for name in raw_from.dtype.names:
        if name in ['roll', 'pitch', 'yaw']:
            result[name] = _angles_linear_interpolate_helper(t, raw_from[name], raw_to[name])
        else:
            result[name] = _linear_interpolate_helper(t, raw_from[name], raw_to[name])
    return result


def linear_dict_of_float_interpolator(datasource, float_index):
    i, t = from_float_index(float_index)
    raw_from, raw_to = datasource[i].raw, datasource[i+1].raw
    interp_dict = raw_from

    for k,value_from in raw_from.items():
        value_to = raw_to[k]
        if isinstance(value_from, list):
            interp_dict[k] = _linear_interpolate_helper(t, np.array(value_from), np.array(value_to)).tolist()
        elif isinstance(value_from, numbers.Real):
            interp_dict[k] = type(value_from)(_linear_interpolate_helper(t, value_from, value_to))
    return interp_dict

def floor_interpolator(datasource, float_index):
    i, _ = from_float_index(float_index)
    return datasource[i].raw

def ceil_interpolator(datasource, float_index):
    i, _ = from_float_index(float_index)
    return datasource[i+1].raw

def nearest_interpolator(datasource, float_index):
    return datasource[round(float_index)].raw
