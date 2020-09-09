from pioneer.das.api.interpolators import linear_dict_of_float_interpolator
from pioneer.das.api.samples import RPM
from pioneer.das.api.sensors.sensor import Sensor

import numpy as np

class Encoder(Sensor):
    def __init__(self, name, platform):
        factories = {'rpm':(RPM, linear_dict_of_float_interpolator)}
        super(Encoder, self).__init__(name, platform, factories)

    def time_travel(self, ts_past, ts_future, x0=0, y0=0, theta0=0):
        """Computes the change of position and orientation between two timestamps"""
        
        first_idx_float = self['rpm'].to_float_index(ts_past)
        last_idx_float = self['rpm'].to_float_index(ts_future)
        
        velocities = np.empty((0,2))
        delta_ts = np.empty(0, dtype='f8')
        for i in range(int(first_idx_float//1), int((last_idx_float)//1)+1):
            if i > 0 and i < len(self['rpm'])-1:
                sample_0, sample_1 = self['rpm'][i], self['rpm'][i+1]
                delta_ts = np.append(delta_ts, (float(sample_1.timestamp) - float(sample_0.timestamp))*1e-6)
                velocities = np.vstack([velocities, self['rpm'][i].meters_per_second()])
        try:
            delta_ts[0] *= (1-first_idx_float%1)
            delta_ts[-1] *= last_idx_float%1
        except: pass
        
        # see http://www.cs.columbia.edu/~allen/F17/NOTES/icckinematics.pdf (equation 5)
        omega = (velocities[:,1] - velocities[:,0])/self.yml['wheel_span']/np.pi
        R = (velocities[:,0] + velocities[:,1])/(2*omega)
        R[np.where(omega==0)] = 0.0
        x, y, theta = x0, y0, theta0
        for i in range(velocities.shape[0]):
            Cx = x - R[i]*np.sin(theta)
            Cy = y + R[i]*np.cos(theta)
            delta_theta = omega[i]*delta_ts[i]
            delta_x = (x-Cx)*np.cos(delta_theta) - (y-Cy)*np.sin(delta_theta) + Cx - x
            delta_y = (x-Cx)*np.sin(delta_theta) + (y-Cy)*np.cos(delta_theta) + Cy - y
            x += delta_x
            y += delta_y
            theta += delta_theta
        return x, y, theta


    def compute_trajectory(self, timestamps):
        trajectory = np.empty((0,3))
        time, x, y, theta = timestamps[0], 0, 0, 0
        for ts in timestamps[1:]:
            x, y, theta = self.time_travel(time, ts, x, y, theta)
            time = ts
            trajectory = np.vstack([trajectory, np.array([x,y,theta])])
        return trajectory
            
