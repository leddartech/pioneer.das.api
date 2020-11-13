#TODO: move this to pioneer.das.acquisition
from pioneer.das.api import platform

try:
    import folium #pip3 install folium
except:
    pass
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import tqdm
import utm

def easting_northing_from_lat_long(latitude, longitude):
    easting, northing, _, _ = utm.from_latlon(latitude, longitude)
    return easting, northing

def distance_traj_step(easting, northing, t=None):
    d_e = np.diff(easting)
    d_n = np.diff(northing)
    if t is not None:
        d_t = np.diff(t)
    return (d_e**2 + d_n**2)**0.5/d_t

def get_trajectory(pfsynch:platform.Synchronized,
                    ref_ts_sensor:str='flir_bfc_img',
                    imu_nav:str='sbgekinox_bcc_navposvel'):
    
    '''simple: return easting, northing, points list, and time of the trajectory following the timestamps of ref_ts_sensor
    '''
    n = len(pfsynch)
    easting, northing, ts = [], [], []
    points = []
    for mu in tqdm.tqdm(range(n)):
        ref_ts = pfsynch[mu][ref_ts_sensor].timestamp
        imu = pfsynch.platform[imu_nav].get_at_timestamp(ref_ts).raw
        lati, longi = imu['latitude'], imu['longitude']
        eg, ng = easting_northing_from_lat_long(lati, longi)
        easting.append(eg)
        northing.append(ng)
        ts.append(ref_ts/1e6)
        points.append([lati, longi])
    
    return np.array(easting, dtype=np.float64), np.array(northing, dtype=np.float64), points, np.array(ts, dtype=np.float64)-ts[0]


def compute_neighbour_step_ratio(xt, yt, t, min_epsilon_precision=1e-5):
    step_ratio_norm = []
    step_ratio_norm.append(0)
    for i in range(1,len(xt)-1):
        d_t_l = np.abs(t[i-1]-t[i])
        d_t_r = np.abs(t[i+1]-t[i])

        d_xt_l = np.maximum(np.abs(xt[i-1]-xt[i])/d_t_l, min_epsilon_precision)
        d_xt_r = np.maximum(np.abs(xt[i+1]-xt[i])/d_t_r, min_epsilon_precision)
        
        d_yt_l = np.maximum(np.abs(yt[i-1]-yt[i])/d_t_l, min_epsilon_precision)
        d_yt_r = np.maximum(np.abs(yt[i+1]-yt[i])/d_t_r, min_epsilon_precision)

        step_ratio_xt = np.maximum(d_xt_l, d_xt_r) / np.minimum(d_xt_l, d_xt_r)
        step_ratio_yt = np.maximum(d_yt_l, d_yt_r) / np.minimum(d_yt_l, d_yt_r)
        
        step_ratio_norm.append((step_ratio_xt**2 + step_ratio_yt**2)**0.5)
    
    step_ratio_norm.append(0)
    return np.array(step_ratio_norm, dtype=np.float)


def compute_standard_score(x, seq_memory: int=200, start_at_zero: bool=True, outliers_threshold: float=100.0):

    '''return the standard score based on a memory sequence of certain length. 
    '''
    m = len(x)
    epsilon_ = 1e-4 # 0.1 mm precision
    z_score = []
    z_score.append(0)
    flag_outliers = np.zeros_like(x, dtype=bool)
    for mu in tqdm.tqdm(range(1, m)):
        a, b = np.maximum(mu - seq_memory, 0), mu
        
        if mu < seq_memory and not start_at_zero:
            z_score.append(0)
            continue
        
        window_seq = x[a:b][~flag_outliers[a:b]]
        
        # if mu > seq_memory and len(window_seq) < 0.25*seq_memory:
        #     z_score.append(0)
        #     continue
        
        seq_mean = np.mean(window_seq)
        seq_std = np.std(window_seq)

        z_ = np.abs((x[mu] - seq_mean)/(seq_std + epsilon_))

        if z_ > outliers_threshold:
            flag_outliers[mu] = 1

        z_score.append(np.copy(z_))
    
    return np.array(z_score)


def get_trajectory_standard_score(pfsynch:platform.Synchronized,
                                    ref_ts_sensor:str='flir_bfc_img',
                                    imu_nav:str='sbgekinox_bcc_navposvel',
                                    traj_seq_memory:int=200):
    
    '''estimation of the smoothness of a trajectory based on the standard score.
    '''

    easting, northing, _, t = get_trajectory(pfsynch, ref_ts_sensor, imu_nav)
    traj_step = distance_traj_step(easting, northing, t)
    z_scores = np.zeros_like(easting)
    z_scores[1:] = compute_standard_score(traj_step, traj_seq_memory-1, False)
    return z_scores


def get_trajectory_step_ratio(pfsynch:platform.Synchronized,
                                ref_ts_sensor:str='flir_bfc_img',
                                imu_nav:str='sbgekinox_bcc_navposvel',
                                traj_min_epsilon_precision:float=1e-6):

    '''estimation of the smoothness of the trajectory based on the ratio of left-right epsilons step
    '''
    easting, northing, _, t = get_trajectory(pfsynch, ref_ts_sensor, imu_nav)
    return compute_neighbour_step_ratio(easting, northing, t, traj_min_epsilon_precision)


def find_trajectory_jump(pfsynch:platform.Synchronized,
                            ref_ts_sensor:str='flir_bfc_img',
                            imu_nav:str='sbgekinox_bcc_navposvel',
                            traj_seq_memory:int=200,
                            traj_jump_threshold:float=15.5,
                            show_result:bool=True):
    
    '''Compute the list of ranges of intervall from pfsynch which are smooth according to traj_jump_threshold.
    '''
    
    print('Computing trajectory')
    easting, northing, points, t = get_trajectory(pfsynch, ref_ts_sensor, imu_nav)
    traj_step = distance_traj_step(easting, northing)

    print('Validate trajectory')
    z_scores = np.zeros_like(easting)
    z_scores[1:] = compute_standard_score(traj_step, traj_seq_memory-1, False)
    jump_flag = (z_scores > traj_jump_threshold).astype(bool)
    
    list_intervals = []
    ids = np.arange(len(jump_flag))[jump_flag]
    for mu in range(len(ids)):
        if mu == 0:
            list_intervals.append([0 , ids[mu]-1])
            continue
        if ids[mu]-ids[mu-1] >= traj_seq_memory:
            list_intervals.append([ids[mu-1], ids[mu]-1])
    
    if show_result:
        t = np.arange(len(easting))
        fig, ax = plt.subplots(2, 1, figsize=(9,10))
        fig.suptitle('Trajectory positions and jumps')
        ax[0].scatter(t, easting)
        ax[0].scatter(t[jump_flag], easting[jump_flag], label='jump flags')
        ax[0].legend()
        ax[0].set_xlabel('Frame number')
        ax[0].set_ylabel('Easting')

        ax[1].scatter(t, northing)
        ax[1].scatter(t[jump_flag], northing[jump_flag], label='jump flags')
        ax[1].legend()
        ax[1].set_xlabel('Frame number')
        ax[1].set_ylabel('Northing')
        plt.show()

        my_map = folium.Map(location=points[0],  zoom_start=15)
        folium.PolyLine(points).add_to(my_map)
        for mu in ids:
            folium.CircleMarker(
                    location=points[mu],
                    radius=5.5,
                    popup='IMU jump: '+ str(mu),
                    color='red',
                    fill=True,
                    fill_color='red'
                ).add_to(my_map)
        return jump_flag, list_intervals, my_map
    
    return jump_flag, list_intervals





if __name__ == '__main__':
    #example of use:

    #see this dataset:
    _dataset = '/nas/pixset/exportedDataset/20200610_195655_rec_dataset_quartier_pierre_exported'
    _ignore = ['radarTI_bfc']
    pf = platform.Platform(_dataset, ignore=_ignore)

    # get the platform synchronized:
    sync_labels = ['*ech*', '*_img*', '*_trr*', '*_trf*',' *_ftrr*', '*xyzit-*']
    interp_labels = ['*_xyzit', 'sbgekinox_*', 'peakcan_*', '*temp', '*_pos*', '*_agc*']

    synch = pf.synchronized(sync_labels=sync_labels, interp_labels=interp_labels, tolerance_us=1e3)

    flags, inters, my_map = find_trajectory_jump(synch,
                                            ref_ts_sensor='flir_bfc_img',
                                            imu_nav='sbgekinox_bcc_navposvel',
                                            traj_seq_memory=200,
                                            traj_jump_threshold=4.0,
                                            show_result=True)

    print('Intervals:', inters)

    








    

   
    


