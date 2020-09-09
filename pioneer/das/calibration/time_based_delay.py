import copy
import os
import pickle
import sys

import matplotlib
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

from das.api import platform
from das.utils import (distance_to_plane, fit_plane_ols, fit_plane_svd,
                       ray_plane_intersection)
from leddar_utils import clouds, images
from yav import amplitudes_to_color, viewer


def signed_distances(pts):
    distances = np.linalg.norm(pts, axis=1)
    sign = np.sign(pts[:, 2])
    distances *= sign
    return distances

def compute_time_based_delays(pts, theoritical_distances):
    """Time based delays computation.

    Simply compute the difference between the measured distances and assumed
    distances.

    Returns the time based delays for each point (pixel).
    """
    distances = signed_distances(pts)
    x0 = distances - theoritical_distances
    return x0


# as measured by Pierre Merriaux
true_distance = 4.196 + 0.11572
dataset = '/nas/exportedDataset/20190409_162629_rec_dataset_time_based_delay_exported'
previous_tbd = '/nas/exportedDataset/20190409_162629_rec_dataset_time_based_delay_exported/eagle_tbased_delays.np.npy'

# as measured by Guillaume Dumont and Pierre Merriaux
true_distance = 4.107 + 0.11572
dataset = '/nas/exportedDataset/20190412_111606_rec_dataset_tbd_vbias_120_exported'
previous_tbd = '/nas/temp/eagle_tbd.npy'

# as measured by Guillaume Dumont and Pierre Merriaux
true_distance = 4.107 + 0.11572
dataset = '/nas/exportedDataset/20190412_112534_rec_dataset_tbd_vbias_235_exported'
previous_tbd = '/nas/temp/eagle_tbd.npy'

# NOTE: VBIAS was set to 55 for this dataset. We may need to set this to the actual
# value to make sure that the time based delays are representative of true
# operational conditions.

outdir = dataset
# time based delay store in the eagle sensor at the time we recorded the data
tbd = np.load(previous_tbd)
# flatten so we can easily index the array
tbd_flat = tbd.ravel()

pf = platform.Platform(dataset)
pf.sensors['eagle_tfc'].config['reject_flags'] = []
# synchronize temperature and distance measurements
sync = pf.synchronized(['eagle_tfc_ech', 'eagle_tfc_sta'])
n = len(sync)


# we want to keep values where we actually measure something
# let's compute the mean distance, amplitude and their std dev.
indices = None
count = None

distances = None
distances2 = None
amplitudes = None
amplitudes2 = None

# also compute the mean temperature
mean_temp = 0.0
for i in tqdm(range(n)):
    data = sync[i]
    state = data['eagle_tfc_sta']
    mean_temp += state.raw['apd_temp'][0]

    # get the valid echoes and the first echo along each ray
    ech = data['eagle_tfc_ech']
    mask = ech.flags == 1
    mask &= images.echoes_visibility_mask(ech.raw['data'])

    # filter random points in distance
    # NOTE: this won't work if the time based delays are set to 0 in the
    # sensor. However, we suspect that they were the cause we need to filter
    # the points here so the lines could be removed.
    mask &= (ech.distances >= 0.75*true_distance)
    mask &= (ech.distances <= 1.25*true_distance)

    if indices is None:
        # initialize everything
        h, v = ech.specs['h'], ech.specs['v']
        indices = np.arange(h*v, dtype=np.int)
        count = np.zeros((h*v))

        distances = np.zeros((h*v))
        amplitudes = np.zeros((h*v))

        distances2 = np.zeros((h*v))
        amplitudes2 = np.zeros((h*v))

    # accumulate
    current_indices = ech.indices[mask]
    count[current_indices] += 1

    # we remove the previous time based delays
    distances[current_indices] += ech.distances[mask] + tbd_flat[current_indices]
    distances2[current_indices] += (ech.distances[mask] + tbd_flat[current_indices])**2
    amplitudes[current_indices] += ech.amplitudes[mask]
    amplitudes2[current_indices] += ech.amplitudes[mask]**2

mean_temp /= n
print('Mean APD temperature', mean_temp)

# take only the samples that we get half of the time or more
valid = count > float(n)/2

# average the valid distances
distances[valid] = distances[valid] / count[valid]
distances[~valid] = 0

# some previous time based delay had values like -250
# so we filter them here. Again the measured distances with time based delays
# would be around -
valid &= (distances >= -10) & (distances <= 10)
distances[~valid] = 0

# compute the
valid_indices = indices[valid]

amplitudes[valid] = amplitudes[valid] / count[valid]
amplitudes[~valid] = 0

distances2[valid] = np.sqrt((distances2[valid] / count[valid] - distances[valid]**2))
distances2[~valid] = 0

amplitudes2[valid] = np.sqrt((amplitudes2[valid] / count[valid] - amplitudes[valid]**2))
amplitudes2[~valid] = 0

# create images for visualization
im_cnt = images.echoes_to_image((v, h), indices, count)
im_cnt = ech.transform_image(im_cnt)

im_amp = images.echoes_to_image((v, h), indices, amplitudes)
im_amp = ech.transform_image(im_amp)

im_amp2 = images.echoes_to_image((v, h), indices, amplitudes2)
im_amp2 = ech.transform_image(im_amp2)

im_dst = images.echoes_to_image((v, h), indices, distances)
im_dst = ech.transform_image(im_dst)

im_dst2 = images.echoes_to_image((v, h), indices, distances2)
im_dst2 = ech.transform_image(im_dst2)

# create a point cloud for visualization
specs = {k:ech.specs[k] for k in ['v', 'h', 'v_fov', 'h_fov']}
angles = clouds.angles(**specs)
directions = clouds.directions(angles)
pts = clouds.to_point_cloud(valid_indices, distances[valid], directions)

# plane equation
# z = 5
# A * x + B * y + C * z + d = 0
# A = B = 0, C = 1, d = -true_distance
plane_params = [0, 0, 1, -true_distance]
true_pts = ray_plane_intersection(directions, plane_params)
true_distances = np.linalg.norm(true_pts, axis=1)

# compute the difference between measured distances and assumed distances
x0 = compute_time_based_delays(pts, true_distances[valid_indices])

# create an image to fill in the missing values
tbd1 = images.echoes_to_image((v, h), valid_indices, x0, min_value=x0.mean())
tbd_new = tbd1.ravel()

with open(os.path.join(outdir, 'eagle_time_based_delays.pkl'), 'wb') as f:
    pickle.dump(dict(time_based_delay=tbd_new, temp=mean_temp), f)

with open(os.path.join(outdir, 'eagle_time_based_delays_combined.pkl'), 'wb') as f:
    pickle.dump(dict(time_based_delay=tbd_new - tbd_flat, temp=mean_temp), f)

tbd1 = ech.transform_image(tbd1)

# previous time based delays
tbd0 = tbd.reshape((specs['v'], specs['h']))
tbd0 = ech.transform_image(tbd0)

corrected_pts = clouds.to_point_cloud(valid_indices, distances[valid]-x0, directions)

def update(v):
    f = v.get_frame()
    data = sync[f]
    ech = data['eagle_tfc_ech']
    mask = ech.flags == 1

    current_indices = ech.indices[mask]
    distances = ech.distances[mask] + tbd_flat[current_indices] - tbd_new[current_indices]

    specs = {k:ech.specs[k] for k in ['v', 'h', 'v_fov', 'h_fov']}
    angles = clouds.angles(**specs)
    directions = clouds.directions(angles)
    pts = clouds.to_point_cloud(current_indices, distances, directions)
    pts = ech.transform(pts, None)

    colors = amplitudes_to_color(ech.amplitudes[mask], log_normalize=True)
    pc2.set_points(pts, colors)

v = viewer(num=n)
pc2 = v.create_point_cloud()
v.add_frame_callback(update)

pc_wnd = v.create_point_cloud_window(title='mean')
pc = pc_wnd.create_point_cloud()
true_pc = pc_wnd.create_point_cloud()
correct_pc = pc_wnd.create_point_cloud()

wnd_cnt = v.create_image_window(title='count')
wnd_amp = v.create_image_window(title='amp')
wnd_dst = v.create_image_window(title='dst')
wnd_amp2 = v.create_image_window(title='amp std')
wnd_dst2 = v.create_image_window(title='dst std')
wnd_tbd0 = v.create_image_window(title='tbd0')
wnd_tbd1 = v.create_image_window(title='tbd')

wnd_cnt.imshow(im_cnt)
wnd_cnt.draw()

wnd_amp.imshow(im_amp)
wnd_amp.draw()

wnd_dst.imshow(im_dst)
wnd_dst.draw()

wnd_amp2.imshow(im_amp2)
wnd_amp2.draw()

wnd_dst2.imshow(im_dst2)
wnd_dst2.draw()

wnd_tbd0.imshow(tbd0)
wnd_tbd0._image.set_norm(matplotlib.colors.Normalize(-11.9, -11.6))
wnd_tbd0.draw()

wnd_tbd1.imshow(tbd1)
wnd_tbd1.draw()

pts = ech.transform(pts, None)
colors = amplitudes_to_color(amplitudes[valid], log_normalize=True)
pc.set_points(pts, colors)

true_pts = ech.transform(true_pts, None)
colors = np.zeros_like(true_pts, dtype='u1')
colors[:, 2] = 255
true_pc.set_points(true_pts, colors)

corrected_pts = ech.transform(corrected_pts, None)
colors = amplitudes_to_color(amplitudes[valid], log_normalize=True)
correct_pc.set_points(corrected_pts, colors)

v.run()
