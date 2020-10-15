from open3d.open3d_pybind.geometry import (PointCloud, KDTreeSearchParamHybrid)
from sklearn.linear_model import LinearRegression, RANSACRegressor

import numpy as np

# , Vector3dVector,
#     estimate_normals, KDTreeSearchParamHybrid, radius_outlier_removal)

def find_inliers(pts, distance_threshold, nb_points):
    if isinstance(pts, PointCloud):
        pcd = pts
    else:
        pcd = PointCloud()
        pcd.points = Vector3dVector(pts)

    _, ind = radius_outlier_removal(pcd, nb_points=nb_points,
        radius=distance_threshold)
    mask = np.zeros(pts.shape[0:1], dtype=np.bool)
    mask[ind] = True
    return mask

def normal_from_parameters(params):
    normal = params[:3].copy()
    normal /= np.linalg.norm(normal)
    return normal

def distance_to_plane(pts, params):
    """Compute the signed distance between a points and a plane. The parameters
    of the plane are given according to

    Arguments:
        pts {np.ndarray} -- The points
        params {np.ndarray} -- The plane parameters

    Returns:
        [np.ndarray] -- Distance
    """
    if pts.ndim == 1 and pts.size == 3:
        pts = pts.reshape(1, 3)
    assert pts.ndim == 2 and pts.shape[1] == 3
    assert params.size == 4
    dst = np.dot(pts, params[:3].reshape(3, 1)) + params[3]
    dst /= np.linalg.norm(params[:3])
    return dst.ravel()

def fit_plane_svd(pts, normalize=True, full_matrices = True):
    """Fit a 3D plane using a SVD

    Arguments:
        pts {np.ndarray} -- The points. Should be Nx3 or Nx4

    Keyword Arguments:
        normalize {bool} -- Subtract the mean to the point cloud (default: {True})

    Returns:
        np.ndarray -- The plane parameters
    """

    min_samples = 3
    assert pts.ndim == 2 and pts.shape[1] in [3, 4]
    assert pts.shape[0] >= min_samples
    n = pts.shape[0]

    pts_mean = pts[:, :3].mean(axis=0)

    if pts.shape[1] == 3:
        pts_ = np.concatenate([pts, np.ones((n, 1))], axis=1)
    else:
        pts_ = pts.copy()

    if normalize:
        # center the point cloud
        pts_[:, :3] -= pts_mean

    # should we pad with a last row of zeros like peter kovesi?
    # see: https://www.peterkovesi.com/matlabfns/Robust/fitplane.m

    # full_matrice = False if try to fit too big vector, and get memory fault
    u, s, v = np.linalg.svd(pts_, full_matrices = full_matrices)
    params = v[3, :]

    if normalize:
        # adjust intercept
        params[3] += np.dot(params[:3], -pts_mean)

    return params

def fit_plane_ols(pts, normalize=True):
    min_samples = 3
    assert pts.ndim == 2 and pts.shape[1] in [3,]
    assert pts.shape[0] >= min_samples

    model = LinearRegression()

    pts_mean = pts.mean(axis=0)
    pts_ = pts.copy()

    if normalize:
        # center the point cloud
        pts_[:, :3] -= pts_mean

    xy = pts_[:, :2]
    z = pts_[:, 2]
    model.fit(xy, z)

    A, B = model.coef_
    D = model.intercept_

    # A * x + B * y + D = z
    # a * x + b * y + c * z + d = 0
    # a = - A
    # b = - B
    # c = 1
    # d = - D
    params = np.array([-A, -B, 1, -D])
    params /= np.linalg.norm(params[:3])

    if normalize:
    # adjust intercept
        params[3] += np.dot(params[:3], -pts_mean)

    return params

def ransac_fit_plane(pts, residual_threshold=0.5, ransac_iter=1000, min_samples=3):
    """Fit a plane using ransac using fit_plane_svd and computing the orthogonal
    distances to find the inliers.

    Arguments:
        pts {np.ndarray} -- The point cloud

    Keyword Arguments:
        residual_threshold {float} -- The threshold distance to the plane (default: {0.5})
        ransac_iter {int} -- The number of ransac iterations (default: {1000})

    Raises:
        ValueError -- The ransac procedure failed to find a plane that satisfies
                      the distance threshold

    Returns:
        (np.ndarray, np.ndarray, np.ndarray) -- Plane parameters, inlier mask,
                                                signed distance to plane
    """
    assert min_samples >= 3
    assert pts.ndim == 2 and pts.shape[1] == 3
    assert pts.shape[0] >= min_samples
    n = pts.shape[0]

    # remove the mean
    pts_mean = pts.mean(axis=0)
    pts = pts - pts_mean

    pts_ = np.concatenate([pts, np.ones((n, 1))], axis=1)
    best_params = None
    num_inliers = 0
    all_indices = np.arange(n)
    for iteration in range(ransac_iter):
        indices = np.random.choice(all_indices, size=min_samples, replace=False)
        selected_pts = pts_[indices, :]
        params = fit_plane_svd(selected_pts, normalize=False)
        dst = np.abs(distance_to_plane(pts, params))
        inliers = dst < residual_threshold
        inliers_count = inliers.sum()
        if inliers_count > num_inliers:
            best_params = params
            num_inliers = inliers_count

    if best_params is None:
        raise ValueError('Could not find a plane with more than 0 inliers')

    inlier_pts = pts_[inliers, :]
    params = fit_plane_svd(inlier_pts, normalize=False)
    dst = distance_to_plane(pts, params)
    inliers = np.abs(dst) < residual_threshold
    # add the mean to the intercept param
    params[3] += np.dot(params[:3], -pts_mean)
    return params, inliers, dst


def ransac_fit_plane_ols(pts, residual_threshold=0.5, ransac_iter=1000, min_samples=3):
    """Fit a plane using RANSAC by solving an ordinary least square problem.
    The main difference with ransac fit plane is that the distances are computed

    Arguments:
        pts {np.ndarray} -- The point cloud

    Keyword Arguments:
        residual_threshold {float} -- The threshold distance to the plane (default: {0.5})
        ransac_iter {int} -- The number of ransac iterations (default: {1000})
        min_samples {int} -- [description] (default: {3})

   Returns:
        (np.ndarray, np.ndarray, np.ndarray) -- Plane parameters, inlier mask,
                                                signed distance to plane
    """
    assert min_samples >= 3
    linear = LinearRegression()
    model = RANSACRegressor(linear, min_samples=min_samples,
        max_trials=ransac_iter, residual_threshold=residual_threshold)

    pts_mean = pts.mean(axis=0)
    pts_ = pts - pts_mean
    xy = pts_[:, :2]
    z = pts_[:, 2]
    model.fit(xy, z)

    A, B = model.estimator_.coef_
    D = model.estimator_.intercept_

    # A * x + B * y + D = z
    # a * x + b * y + c * z + d = 0
    # a = - A
    # b = - B
    # c = 1
    # d = - D
    params = np.array([-A, -B, 1, -D])
    params /= np.linalg.norm(params[:3])
    params[3] += np.dot(params[:3], -pts_mean)

    dst = distance_to_plane(pts, params)
    inliers = np.abs(dst) < residual_threshold

    return params, inliers, dst


def estimate_plane(pts, residual_threshold=0.5, ransac_iter=1000, min_samples=3):
    """Estimate a plane from a 3D points array.

    Plane parameters are estimated using this equation

    a*x + b*y + c*z + d = 0

    returned parameters are in the form

    [a, b, c, d]

    Arguments:
        pts {np.ndarray} -- The points array

    Keyword Arguments:
        residual_threshold {float} -- The ransac residual threshold (default: {0.5})
        ransac_iter {int} -- Number of ransac iterations (default: {100})

    Returns:
        (np.ndarray, np.ndarray, np.ndarray) -- Plane parameters, inlier mask, signed distance to plane
    """
    return ransac_fit_plane(pts, residual_threshold, ransac_iter, min_samples)

def estimate_ground_plane(pts, residual_threshold=0.5, ransac_iter=1000,
                          normal_threshold=0.85, up=None,
                          mask=None, min_samples=3):
    """Estimate the ground plane from a point cloud.

    The plane is not necessarily the ground. It is just a plane whose normal
    is roughly parallel to some `up` vector.


    Arguments:
        pts {np.ndarray} -- Input point cloud

    Keyword Arguments:
        residual_threshold {float} -- The threshold for the RANSAC (default: {0.5})
        ransac_iter {int} -- Number of ransac iterations (default: {100})
        normal_threshold {float} -- The normal threshold to determine potential
                                    ground plane points (default: {0.85})
        up {np.ndarray} -- The up vector indicating an approximate normal vector
                           for the ground plane.
        mask {np.ndarray} -- Point validity mask (only the points
                              where mask == True are used (default: None))

    Returns:
        [np.ndarray] -- Array of the indices of the points that belong to the
                        ground plane
    """
    if up is None:
        # default up vector is z
        up = np.array([0, 0, 1])

    # apply mask
    indices = np.arange(pts.shape[0])
    if mask is None:
        masked_pts = pts
    else:
        masked_pts = pts[mask, :]
        indices = indices[mask]

    # estimate the normals to filter most of the non ground plane points
    pc = PointCloud()
    pc.points = Vector3dVector(masked_pts)
    estimate_normals(pc, search_param=KDTreeSearchParamHybrid(radius=1, max_nn=30))

    normals = np.array(pc.normals)
    normal_mask = np.abs(np.dot(normals, up)) > normal_threshold

    putative_indices = indices[normal_mask]
    putative_ground_pts = masked_pts[normal_mask, :]

    if putative_ground_pts.shape[0] < min_samples:
        return None, None, None, None

    params, inliers, dst = ransac_fit_plane_ols(putative_ground_pts,
        residual_threshold, ransac_iter, min_samples)

    # make sure the plane normal is in the up direction
    normal = normal_from_parameters(params)
    if normal.dot(up) < 0:
        params = -params

    dst = distance_to_plane(pts, params)
    below = dst < -residual_threshold
    all_inliers = (np.abs(dst) < residual_threshold) & ~below

    return params, all_inliers, dst, below

def ray_plane_intersection(ray, plane, origin=None):
    if origin is None:
        origin = np.zeros(3)

    # a * x + b * y + c * z + d = 0

    # ray is the line from origin to lidar point

    # A point in the plane
    # z = - (a * x + b * y + d) / c
    # with x == y == 0
    # z = - d / c
    _, _, c, d = plane
    assert c != 0
    point_in_plane = np.array([[0.0, 0.0, -d/c]])
    normal = normal_from_parameters(plane)

    # See wikipedia: https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
    t = np.dot((point_in_plane - origin), normal) / np.dot(ray, normal)
    # Assign new point
    pt = origin + ray * t.reshape(-1, 1)
    return pt

if __name__ == '__main__':
    import sys
    from das.api.platform import Platform
    from yav import viewer

    def synthetic_example():
        x = np.linspace(-1, 1, 100).astype('f4')
        y = np.linspace(-1, 1, 100).astype('f4')
        true_params = [0.0, 1.0, 0.0, 0.0]
        x, y = [c.ravel() for c in np.meshgrid(x, y)]
        z = - x - y + 0.1 * np.random.rand(*x.shape)
        pts = np.stack([z, y, x], axis=1)
        params, inliers, dst = estimate_plane(pts, 0.2, 10000)

        v = viewer(num=1, axes='widget')
        pcd = v.create_point_cloud()

        colors = np.ones_like(pts)
        colors[inliers, :] = (1.0, 0.0, 0.0)

        pcd.set_points(pts, colors=(colors*255).astype('u1'))
        v.run()

    def platform_example():
        leddar = 'eagle_tfc_ech'
        plat = Platform('/nas/exportedDataset/20181226_162527_RecFile_dataCalib_eagle_mairie4_exported')
        source = plat['eagle_tfc_ech']
        pts = source[800].point_cloud()
        # pts = source[0].point_cloud()

        params, inliers, dst, below = estimate_ground_plane(pts, 0.4, up=[0, -1, 0])

        v = viewer(num=1, axes='widget')
        pcd = v.create_point_cloud()

        colors = np.ones_like(pts)
        colors[inliers, :] = (1.0, 0.0, 0.0)
        colors[below, :] = (0.0, 0.0, 1.0)

        pcd.set_points(pts, colors=(colors*255).astype('u1'))
        v.run()

    # synthetic_example()
    platform_example()