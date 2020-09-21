import copy
import numpy as np

def load_chessboard_specifications(platform):
    ignore = platform.yml['ignore']
    if 'camera_intrinsics' in ignore:
        camera_intrinsics = ignore['camera_intrinsics']
        assert 'chessboard' in camera_intrinsics, \
            'Could not find chessboard section in ignore.camera_instrinsics'
    elif 'chessboard' in ignore:
        camera_intrinsics = ignore
    else:
        raise ValueError('Cound not find chessboard section in ignore section.'
            ' Use ignore.camera_intrinsics.chessboard or ignore.chessboard')

    chessboard = copy.deepcopy(camera_intrinsics['chessboard'])
    ny = chessboard['ny']
    nx = chessboard['nx']
    dx = chessboard['dx']
    dy = chessboard['dy']

    points = np.zeros((nx * ny, 3), dtype=np.float32)
    # generate a row-major grid of points, i.e. x changes faster than y
    points[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    points[:, 0] *= dx
    points[:, 1] *= dy

    chessboard['points'] = points

    # load reflective spheres from positions
    spheres = chessboard.get('reflective_spheres', [])
    reflective_spheres = []
    for sphere in spheres:
        x, y = sphere['corner_xy']
        assert x >= 0 and x < nx and y >= 0 and y < ny, \
            'Invalid coordinates for reflective spheres'
        dxy = sphere['offsets']
        pidx = y*nx + x
        point = points[pidx, :] + [*dxy, 0.0]

        reflective_spheres.append(point)


    reflective_spheres = np.array(reflective_spheres)
    chessboard['reflective_spheres'] = reflective_spheres

    return chessboard
