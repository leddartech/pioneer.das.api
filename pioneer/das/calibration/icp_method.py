from pioneer.common.logging_manager import LoggingManager
from pioneer.das.api import platform, interpolators

try :
    from open3d import *
    from open3d.open3d_pybind.geometry import *
    from open3d.open3d_pybind.utility import *
    from open3d.open3d_pybind.registration import *

except:
    LoggingManager.instance().warning("please install open3d - pip3 install open3d")

import warnings

try :
    from pyquaternion import Quaternion
except:
    LoggingManager.instance().warning("please install pyquaternion - pip3 install pyquaternion")


from tqdm import tqdm

import numpy as np
import transforms3d as tf3d

def quaternion_R(log_quaternion):
    """
    Converts the logarithm quaternion to a matrix of rotation 3x3
    """
    q_exp = Quaternion.exp(log_quaternion)
    unit = q_exp.normalised
    R = unit.rotation_matrix

    return R

def log_Quaternion(quaternion):
    """
    Estimate the logarithm of a Quaternion
    """
    log_quaternion = Quaternion.log(quaternion)

    return log_quaternion

def icp_result(source,
                target,
                target_datasource,
                init_matrix,
                imu_speed,
                sync,
                mechanical,
                max_correspondence_distance=0.5, 
                method='Point', 
                max_iteration=50
                ):
    """
    Function where an average matrix 4x4 is estimated with an ICP method
    The rotation part is calculated by taking the mean of logarithmic quaternion and reconvert it to a 3x3 matrix
    The translation part is just the mean of the translation matrix 
    
    Arguments : 
        source : name of the sensor for the source
        target : name of the sensor for the target
        target_datasource : target_datasource
        imu_speed : datasource containing the imu infos. Necessary because we only estimate static frames where the vehicle is not moving (exqmple pf['sbgekinox_bcc']['navposvel'])
        matrice : 4x4 numpy array : alignment transformation.
        sync : plateform synchronisée
        mechanical : bool to know if we are dealing with a mechanical sensor for the target 
    return :
        matrice_final : 4x4 calibration result
        icp_result : algorithm informations stepp by step (list of dict)
    
    """

    frame = 0
    icp_result = []
    # Quaternion properties  
    rotation_quat = Quaternion()
    translation = [0.0,0.0,0.0]
    for i in tqdm(range(len(sync))):
    
        pts_source = sync[i][source].point_cloud()
        if mechanical:
            pts_target = target_datasource.get_at_timestamp(sync[i][source].timestamp, interpolator=interpolators.nearest_interpolator).point_cloud()
        else :
            pts_target = sync[i][target].point_cloud() 

        if imu_speed is not None: # Case where datasource containing the imu infos exists
            speed_imu_EastNorth = imu_speed.get_at_timestamp(sync[i][source].timestamp, interpolator=interpolators.linear_ndarray_interpolator).raw
            ## Value of the imu speed in km/h Don't forget. Make sure the vehicle is not moving
            if np.abs(speed_imu_EastNorth['velocity_n'])<0.05 and np.abs(speed_imu_EastNorth['velocity_e'])<0.05:
                frame += 1
                ## ICP script
                quater,transla,results = icp_routine(pts_source,pts_target,init_matrix,frame,imu_speed,speed_imu_EastNorth,
                                                        max_correspondence_distance, method, max_iteration)
                rotation_quat += quater
                translation += transla
                icp_result.append(results) 
                
        else :
            frame +=1
            ## ICP script
            quater,transla,results = icp_routine(pts_source, pts_target, init_matrix, frame, None, None,
                                                        max_correspondence_distance, method, max_iteration)
            rotation_quat += quater
            translation += transla 
            icp_result.append(results)


    rotation_f = rotation_quat/frame
    rotation_final = quaternion_R(rotation_f)
    translation = translation/frame
    matrice_final = np.eye(4)
    matrice_final[:3,:3] = rotation_final
    matrice_final[:3,3] = translation

    return matrice_final,icp_result

def icp_routine(pts_source,
                pts_target,
                init_matrix,
                frame,
                imu_speed=None,
                speed_imu_EastNorth=None,
                max_correspondence_distance=0.5, 
                method='Point', 
                max_iteration=50):

    transformation, full_result = icp(pts_source,
                                        pts_target,
                                        init_matrix,
                                        max_correspondence_distance,
                                        method,
                                        max_iteration)
    
    rotation_icp, translation_icp = transformation[:3, :3], transformation[:3, 3]
    d = {}
    d['correspondence_set'] = full_result.correspondence_set
    d['frame']=frame
    if imu_speed is not None:
        d['velocity_n']=speed_imu_EastNorth['velocity_n']
        d['velocity_e']=speed_imu_EastNorth['velocity_e']
    d['fitness'] = full_result.fitness
    d['inlier_rmse'] = full_result.inlier_rmse
    d['transformation'] = full_result.transformation
        
    # Using Quaternion to calculate the average rotation matrix and 
    # We used quaternion to avoid issues with adding euler matrices
    # We convert it with a logarithmic function, take the mean of it, reconvert it with an exponentiel function and convert to a matrix rotation 3x3
    euler = tf3d.euler.mat2euler(rotation_icp) 
    axis_angle = tf3d.euler.euler2axangle(euler[0],euler[1],euler[2])
    quaternion = Quaternion(axis =(axis_angle[0][0], axis_angle[0][1], axis_angle[0][2]),radians= axis_angle[1])
    log_quaternion = log_Quaternion(quaternion)

    return log_quaternion, translation_icp, d

def icp(source,
            target,
            init_matrix, 
            max_correspondence_distance=0.5,
            method="Point",
            max_iteration=50):
    """
    Documentation open this package can be found at : http://www.open3d.org/docs/release/introduction.html
    The input are two point clouds and an initial transformation that roughly aligns 
    the source point cloud to the target point cloud. The output is a refined transformation that tightly aligns the two point clouds
    Function evaluate_registration calculates two main metrics. fitness measures the overlapping area (# of inlier correspondences / # of points in target). Higher the better. 
    inlier_rmse measures the RMSE of all inlier correspondences. Lower the better.
    
    Arguments : 
        source : point cloud
        target : point cloud
        rotation : matrix 3x3
        translation : matrix 1x3
        threshold_* : rayon of searching (in meters)
    """
    source_ = PointCloud()
    source_.points = Vector3dVector(source)
    target_ = PointCloud()
    target_.points = Vector3dVector(target)

    if method == "Point":
        reg_p2p = registration_icp(source_, target_, max_correspondence_distance, init_matrix,
                                        TransformationEstimationPointToPoint(),
                                        ICPConvergenceCriteria(max_iteration = max_iteration)
                                    )

    elif method == "Plane":
        source_.estimate_normals(search_param = KDTreeSearchParamHybrid(radius = 1.0, max_nn = 30))
        target_.estimate_normals(search_param = KDTreeSearchParamHybrid(radius = 1.0, max_nn = 30))
        reg_p2p = registration_icp(source_, target_, max_correspondence_distance, init_matrix,
                                        TransformationEstimationPointToPlane(),
                                        ICPConvergenceCriteria(max_iteration = max_iteration)
                                    )
    
    
    
    return reg_p2p.transformation, reg_p2p 


