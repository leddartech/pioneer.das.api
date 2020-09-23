# -*- coding: utf-8 -*-
# script de calibration generique avec gestion des boucles
# il y a des exemples de fichier de config yaml des decription de transformation/boucle dans le repertoire loopYAMLfileExample
# s'utilise après le script homologs.py
from extrinsic_left_right_lca3_loop_calibration import RansacCalibration, axangle4to3, axangle3to4, axangleX0init, x2mat, state2Affine, inverseTransform, error_projection, getEulerAngleZYX

from mpl_toolkits import mplot3d
from numpy import dtype, argsort
from scipy.optimize import minimize
from scipy.sparse.csgraph import shortest_path
from transforms3d import euler, axangles

import argparse
import cv2
import math

import matplotlib.pyplot as plt

import numpy as np
import os
import pickle
import time
import transforms3d
import warnings
import yaml

class RigidTransformation:
    transformations_number = 0

    def __init__(self, data_filename, name, verbose=True):
        self.verbose = verbose
        self.name = name
        pts = np.loadtxt(data_filename)
        self.ptsSource = np.concatenate([pts[:, 1:4], np.ones(
            (pts[:, 1:4].shape[0], 1), dtype=pts[:, 1:4].dtype)], axis=1).T
        self.ptsTarget = np.concatenate([pts[:, 4:7], np.ones(
            (pts[:, 4:7].shape[0], 1), dtype=pts[:, 4:7].dtype)], axis=1).T

        # parcouru par un dic, ce sera dans l'ordre alphabetique
        self.transformations_id = RigidTransformation.transformations_number
        RigidTransformation.transformations_number += 1

    def getAffine(self, x):
        # extraction de la partie concernant cette transfomation du vecteur d'etat global
        Tr = state2Affine(
            x[self.transformations_id*6:self.transformations_id*6+6])
        return Tr

    def error_projection(self, x):
        # get affine
        Tr = self.getAffine(x)
        return error_projection(Tr, self.ptsSource_ransac, self.ptsTarget_ransac)

    def compute_init_ransac(self, ransac_solver):
        self.ransac_Tr_init, self.ransac_inlier_idx, self.ransac_error = ransac_solver.calibrate(
            self.ptsSource, self.ptsTarget)

        if(self.verbose):
            print('Ransac for transformation {}, {}/{} points selected, error {}m\n{}'.format(self.name,
                                                                                              self.ransac_inlier_idx.shape[0], self.ptsSource.shape[1], np.sqrt(self.ransac_error), self.ransac_Tr_init))

        # conserver les points selectionnés par le ransac
        self.ptsSource_ransac = self.ptsSource[:, self.ransac_inlier_idx]
        self.ptsTarget_ransac = self.ptsTarget[:, self.ransac_inlier_idx]

        # transform Tr affine to x vector
        self.x0 = axangleX0init(self.ransac_Tr_init)
        return self.x0

    def set_X0_X_vector(self, x, ransac_solver):
        x_Tr = self.compute_init_ransac(ransac_solver)
        x[self.transformations_id*6:self.transformations_id*6+6] = x_Tr

    def print_result(self,x):
        tr_affine = self.getAffine(x)
        print('Transformation {} result:'.format(self.name))
        print(tr_affine)
        print('euler:{}'.format(np.rad2deg(getEulerAngleZYX(tr_affine))))
        print('axis angles:{}'.format(
            axangles.mat2axangle(tr_affine[0:3,0:3])))
        # erreur moyenne en metre
        error = np.sqrt(error_projection(tr_affine, self.ptsSource_ransac, self.ptsTarget_ransac, True))
        print('error de projection mean:{}m'.format(error))

class LoopTransformation:
    def __init__(self, loop_formula, tranformations_dic, verbose=True):
        self.verbose = verbose
        self.formula = loop_formula
        self.transformations = tranformations_dic
        self.Tr_list = []
        for member in loop_formula.split('->'):
            # on verifie que toutes les membres existent
            test1 = member in self.transformations
            test2 = member[0] == '-' and member[1:] in self.transformations
            if not (test1 or test2):
                warnings.warn('Transformation {} unknown in formula {}'.format(
                    member, self.formula))
                return None  # transformation non valide
            if(member[0] == '-'):
                self.Tr_list.append({'inv': True, 'TrName': member[1:]})
            else:
                self.Tr_list.append({'inv': False, 'TrName': member})

    # composition de la transformation affine formant la boucle

    def getAffine(self, x):
        Tr = np.eye(4)
        for t in self.Tr_list:
            if t['inv']:
                Tr = inverseTransform(
                    self.transformations[t['TrName']].getAffine(x)).dot(Tr)
            else:
                Tr = (self.transformations[t['TrName']].getAffine(x)).dot(Tr)
        #print('T_loop\n{}'.format(Tr))
        return Tr

    # error de projection evaluée avec les points ransac source de la premiere transformation presente dans la formule de la boucle
    def error_projection(self, x):
        pts = self.transformations[self.Tr_list[0]['TrName']].ptsSource_ransac
        Tr = self.getAffine(x)
        return error_projection(Tr, pts, pts)

    def print_result(self,x):
        print('Loop {} result:'.format(self.formula))
        Tr=self.getAffine(x)
        print(Tr)
        print('euler:{}'.format(np.rad2deg(getEulerAngleZYX(Tr))))
        print('axis angles:{}'.format(
            axangles.mat2axangle(Tr[0:3, 0:3])))
        print('Norm loop translation:{}'.format(
            np.linalg.norm(Tr[0:3, 3])))

        pts = self.transformations[self.Tr_list[0]['TrName']].ptsSource_ransac
        error = np.sqrt(error_projection(Tr, pts, pts, True))
        print('error de projection mean:{}m'.format(error))

    def print_ransac_init(self, x0):
        print('Loop {} ransac init:'.format(self.formula))
        Tr=self.getAffine(x0)
        print('axis angles:{}'.format(
            axangles.mat2axangle(Tr[0:3, 0:3])))
        print('Norm loop translation:{}'.format(
            np.linalg.norm(Tr[0:3, 3])))

        pts = self.transformations[self.Tr_list[0]['TrName']].ptsSource_ransac
        error = np.sqrt(error_projection(Tr, pts, pts, True))
        print('error de projection mean:{}m'.format(error))

class LoopCalibration:
    def __init__(self, config, base_folder, verbose=True):
        self.verbose = verbose
        self.config = config
        self.base_folder = base_folder
        self.nb_transformations = len(config['transformations'])

        self.solution = None
        # ajout des transformations
        self.transformations = {}
        for tr_name in config['transformations'].keys():
            self.transformations[tr_name] = RigidTransformation(
                os.path.join(base_folder, config['transformations'][tr_name]), tr_name, self.verbose)
        # ajout des loops
        self.loops = {}
        if config['loops'] is None:
            config['loops'] = {}
        else :
            for loop_name in config['loops'].keys():
                self.loops[loop_name] = LoopTransformation(
                    config['loops'][loop_name], self.transformations, self.verbose)
        # 6 x rigid transfomations [Tx,Ty,Tz,Ax,Ay,Az] each time
        self.x0 = np.zeros(
            len(self.transformations) * 6, dtype=np.float64)

        # set default ransac config
        if 'ransac' in config and 'nb_iterations' in config['ransac']:
            self.ransac_nb_iterations = config['ransac']['nb_iterations']
        else:
            self.ransac_nb_iterations = 5000
        if 'ransac' in config and 'distance_threshold' in config['ransac']:
            self.ransac_distance_threshold = config['ransac']['distance_threshold']
        else:
            self.ransac_distance_threshold = 0.15

        # ransac solver
        self.ransac_solver_init = RansacCalibration(
            self.ransac_nb_iterations, self.ransac_distance_threshold)

    # calcul les x initiaux
    def compute_x0(self):
        for tr_name in self.transformations:
            self.transformations[tr_name].set_X0_X_vector(
                self.x0, self.ransac_solver_init)
        print('x0={}'.format(self.x0))


    # fonction objective globale
    def objective(self, x):
        #print(x)
        cost = 0
        for tr_name in self.transformations:
            cost += self.transformations[tr_name].error_projection(x)

        for loop_name in self.loops:
            cost += self.loops[loop_name].error_projection(x)

        return(cost)

    def optimize(self):
        self.solution = minimize(self.objective, self.x0,
                            method='SLSQP', options={'disp': True, 'maxiter': 200})
        if(self.verbose):
            print(self.solution)

        return(self.solution.x)


    def print_result(self):
        for tr_name in self.transformations:
            self.transformations[tr_name].print_result(self.solution.x)

        for loop_name in self.loops:
            self.loops[loop_name].print_result(self.solution.x)

    def print_ransac_loop_init(self):
        for loop_name in self.loops:
            self.loops[loop_name].print_ransac_init(self.x0)



    def dump_solution(self, multi_file = False):
        data = {}
        for tr_name in self.transformations:
            Tr = self.transformations[tr_name].getAffine(self.solution.x)
            TrInv = inverseTransform(Tr)
            data[tr_name]= Tr
            data[tr_name+'Inv']= TrInv
            Tr_ransac = self.transformations[tr_name].getAffine(self.x0)
            TrInv_ransac = inverseTransform(Tr_ransac)
            data[tr_name+'_ransac']= Tr_ransac
            data[tr_name+'Inv'+'_ransac']= TrInv_ransac

        with open(os.path.join(self.base_folder, self.config['output']),'wb') as f:
            pickle.dump(data, f)

        # export as single file per transformation
        if multi_file:
            for tr_name in self.transformations:
                Tr = self.transformations[tr_name].getAffine(self.solution.x)
                with open(os.path.join(self.base_folder, tr_name + '.pkl'),'wb') as f:
                    pickle.dump(Tr, f)

    def get_transform(self, x, specification):
        if '->' in specification:
            tform = LoopTransformation(specification, self.transformations, True)
            return tform.getAffine(x)
        elif specification.startswith('-'):
            tform = self.transformations[specification[1:]]
            return inverseTransform(tform.getAffine(x))
        else:
            tform = self.transformations[specification]
            return tform.getAffine(x)

        raise ValueError('Should not be here.')

    def build_transformation_graph(self):
        """Build a distance matrix between all sensors. Here distance has
        nothing to do with the physical distance between sensors but represents
        the existence of a transformation between the two sensors.

        Returns:
            [tuple] -- Tuple containing a list with lexicographically ordered
                       sensor names and a np.ndarray containing the distances
                       be
        """

        sensors = [s for t in self.transformations for s in t.split('-')]
        sensors = list(set(sensors))
        sensors.sort()

        n = len(sensors)
        distance_matrix = np.zeros((n, n))
        for t in self.transformations.keys():
            source, target = t.split('-')
            source_index = sensors.index(source)
            target_index = sensors.index(target)
            assert source_index >= 0, 'Could not find sensor {} index'.format(source)
            assert target_index >= 0, 'Could not find sensor {} index'.format(target)

            # set the distance to one when the two sensors have a transformation
            # between them.
            distance_matrix[source_index, target_index] = 1
            distance_matrix[target_index, source_index] = 1

        return sensors, distance_matrix


    def create_sensor_transformation_pairs(self, reference):
        """Create a list of tuples where each tuple contains a sensor name
        and the shortest transformation chain to reach this sensor from the
        reference sensor

        Arguments:
            reference {str} -- The reference sensor name

        Returns:
            [list] -- List of sensor and transformation combination
        """
        sensors, distance_matrix = self.build_transformation_graph()
        d, p = shortest_path(distance_matrix, directed=False,
                             return_predecessors=True)

        def get_path(pred, i, j):
            path = [j]
            k = j
            while pred[i, k] != -9999:
                path.append(pred[i, k])
                k = pred[i, k]
            return path[::-1]

        transforms = []
        for sensor in sensors:
            if sensor == reference:
                transforms.append((sensor, None))
                continue

            source_index = sensors.index(reference)
            target_index = sensors.index(sensor)

            assert source_index >= 0, \
                'Could not find sensor {} index'.format(reference)
            assert target_index >= 0, \
                'Could not find sensor {} index'.format(target_index)

            path = get_path(p, source_index, target_index)
            sensor_transform = []
            for i in range(len(path)-1):
                s1 = sensors[path[i]]
                s2 = sensors[path[i+1]]

                t = s1 + '-' + s2

                # make sure we find a transformation that exist
                # in our list.
                if t not in self.transformations:
                    # try the inverse of the inverse transform if the direct
                    # transform does not exist
                    tinv = s2 + '-' + s1
                    assert tinv in self.transformations
                    t = '-' + tinv

                sensor_transform.append(t)

            # generate a transform that uses the same naming as
            # the loops.
            transform = '->'.join(sensor_transform)

            transforms.append((sensor, transform))

        return transforms


    def plot(self, v, x, title, cfg=None):
        if cfg is None:
            return

        assert isinstance(cfg, dict)
        reference = cfg['reference']
        transforms = self.create_sensor_transformation_pairs(reference)

        wnd = v.create_point_cloud_window(title=title)
        wnd.create_caption2d(reference, [0, 0, 0])
        for sensor, tform_spec in transforms:
            if sensor == reference:
                continue

            # TODO: combine transform using the spec and avoid using
            # the defined loops and transformations

            matrix = self.get_transform(x, tform_spec)
            # we need the inverse transform
            matrix = inverseTransform(matrix)

            wnd.create_axes(matrix=matrix)
            pos = matrix[:3, 3].tolist()
            wnd.create_caption2d(sensor, pos)

    def plot_points(self, v, x, cfg=None):
        from functools import partial
        from das.coco_label_categories_colors import COLORS

        if cfg is None:
            return

        assert isinstance(cfg, dict)
        reference = cfg['reference']

        wnd = v.create_point_cloud_window(title='points')
        pcd = {}
        transformations = sorted(self.transformations.keys())
        transformations = [t for t in transformations if reference in t]

        def on_checkbox(name, state):
            pcd[name][0].show(state != 0)
            pcd[name][1].show(state != 0)
            wnd.render()

        for i, t in enumerate(transformations):
            source, target = t.split('-')
            m = self.get_transform(x, t)
            if reference == source:
                m = inverseTransform(m)
                pts1 = self.transformations[t].ptsSource
                pts2 = np.dot(m, self.transformations[t].ptsTarget)

                pts1_ransac = self.transformations[t].ptsSource_ransac
                pts2_ransac = np.dot(m, self.transformations[t].ptsTarget_ransac)
            else:
                pts1 = self.transformations[t].ptsTarget
                pts2 = np.dot(m, self.transformations[t].ptsSource)

                pts1_ransac = self.transformations[t].ptsTarget_ransac
                pts2_ransac = np.dot(m, self.transformations[t].ptsSource_ransac)


            pts1 = pts1[:3, :].T
            pts2 = pts2[:3, :].T

            pts1_ransac = pts1_ransac[:3, :].T
            pts2_ransac = pts2_ransac[:3, :].T

            pc1 = wnd.create_point_cloud()
            pc2 = wnd.create_point_cloud()

            pc1_ransac = wnd.create_point_cloud()
            pc2_ransac = wnd.create_point_cloud()

            colors1 = np.zeros_like(pts1, dtype='u1')
            colors1[...] = (255, 255, 255)

            colors2 = np.zeros_like(pts2, dtype='u1')
            colors2[...] = COLORS[i+1]

            pc1.set_points(pts1, colors=colors1)
            pc2.set_points(pts2, colors=colors2)

            pc1_ransac.set_points(pts1_ransac, colors=colors1)
            pc2_ransac.set_points(pts2_ransac, colors=colors2)

            v.add_checkbox(t, partial(on_checkbox, t), True)
            v.add_checkbox(t + '_ransac', partial(on_checkbox, t + '_ransac'), True)

            pcd[t] = [pc1, pc2]
            pcd[t + '_ransac'] = [pc1_ransac, pc2_ransac]




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str,
                        default='loop.yml', help='Configuration file')
    parser.add_argument('-o', '--output', type=str,
                        default=None, help='Output folder')
    parser.add_argument('-q', '--quiet', action='store_true',
                        default=False, help='Quiet mode')
    parser.add_argument('-s', '--singlefile', action='store_true',
                        default=False, help='Single file output')



    args = parser.parse_args()
    verbose = not args.quiet
    config_folder = os.path.split(os.path.abspath(args.config))[0]
    with open(args.config, 'r') as f:
        config = yaml.load(f)

    loop_calibration = LoopCalibration(config, config_folder, verbose)
    loop_calibration.compute_x0()
    loop_calibration.print_ransac_loop_init()
    x = loop_calibration.optimize()
    if(verbose):
        loop_calibration.print_result()

    loop_calibration.dump_solution(not args.singlefile)
    plot_cfg = config.get('plot', None)
    if plot_cfg is not None:
        from yav import viewer, amplitudes_to_color
        v = viewer(num=1, title='calibration_loops')
        loop_calibration.plot(v, loop_calibration.x0, 'ransac', plot_cfg)
        loop_calibration.plot(v, x, 'solution', plot_cfg)
        loop_calibration.plot_points(v, x, plot_cfg)
        v.run()


