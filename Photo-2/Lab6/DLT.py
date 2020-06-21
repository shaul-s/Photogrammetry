import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import rq
import pandas as pd
from Camera import *
from SingleImage import *


class dlt():
    """
    DLT module
    :param ground_points: Ground points
    :param picture_points: Picture points
    """

    def __init__(self, ground_points, picture_points):
        self.ground_points = ground_points
        self.picture_points = picture_points

    @staticmethod
    def generatePicturePoints(f, xp, yp, omega, phi, kappa, X0, Y0, Z0, ground_points):
        """
        :param f: Focal length (mm)
        :param xp: Camera center fix on axis x (micron)
        :param yp: Camera center fix on axis y (micron)
        :param omega: External rotation (rad)
        :param phi: External rotation (rad)
        :param kappa: External rotation (rad)
        :param X0: External location of the camera (m)
        :param Y0: External location of the camera (m)
        :param Z0: External location of the camera (m)
        :param ground_points: Ground points
        :return: Picture points
        :type f: float
        :type xp: float
        :type yp: float
        :type omega: float
        :type phi: float
        :type kappa: float
        :type X0: float
        :type Y0: float
        :type Z0: float
        :type ground_points: ndarray
        :rtype: ndarray
        """

        # Set "show_output = True" if you want to see output
        show_output = False

        r = np.array(R.from_euler('zyx', [omega, phi, kappa], degrees=True).as_matrix())

        c = np.array([[X0], [Y0], [Z0]])

        k = np.array([[-f, 0, xp],
                      [0, -f, yp],
                      [0, 0, 1]])

        p = np.dot(np.dot(k, r), np.hstack((np.identity(3), -c)))

        ground_points_temp = np.vstack((ground_points.T, np.ones((1, ground_points.shape[0]))))

        picture_box_points = np.dot(p, ground_points_temp)

        picture_box_points_actual = picture_box_points.T
        picture_box_points_actual[:, 0] = picture_box_points_actual[:, 0] / picture_box_points_actual[:, 2]
        picture_box_points_actual[:, 1] = picture_box_points_actual[:, 1] / picture_box_points_actual[:, 2]
        picture_box_points_actual[:, 2] = picture_box_points_actual[:, 2] / picture_box_points_actual[:, 2]

        if show_output:  # Set "show_output = True" if you want to see output
            print('Rotation matrix\n', pd.DataFrame(r))
            print('\nCamera location\n', pd.DataFrame(c))
            print('\nK Matrix\n', pd.DataFrame(k))
            print('\nP Matrix\n', pd.DataFrame(p))
            print('\nGround points of the box\n', pd.DataFrame(ground_points_temp))
            print('\nPicture points of the box (wx,wy,w)\n', pd.DataFrame(picture_box_points.T))
            print('\nPicture points of the box (x,y)\n', pd.DataFrame(picture_box_points_actual[:, :2]))

        return picture_box_points_actual[:, :2]

    @staticmethod
    def solveExteriorOrientation(picture_points, ground_points):
        """
        :param picture_points: (x,y) Assuming w=1, Picture (mm)
        :param ground_points: (X,Y,Z) Ground (m)
        :return:
        """

        # Set "show_output = True" if you want to see output
        show_output = False

        # Adding homogenius vector of 1 to ground points
        ground_points = np.hstack((ground_points, np.ones((ground_points.shape[0], 1))))

        # Creating A
        A_rows = picture_points.shape[0] * 2
        A = np.zeros((A_rows, 12))

        A[0::2, 4:8] = -ground_points
        A[0::2, 8:12] = ground_points * np.reshape(picture_points[:, 1], (picture_points.shape[0], 1))
        A[1::2, 0:4] = ground_points
        A[1::2, 8:12] = -ground_points * np.reshape(picture_points[:, 0], (picture_points.shape[0], 1))

        # for row in range(0, A_rows, 2):  # Two rows each time
        #
        #     # First restriction
        #     A[row, 4:8] = -ground_points[int(row / 2), :]
        #     A[row, 8:12] = ground_points[int(row / 2), :] * picture_points[int(row / 2), 1]
        #
        #     # Second restriction
        #     A[row + 1, 0:4] = ground_points[int(row / 2), :]
        #     A[row + 1, 8:12] = -ground_points[int(row / 2), :] * picture_points[int(row / 2), 0]

        N = np.dot(A.T, A)

        egi_vals, egi_vect = np.linalg.eig(N)

        min_egi_val_index = np.argmin(egi_vals)

        v = egi_vect[:, min_egi_val_index]

        p = v.reshape((3, 4))

        c = -np.dot(np.linalg.inv(p[:, 0:3]), p[:, 3].reshape((3, 1)))

        r, q = rq(p[:, 0:3])
        if r[0, 0] > 0:
            r_fixed = np.round(np.dot(r / np.abs(r[-1, -1]), np.diag([-1, 1, 1])), 6)
        else:
            r_fixed = np.round(np.dot(r / np.abs(r[-1, -1]), np.diag([1, -1, -1])), 6)
        q = np.dot(np.diag([1, -1, -1]), q)
        angles = (R.from_matrix(q)).as_euler('zyx', degrees=True)

        if show_output:  # Set "show_output = True" if you want to see output
            print('A matrix\n', pd.DataFrame(A))
            print('\nNormal matrix\n', pd.DataFrame(N))
            print('\nEigenvalues\n', pd.DataFrame(egi_vals))
            print('\nNormalized eigenvectors\n', pd.DataFrame(egi_vect))
            print('\nV correspond to minimal eigen value\n', pd.DataFrame(v))
            print('\nP matrix\n', pd.DataFrame(p))
            print('\nCamera exterior location\n', pd.DataFrame(c))
            print('\nCamera exterior rotation\n', pd.DataFrame(q))
            print('\nCamera exterior rotation in angles Omega Phi Kappa\n', pd.DataFrame(angles))
            print('\nK matrix\n', pd.DataFrame(r_fixed))

        return [-r_fixed[0, 0], r_fixed[0, 2], r_fixed[1, 2], *angles.tolist(), *c[:, 0].tolist()]

    @staticmethod
    def addNoise(array):
        """
        return the same array + noise between 0 and 1
        """
        return array + np.random.random(array.shape)


if __name__ == '__main__':
    ### --- Set True to the test you wish to perform --- ###
    ground_box_points = np.array([[-5, -5, -5],
                                  [-5, 5, -5],
                                  [5, 5, -5],
                                  [5, -5, -5],
                                  [-5, -5, 5],
                                  [-5, 5, 5],
                                  [5, 5, 5],
                                  [5, -5, 5]])

    run_class_example = False
    if run_class_example:  # Recreating class example

        picture_points = dlt.generatePicturePoints(153, 0.2, 0.2, 60, 45, -30, 15, 10, 50, ground_box_points)
        print('\nPicture points of the box (x,y)\n', pd.DataFrame(picture_points))

    solve_exterior_dlt = False
    if solve_exterior_dlt:  # Recreating class example

        # generatePicturePoints(f, xp, yp, omega, phi, kappa, X0, Y0, Z0, ground_points)
        picture_points = dlt.generatePicturePoints(153, 0.2, 0.2, 60, 45, -30, 15, 10, 50, ground_box_points)
        print('\nGround points of the box (x,y,z)\n', pd.DataFrame(ground_box_points))
        print('\nPicture points of the box (x,y)\n', pd.DataFrame(picture_points))

        exterior = dlt.solveExteriorOrientation(picture_points, ground_box_points)
        print('\nExterior Orientation f,xp,xy,omega,phi,kappa,X0,Y0,Z0\n', pd.DataFrame(exterior))

    ### --- Checking the effect of the observation axis system on the DLT EOP extraction --- ###
    obsSystemEffect = False
    if obsSystemEffect:
        # generatePicturePoints(f, xp, yp, omega, phi, kappa, X0, Y0, Z0, ground_points)
        picture_points = dlt.generatePicturePoints(153, 0.2, 0.2, 60, 45, -30, 15, 10, 50, ground_box_points)
        # assuming image size of 800x800 mm we want to transform the system to an upper left system:
        picture_points = np.array([0, 800]) + (picture_points - np.array([-400, 400]))  # * np.array([1, -1])
        # computing EOP using DLT:
        exterior = dlt.solveExteriorOrientation(picture_points, ground_box_points)
        print('\nExterior Orientation f,xp,xy,omega,phi,kappa,X0,Y0,Z0\n', pd.DataFrame(exterior))

    ### --- Checking the effect of the control network axis system on the DLT EOP extraction --- ###
    ctrlNetworkEffect = False
    if ctrlNetworkEffect:
        # adding noise to ground points
        ground_box_points = dlt.addNoise(ground_box_points)
        # generatePicturePoints(f, xp, yp, omega, phi, kappa, X0, Y0, Z0, ground_points)
        picture_points = dlt.generatePicturePoints(153, 0.2, 0.2, 60, 45, -30, 15, 10, 50, ground_box_points)
        # computing EOP using DLT:
        exterior = dlt.solveExteriorOrientation(picture_points, ground_box_points)
        print('\nExterior Orientation f,xp,xy,omega,phi,kappa,X0,Y0,Z0\n', pd.DataFrame(exterior))

        # now we move the ctrl ntwrk to a far point. say 400,400:
        real_eop = np.array([153, 0.2, 0.2, 60, 45, -30, 15, 10, 50])
        ground_box_points = ground_box_points - np.array([400, 400, 0])
        picture_points = dlt.generatePicturePoints(153, 0.2, 0.2, 60, 45, -30, 15, 10, 50, ground_box_points)
        exterior = dlt.solveExteriorOrientation(picture_points, ground_box_points)
        print('\nExterior Orientation f,xp,xy,omega,phi,kappa,X0,Y0,Z0\n', pd.DataFrame(exterior))
        print('\n', pd.DataFrame(np.abs(exterior - real_eop)))

    ### --- Checking the difference between euclidean EOP extraction and DLT --- ###
    euclidean_vs_DLT = True
    if euclidean_vs_DLT:
        # def __init__(self, focal_length, principal_point, radial_distortions, decentering_distortions, fiducial_marks,
        #                  sensor_size)

        cam = Camera(153, [0.2, 0.2], None, None, None, 800)
        img = SingleImage(cam)
        img.exteriorOrientationParameters = (
            [15, 10, 50, np.deg2rad(-0), np.deg2rad(0), np.deg2rad(0)])  # EOP real values
        ground_box_points = ground_box_points - np.array([20, 20, 0])
        picture_points = dlt.generatePicturePoints(153, 0.2, 0.2, 0, 0, -0, 15, 10, 50, ground_box_points)

        projection = img.GroundToImage(ground_box_points)
        # projection = np.array([0, 800]) + (projection - np.array([-400, 400]))

        real_eop = np.array([153, 0.2, 0.2, 15, 10, 50, 0, 0, 0])
        extOri = img.ComputeExteriorOrientation(projection, ground_box_points.T, 1e-6)
        extOri[0][3::] = np.rad2deg(extOri[0][3::])
        print(pd.DataFrame(extOri[0]))
        print(pd.DataFrame(np.abs(extOri[0] - real_eop[3::])))
