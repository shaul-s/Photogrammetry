import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import rq
import pandas as pd


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
        picture_box_points_actual[:, 2] = picture_box_points_actual[:, 1] / picture_box_points_actual[:, 2]

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

        for row in range(0, A_rows, 2):  # Two rows each time

            # First restriction
            A[row, 4:8] = -ground_points[int(row / 2), :]
            A[row, 8:12] = ground_points[int(row / 2), :] * picture_points[int(row / 2), 1]

            # Second restriction
            A[row + 1, 0:4] = ground_points[int(row / 2), :]
            A[row + 1, 8:12] = -ground_points[int(row / 2), :] * picture_points[int(row / 2), 0]

        N = np.dot(A.T, A)

        egi_vals, egi_vect = np.linalg.eig(N)

        min_egi_val_index = np.argmin(egi_vals)

        v = egi_vect[:, min_egi_val_index]

        p = v.reshape((3, 4))

        c = -np.dot(np.linalg.inv(p[:, 0:3]), p[:, 3].reshape((3, 1)))

        r, q = rq(p[:, 0:3])
        r_fixed = np.round(np.dot(r / np.abs(r[2, 2]), np.diag([1, -1, -1])), 6)
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


if __name__ == '__main__':

    run_class_example = False
    if run_class_example:  # Recreating class example
        ground_box_points = np.array([[-5, -5, -5],
                                      [-5, 5, -5],
                                      [5, 5, -5],
                                      [5, -5, -5],
                                      [-5, -5, 5],
                                      [-5, 5, 5],
                                      [5, 5, 5],
                                      [5, -5, 5]])
        picture_points = dlt.generatePicturePoints(153, 0.2, 0.2, 60, 45, -30, 15, 10, 50, ground_box_points)
        print('\nPicture points of the box (x,y)\n', pd.DataFrame(picture_points))

    solve_exterior_dlt = True
    if solve_exterior_dlt:  # Recreating class example
        ground_box_points = np.array([[-5, -5, -5],
                                      [-5, 5, -5],
                                      [5, 5, -5],
                                      [5, -5, -5],
                                      [-5, -5, 5],
                                      [-5, 5, 5],
                                      [5, 5, 5],
                                      [5, -5, 5]])
        picture_points = dlt.generatePicturePoints(153, 0.2, 0.2, 60, 45, -30, 15, 10, 50, ground_box_points)
        print('\nGround points of the box (x,y,z)\n', pd.DataFrame(ground_box_points))
        print('\nPicture points of the box (x,y)\n', pd.DataFrame(picture_points))

        exterior = dlt.solveExteriorOrientation(picture_points, ground_box_points)
        print('\nExterior Orientation f,xp,xy,omega,phi,kappa,X0,Y0,Z0\n', pd.DataFrame(exterior))
