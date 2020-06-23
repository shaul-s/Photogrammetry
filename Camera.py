import numpy as np
import cv2
import glob
from MatrixMethods import *
from SingleImage import *
from ObjectsSynthetic import *


class Camera(object):

    def __init__(self, focal_length, principal_point, radial_distortions, decentering_distortions, fiducial_marks,
                 sensor_size):
        """
        Initialize the Camera object

        :param focal_length: focal length of the camera(mm)
        :param principal_point: principle point 1x2 (mm)
        :param radial_distortions: the radial distortion parameters K0, K1, K2 ...
        :param decentering_distortions: decentering distortion parameters P0, P1, P2 ...
        :param fiducial_marks: fiducial marks in camera space
        :param sensorSize: size of sensor

        :type focal_length: double
        :type principal_point: np.array
        :type radial_distortions: dict
        :type decentering_distortions: dict
        :type fiducial_marks: np.array
        :type sensorSize: double

        """
        # private parameters
        self.__focal_length = focal_length
        self.__principal_point = principal_point
        self.__radial_distortions = radial_distortions
        self.__decentering_distortions = decentering_distortions
        self.__fiducial_marks = fiducial_marks
        self.__CalibrationParam = None
        self.__sensor_size = sensor_size

        # Build calibration matrix K
        self.K = np.zeros((3, 3))

    def __updateK(self):
        self.K = np.diag(np.array([self.focalLength, self.focalLength, 1]))
        self.K[:2, 2] = self.principalPoint

    @property
    def focalLength(self):
        """
        Focal length of the camera

        :return: focal length

        :rtype: float

        """
        return self.__focal_length

    @focalLength.setter
    def focalLength(self, val):
        """
        Set the focal length value

        :param val: value for setting

        :type: float

        """

        self.__focal_length = val

    @property
    def fiducialMarks(self):
        """
        Fiducial marks of the camera, by order

        :return: fiducial marks of the camera

        :rtype: np.array nx2

        """

        return self.__fiducial_marks

    @property
    def principalPoint(self):
        """
        Principal point of the camera

        :return: principal point coordinates

        :rtype: np.ndarray

        """

        return self.__principal_point

    @principalPoint.setter
    def principalPoint(self, values_array):
        """
        :param values_array: xp, yp
        :type: np.array

        """
        self.__principal_point = values_array

    @property
    def calibrationMatrix(self):
        """
        calibration matrix K
        :return: K ndarray 3x3
        """
        return np.array([[-self.focalLength, 0, self.principalPoint[0]],
                         [0, -self.focalLength, self.principalPoint[1]], [0, 0, 1]])

    @property
    def sensorSize(self):
        """
        Sensor size of the camera

        :return: sensor size

        :rtype: float

        """
        return self.__sensor_size

    @property
    def radialDistortions(self):
        """
        radial distortions K1 and K2

        :return: radial distortions K1 and K2

        :rtype: dict

        """
        return self.__radial_distortions

    @property
    def decenteringDistortions(self):
        """
        decentring distortions K1 and K2

        :return: decentring distortions K1 and K2

        :rtype: dict

        """
        return self.__decentering_distortions

    @staticmethod
    def calibrateCamera_checkers():

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((7 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:180:20, 0:140:20].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        images = glob.glob('calibration_images/*.jpg')

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 7), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                # img = cv2.drawChessboardCorners(img, (9, 7), corners2, ret)
                # cv2.imshow('img', img)
                # cv2.waitKey(1)

            # cv2.destroyAllWindows()

        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error

        print("total error: ", mean_error / len(objpoints))

        return K, rvecs, tvecs

    def CorrectionToRadialDistortions(self, camera_points):
        """
        Fixing Radial distortion on camera points
        :param camera_points: camera points
        :return: fixed camera points
        """
        angle = np.arctan2(camera_points[:, 1], camera_points[:, 0])  # angle for each point
        range = np.sqrt(camera_points[:, 1] ** 2 + camera_points[:, 0] ** 2)  # range from center each point
        # distortion = self.radial_distortions[0]*range**3+self.radial_distortions[1]*range**5

        distortion = self.radialDistortions['K1'] * range ** 3 + self.radialDistortions[
            'K2'] * range ** 5  # distortion size
        camera_points[:, 0] = camera_points[:, 0] - distortion * np.cos(angle)  # fix for x
        camera_points[:, 1] = camera_points[:, 1] - distortion * np.sin(angle)  # fix for y
        return camera_points

    def CameraToIdealCamera(self, camera_points):
        """
        Transform camera coordinates to an ideal system.
        currently supports correction to ppa and radial distortions

        :param camera_points: set of points in camera space

        :type camera_points: np.array nx2

        :return: fixed point set

        :rtype: np.array nx2

        .. warning::

            This function is empty, need implementation
        """
        return self.CorrectionToRadialDistortions(self.CorrectionToPrincipalPoint(camera_points))

    def IdealCameraToCamera(self, camera_points):
        r"""
        Transform from ideal camera to camera with distortions

        :param camera_points: points in ideal camera space

        :type camera_points: np.array nx2

        :return: corresponding points in image space

        :rtype: np.array nx2

        .. warning::

            This function is empty, need implementation
        """
        pass  # delete for implementation

    def ComputeDecenteringDistortions(self, camera_points):
        """
        Compute decentering distortions for given points

        :param camera_points: points in camera space

        :type camera_points: np.array nx2

        :return: decentering distortions: d_x, d_y

        :rtype: tuple of np.array

        .. warning::

            This function is empty, need implementation
        """
        pass  # delete for implementation

    def ComputeRadialDistortions(self, camera_points):
        """
        Compute radial distortions for given points

        :param camera_points: points in camera space

        :type camera_points: np.array nx2

        :return: radial distortions: delta_x, delta_y

        :rtype: tuple of np.array

        """
        pass  # delete for implementation

    def CorrectionToPrincipalPoint(self, camera_points):
        """
        Correcting camera points
        :param camera_points: np.ndarray (n,2/3)
        :return: np.ndarray (n,2/3)
        """
        camera_points[:, 0] = camera_points[:, 0] - self.principalPoint[0]
        camera_points[:, 1] = camera_points[:, 1] - self.principalPoint[1]
        return camera_points

    def cameraSysCorners(self):
        """
        sensor size in mm
        camera system corners of the sensor
        :return:
        """
        sens = self.sensorSize
        a = [sens / 2, -sens / 2, -self.focalLength]
        b = [-sens / 2, -sens / 2, -self.focalLength]
        c = [-sens / 2, sens / 2, -self.focalLength]
        d = [sens / 2, sens / 2, -self.focalLength]

        return np.array([a, b, c, d])

    def Calibration(self, approx_values, groundPoints, cameraPoints, epsilon=1e-6):
        def updateValues(apx, dX):
            """
            updating the approx values in the specific order
            :param apx: approx values for EOP & IOP
            :param dX: deltaX vector from non-linear least square adjustment
            :return: updated dictionary with EOP & IOP
            """
            dX = np.reshape(dX.astype(float), (dX.shape[0]))
            apx['f'] += dX[0]
            apx['xp'] += dX[1]
            apx['yp'] += dX[2]
            apx['K1'] += dX[3]
            apx['K2'] += dX[4]
            apx['X0'] += dX[5]
            apx['Y0'] += dX[6]
            apx['Z0'] += dX[7]
            apx['omega'] += dX[8]
            apx['phi'] += dX[9]
            apx['kappa'] += dX[10]

        # compute observations vector and design matrix
        l0 = self.__ComputeObservationVector(approx_values, groundPoints, cameraPoints)
        l = cameraPoints.reshape(np.size(cameraPoints), 1) - l0
        A = self.__ComputeDesignMatrix(approx_values, groundPoints, cameraPoints)

        # compute 1st iteration dX
        N = np.dot(A.T, A)
        u = np.dot(A.T, l)
        deltaX = np.dot(la.inv(N), u)

        # update orientation pars
        updateValues(approx_values, deltaX)

        # computing next iterations
        while la.norm(deltaX) > epsilon:
            # compute new observation vector & design matrix
            l0 = self.__ComputeObservationVector(approx_values, groundPoints, cameraPoints)
            l = cameraPoints.reshape(np.size(cameraPoints), 1) - l0
            A = self.__ComputeDesignMatrix(approx_values, groundPoints, cameraPoints)

            # compute next iterations
            N = np.dot(A.T, A)
            u = np.dot(A.T, l)
            deltaX = np.dot(la.inv(N), u)

            # update orientation pars
            updateValues(approx_values, deltaX)

        # compute residuals
        l_a = np.reshape(self.__ComputeObservationVector(approx_values, groundPoints, cameraPoints), (-1, 1))
        v = l_a - cameraPoints.reshape(np.size(cameraPoints), 1)
        if (np.size(A, 0) - np.size(deltaX)) != 0:
            sig = np.dot(v.T, v) / (np.size(A, 0) - np.size(deltaX))
            sigmaX = sig[0] * la.inv(N)
        else:
            sigmaX = None

        # update the camera object
        approx_values['K1'] = approx_values['K1'] * 1e-5
        approx_values['K2'] = approx_values['K2'] * 1e-10

        return [approx_values, sigmaX, v]

    def compute_CalibrationMatrix(self, v1, v2, v3):
        """
        Compute the calibration parameters based on the orthocenter of the triangle
        defined by three vanishing points

        :param v1: first vanishing point
        :param v2: second vanishing point
        :param v3: third vanishing point

        :type v1: np.array
        :type v2: np.array
        :type v3: np.array

        :return: calibration matrix

        :rtype: np.array 3x3
        """

        # Solve linear system with xp and yp as unknowns

        # matrix A
        A = np.array([[v3[0] - v2[0], v3[1] - v2[1]], [v1[0] - v2[0], v1[1] - v2[1]]])
        b = np.diag(A.dot(np.array([v1, v3])))
        x = np.linalg.solve(A, b)

        xp = x[0]
        yp = x[1]

        # Compute the focal length
        focal = np.sqrt(np.abs(- (v1[0:2] - x.flatten()).dot(v2[0:2] - x.flatten())))

        self.focalLength = focal
        self.principalPoint = np.array([xp, yp])
        self.__updateK()
        return self.K

    # ---------------------- Private methods ----------------------

    def __ComputeDesignMatrix(self, approx_values, groundPoints, cameraPoints):
        """
        Compute the derivatives of the collinear law (design matrix)

        :param approx_values: approximate values for EOP & IOP
        :param groundPoints: Ground coordinates of the control points
        :param cameraPoints: Sampled camera points

        :type approx_values: dict
        :type groundPoints: np.array nx3
        :type cameraPoints: np.array nx2

        :return: Matrix A

        :rtype: np.array nx11
        """
        # initialization for readability
        omega = approx_values['omega']
        phi = approx_values['phi']
        kappa = approx_values['kappa']

        # Coordinates subtraction
        dX = groundPoints[:, 0] - approx_values['X0']
        dY = groundPoints[:, 1] - approx_values['Y0']
        dZ = groundPoints[:, 2] - approx_values['Z0']
        dXYZ = np.vstack([dX, dY, dZ])

        rotationMatrixT = Compute3DRotationMatrix(omega, phi, kappa).T
        rotatedG = rotationMatrixT.dot(dXYZ)
        rT1g = rotatedG[0, :]
        rT2g = rotatedG[1, :]
        rT3g = rotatedG[2, :]

        focalBySqauredRT3g = approx_values['f'] / rT3g ** 2

        dxdg = rotationMatrixT[0, :][None, :] * rT3g[:, None] - rT1g[:, None] * rotationMatrixT[2, :][None, :]
        dydg = rotationMatrixT[1, :][None, :] * rT3g[:, None] - rT2g[:, None] * rotationMatrixT[2, :][None, :]

        dgdX0 = np.array([-1, 0, 0], 'f')
        dgdY0 = np.array([0, -1, 0], 'f')
        dgdZ0 = np.array([0, 0, -1], 'f')

        range = np.sqrt(cameraPoints[:, 1] ** 2 + cameraPoints[:, 0] ** 2)

        # Derivatives with respect to X0
        dxdX0 = -focalBySqauredRT3g * np.dot(dxdg, dgdX0)
        dydX0 = -focalBySqauredRT3g * np.dot(dydg, dgdX0)

        # Derivatives with respect to Y0
        dxdY0 = -focalBySqauredRT3g * np.dot(dxdg, dgdY0)
        dydY0 = -focalBySqauredRT3g * np.dot(dydg, dgdY0)

        # Derivatives with respect to Z0
        dxdZ0 = -focalBySqauredRT3g * np.dot(dxdg, dgdZ0)
        dydZ0 = -focalBySqauredRT3g * np.dot(dydg, dgdZ0)

        dRTdOmega = Compute3DRotationDerivativeMatrix(omega, phi, kappa, 'omega').T
        dRTdPhi = Compute3DRotationDerivativeMatrix(omega, phi, kappa, 'phi').T
        dRTdKappa = Compute3DRotationDerivativeMatrix(omega, phi, kappa, 'kappa').T

        gRT3g = dXYZ * rT3g

        # Derivatives with respect to Omega
        dxdOmega = -focalBySqauredRT3g * (dRTdOmega[0, :][None, :].dot(gRT3g) -
                                          rT1g * (dRTdOmega[2, :][None, :].dot(dXYZ)))[0]

        dydOmega = -focalBySqauredRT3g * (dRTdOmega[1, :][None, :].dot(gRT3g) -
                                          rT2g * (dRTdOmega[2, :][None, :].dot(dXYZ)))[0]

        # Derivatives with respect to Phi
        dxdPhi = -focalBySqauredRT3g * (dRTdPhi[0, :][None, :].dot(gRT3g) -
                                        rT1g * (dRTdPhi[2, :][None, :].dot(dXYZ)))[0]

        dydPhi = -focalBySqauredRT3g * (dRTdPhi[1, :][None, :].dot(gRT3g) -
                                        rT2g * (dRTdPhi[2, :][None, :].dot(dXYZ)))[0]

        # Derivatives with respect to Kappa
        dxdKappa = -focalBySqauredRT3g * (dRTdKappa[0, :][None, :].dot(gRT3g) -
                                          rT1g * (dRTdKappa[2, :][None, :].dot(dXYZ)))[0]

        dydKappa = -focalBySqauredRT3g * (dRTdKappa[1, :][None, :].dot(gRT3g) -
                                          rT2g * (dRTdKappa[2, :][None, :].dot(dXYZ)))[0]

        # Derivatives with respect to f
        dxdf = -rT1g / rT3g
        dydf = -rT2g / rT3g

        # Derivatives with respect to xp
        dxdxp = np.full((cameraPoints.shape[0]), 1)
        dxdyp = np.full((cameraPoints.shape[0]), 0)

        # Derivatives with respect to yp
        dydxp = np.full((cameraPoints.shape[0]), 0)
        dydyp = np.full((cameraPoints.shape[0]), 1)

        # Derivatives with respect to K1
        dxdK1 = -1e-5 * (cameraPoints[:, 0] - approx_values['xp']) * (range ** 2)
        dydK1 = -1e-5 * (cameraPoints[:, 1] - approx_values['yp']) * (range ** 2)

        # Derivatives with respect to K2
        dxdK2 = -1e-10 * (cameraPoints[:, 0] - approx_values['xp']) * (range ** 4)
        dydK2 = -1e-10 * (cameraPoints[:, 1] - approx_values['yp']) * (range ** 4)
        """
        # Derivatives with respect to K3
        dxdK3 = -1e-15 * (cameraPoints[:, 0] - approx_values['xp']) * (range ** 6)
        dydK3 = -1e-15 * (cameraPoints[:, 1] - approx_values['yp']) * (range ** 6)
        """
        # all derivatives of x and y
        dd = np.array(
            [np.vstack([dxdf, dxdxp, dxdyp, dxdK1, dxdK2, dxdX0, dxdY0, dxdZ0, dxdOmega, dxdPhi, dxdKappa]).T,
             np.vstack(
                 [dydf, dydxp, dydyp, dydK1, dydK2, dydX0, dydY0, dydZ0, dydOmega, dydPhi, dydKappa]).T])

        a = np.zeros((2 * dd[0].shape[0], len(approx_values)))
        a[0::2] = dd[0]
        a[1::2] = dd[1]

        return a

    def __ComputeObservationVector(self, approx_values, groundPoints, cameraPoints):
        """
        Compute observation vector for solving the exterior orientation parameters of a single image
        based on their approximate values.
        note that K1 & K2 are multiplied by 1e-5 & 1e-10 respectively

        :param approx_values: approximate values for EOP & IOP (6 + 5 vars)
        :param groundPoints: Ground coordinates of the control points
        :param cameraPoints: Sampled camera points

        :type approx_values: dict
        :type groundPoints: np.array nx3
        :type cameraPoints: np.array nx2

        :return: Vector l0

        :rtype: np.array nx1
        """
        # initialization for readability
        omega = approx_values['omega']
        phi = approx_values['phi']
        kappa = approx_values['kappa']

        n = groundPoints.shape[0]  # number of points

        # Coordinates subtraction
        dX = groundPoints[:, 0] - approx_values['X0']
        dY = groundPoints[:, 1] - approx_values['Y0']
        dZ = groundPoints[:, 2] - approx_values['Z0']
        dXYZ = np.vstack([dX, dY, dZ])
        rotated_XYZ = np.dot(Compute3DRotationMatrix(omega, phi, kappa).T, dXYZ).T

        range = np.sqrt(cameraPoints[:, 1] ** 2 + cameraPoints[:, 0] ** 2)

        # l0 initialization
        l0 = np.empty(n * 2)

        # Computation of the observation vector based on approximate EOP & IOP:
        l0[::2] = approx_values['xp'] - approx_values['f'] * rotated_XYZ[:, 0] / rotated_XYZ[:, 2] - \
                  (cameraPoints[:, 0] - approx_values['xp']) * (
                          1e-5 * approx_values['K1'] * (range ** 2) + 1e-10 * approx_values['K2'] * (range ** 4))

        l0[1::2] = approx_values['yp'] - approx_values['f'] * rotated_XYZ[:, 1] / rotated_XYZ[:, 2] - \
                   (cameraPoints[:, 1] - approx_values['yp']) * (
                           1e-5 * approx_values['K1'] * (range ** 2) + 1e-10 * approx_values['K2'] * (range ** 4))

        return np.reshape(l0, (l0.shape[0], 1))


if __name__ == '__main__':
    """
    f0 = 4360.
    xp0 = 2144.5
    yp0 = 1424.5
    K1 = 0
    K2 = 0
    P1 = 0
    P2 = 0
    
    # define the initial values vector
    f = 25  # in milimeters
    sensor_size = 60  # in milimeters
    cam = Camera(f, np.array([-0.001, 0.003]), {'K1': -0.5104e-8, 'K2': 0.1150e-12},
                 {'P1': -0.8776e-7, 'P2': 0.1722e-7}, None, sensor_size)

    img = SingleImage(cam)
    img.exteriorOrientationParameters = ([0, 0, 50, 0, 0, 0])
    pts = createCalibrationField(5, 3, 2)

    #projection = cam.CameraToIdealCamera(img.GroundToImage(pts))  # including correction to ppa + radial distortions
    projection = img.GroundToImage(pts)
    approx_values = {'f': 25, 'xp': -0.001, 'yp': 0.003, 'K1': -0.5104e-8, 'K2': 0.1150e-12,
                     'X0': 0.5, 'Y0': 0.5, 'Z0': 49, 'omega': 0, 'phi': 0, 'kappa': 0.001}

    cam.Calibration(approx_values, pts, projection)
    

    cam = Camera(25, np.array([-0.001, 0.003]), {'K1': -0.5104e-8, 'K2': 0.1150e-12},
                 {'P1': -0.8776e-7, 'P2': 0.1722e-7}, None, None)
    img = SingleImage(cam)
    img.exteriorOrientationParameters = ([0, 0, 50, 0, 0, 0])

    pts = createCalibrationField(5, 3, 2)

    projection = img.GroundToImage(pts)

    approx_values = {'f': 20, 'xp': 0, 'yp': 0, 'K1': 0.001, 'K2': 0.001,
                     'X0': 0.5, 'Y0': 0.5, 'Z0': 49, 'omega': 0.09, 'phi': 0.08, 'kappa': 0.5}

    # approx_values = {'f': 25, 'xp': 0, 'yp': 0, 'K1': 0, 'K2': 0,
    #                'X0': 0.5, 'Y0': 0.5, 'Z0': 49, 'omega': 0.8, 'phi': 0.6, 'kappa': 0.7}

    adjustment = cam.Calibration(approx_values, pts, projection)

    print(adjustment[0])
    print('\n')
    # print('The Q Accuracies matrix is:', adjustment[1])
    print('\n')
    # print('The residuals vector is:', adjustment[2])
    # print('hi')
    """
