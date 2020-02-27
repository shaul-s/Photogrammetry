import numpy as np
from Camera import Camera
from scipy import linalg as la


class SingleImage(object):

    def __init__(self, camera, dim, cp):
        """
        Initialize the SingleImage object

        :param camera: instance of the Camera class
        :param dim: dimensions of image cols x rows
        :param cp: array of sampled control points in pix

        :type camera: Camera
        :type dim: np.array 1x2
        :type cp: np.array nx2

        """
        self.__camera = camera
        self.__dim = dim
        self.__cp = cp
        self.__isSolved = False
        self.__exteriorOrientationParameters = np.array([0, 0, 0, 0, 0, 0], 'f')
        self.__rotationMatrix = None

    @property
    def camera(self):
        """
        The camera that took the image

        :rtype: Camera

        """
        return self.__camera

    @camera.setter
    def camera(self, val):
        """
        Set the camera parameters

        :param val: value for setting

        :type: dict

        """

        self.__camera = val

    @property
    def exteriorOrientationParameters(self):
        r"""
        Property for the exterior orientation parameters

        :return: exterior orientation parameters in the following order, **however you can decide how to hold them (dictionary or array)**

        .. math::
            exteriorOrientationParameters = \begin{bmatrix} X_0 \\ Y_0 \\ Z_0 \\ \omega \\ \varphi \\ \kappa \end{bmatrix}

        :rtype: np.ndarray or dict
        """
        return self.__exteriorOrientationParameters

    @exteriorOrientationParameters.setter
    def exteriorOrientationParameters(self, parametersArray):
        r"""

        :param parametersArray: the parameters to update the ``self.__exteriorOrientationParameters``

        **Usage example**

        .. code-block:: py

            self.exteriorOrintationParameters = parametersArray

        """
        self.__exteriorOrientationParameters = parametersArray

    @property
    def rotationMatrix(self):
        """
        The rotation matrix of the image

        Relates to the exterior orientation
        :return: rotation matrix

        :rtype: np.ndarray (3x3)
        """

        R = self.__Compute3DRotationMatrix(self.exteriorOrientationParameters[3], self.exteriorOrientationParameters[4],
                                           self.exteriorOrientationParameters[5])

        return R

    @property
    def isSolved(self):
        """
        True if the exterior orientation is solved

        :return True or False

        :rtype: boolean
        """
        return self.__isSolved

    def ComputeApproximateVals(self, groundPoints):
        """
        Compute exterior orientation approximate values via 2-D conform transformation

        :param imagePoints: points in image space (x y)
        :param groundPoints: corresponding points in world system (X, Y, Z)

        :type imagePoints: np.ndarray [nx2]
        :type groundPoints: np.ndarray [nx3]

        :return: Approximate values of exterior orientation parameters
        :rtype: np.ndarray or dict

        .. note::

            - ImagePoints should be transformed to ideal camera using ``self.ImageToCamera(imagePoints)``. See code below
            - The focal length is stored in ``self.camera.focalLength``
            - Don't forget to update ``self.exteriorOrientationParameters`` in the order defined within the property
            - return values can be a tuple of dictionaries and arrays.
        """

        # Find approximate values
        cameraPoints = self.ApproximateImg2Camera(self.__cp).reshape(np.size(self.__cp), 1)
        groundPointsXY = groundPoints[:, 0:2]
        groundPointsXY = groundPointsXY.reshape(np.size(groundPointsXY), 1)
        groundPointsZ = groundPoints[:, 2]

        n = int(len(cameraPoints))  # number of observations
        u = 4  # 4 conform parameters

        A = np.zeros((n, u))  # A matrix (n,u)

        j = 0
        for i in range(len(cameraPoints)):
            if i % 2 == 0:
                A[i, 0] = 1
                A[i, 1] = 0
                A[i, 2] = cameraPoints[j]
                A[i, 3] = cameraPoints[j + 1]
            else:
                A[i, 0] = 0
                A[i, 1] = 1
                A[i, 2] = cameraPoints[j + 1]
                A[i, 3] = -cameraPoints[j]
                j += 2

        X = np.dot(la.inv(np.dot(np.transpose(A), A)), np.dot(np.transpose(A), groundPointsXY))

        #  now we can compute the rest of the params
        X0 = X[0]
        Y0 = X[1]
        kappa = np.arctan2(-X[3], X[2])
        lam = np.sqrt(X[2] ** 2 + X[3] ** 2)
        Z0 = np.average(groundPointsZ) + (lam) * self.camera.focalLength

        # adjustment_results = {"X0": X0[0], "Y0": Y0[0], "Z0": Z0[0], "omega": 0, "phi": 0,
        #          "kappa": np.rad2deg(kappa[0])}

        self.__exteriorOrientationParameters = np.array(
            [X0[0], Y0[0], Z0[0], 0, 0, kappa[0]]).T  # updating the exterior orientation params

    def ImageToCamera(self, imagePoints):
        """

        Transforms image points to ideal camera points

        :param imagePoints: image points

        :type imagePoints: np.array nx2

        :return: corresponding camera points

        :rtype: np.array nx2

        .. warning::

            This function is empty, need implementation

        .. note::

            The inner orientation parameters required for this function are held in ``self.innerOrientationParameters``

        """
        inverse_pars = self.ComputeInverseInnerOrientation()
        imagePoints = imagePoints.T

        if imagePoints.size == 2:
            imagePoints = np.reshape(np.array(imagePoints), (np.size(imagePoints), 1))

        T = np.array([[inverse_pars[0]], [inverse_pars[1]]])
        R = np.array([[inverse_pars[2], inverse_pars[3]], [inverse_pars[4], inverse_pars[5]]])

        return (np.dot(R, imagePoints - T)).T

    def ComputeExteriorOrientation(self, imagePoints, groundPoints, epsilon):
        """
        Compute exterior orientation parameters.

        This function can be used in conjecture with ``self.__ComputeDesignMatrix(groundPoints)`` and ``self__ComputeObservationVector(imagePoints)``

        :param imagePoints: image points
        :param groundPoints: corresponding ground points

            .. note::

                Angles are given in radians

        :param epsilon: threshold for convergence criteria

        :type imagePoints: np.array nx2
        :type groundPoints: np.array nx3
        :type epsilon: float

        :return: Exterior orientation parameters: (X0, Y0, Z0, omega, phi, kappa), their accuracies, and residuals vector. *The orientation parameters can be either dictionary or array -- to your decision*

        :rtype: dict


        .. warning::

           - This function is empty, need implementation
           - Decide how the parameters are held, don't forget to update documentation

        .. note::

            - Don't forget to update the ``self.exteriorOrientationParameters`` member (every iteration and at the end).
            - Don't forget to call ``cameraPoints = self.ImageToCamera(imagePoints)`` to correct the coordinates that are sent to ``self.__ComputeApproximateVals(cameraPoints, groundPoints)``
            - return values can be a tuple of dictionaries and arrays.

        """
        cameraPoints = self.ImageToCamera(imagePoints)
        self.__ComputeApproximateVals(cameraPoints, groundPoints)
        l0 = self.__ComputeObservationVector(groundPoints.T)
        l0 = np.reshape(l0, (-1, 1))
        l = cameraPoints.reshape(np.size(cameraPoints), 1) - l0
        A = self.__ComputeDesignMatrix(groundPoints.T)

        N = np.dot(A.T, A)
        u = np.dot(A.T, l)
        deltaX = np.dot(la.inv(N), u)

        # update orientation pars
        self.__exteriorOrientationParameters = np.add(self.__exteriorOrientationParameters, np.reshape(deltaX, 6))

        while la.norm(deltaX) > epsilon:
            l0 = self.__ComputeObservationVector(groundPoints.T)
            l0 = np.reshape(l0, (-1, 1))
            l = cameraPoints.reshape(np.size(cameraPoints), 1) - l0
            A = self.__ComputeDesignMatrix(groundPoints.T)
            N = np.dot(A.T, A)
            u = np.dot(A.T, l)
            deltaX = np.dot(la.inv(N), u)
            # update orientation pars
            self.__exteriorOrientationParameters = np.add(self.__exteriorOrientationParameters, np.reshape(deltaX, 6))

        # compute residuals
        l_a = np.reshape(self.__ComputeObservationVector(groundPoints.T), (-1, 1))
        v = l_a - cameraPoints.reshape(np.size(cameraPoints), 1)
        if (np.size(A, 0) - np.size(deltaX)) != 0:
            sig = np.dot(v.T, v) / (np.size(A, 0) - np.size(deltaX))
            sigmaX = sig[0] * la.inv(N)
        else:
            sigmaX = None

        return [self.exteriorOrientationParameters, sigmaX, v]

    def GroundToImage(self, groundPoints):
        """
        Transforming ground points to image points

        :param groundPoints: ground points [m]

        :type groundPoints: np.array nx3

        :return: corresponding Image points

        :rtype: np.array nx2

        """
        X0 = float(self.exteriorOrientationParameters[0])
        Y0 = float(self.exteriorOrientationParameters[1])
        Z0 = float(self.exteriorOrientationParameters[2])

        xp = float(self.camera.principalPoint[0])
        yp = float(self.camera.principalPoint[1])

        R = self.rotationMatrix
        r11 = float(R[0, 0])
        r12 = float(R[0, 1])
        r13 = float(R[0, 2])
        r21 = float(R[1, 0])
        r22 = float(R[1, 1])
        r23 = float(R[1, 2])
        r31 = float(R[2, 0])
        r32 = float(R[2, 1])
        r33 = float(R[2, 2])

        f = self.camera.focalLength

        camPoints = []

        for i in range(groundPoints.shape[0]):
            x = xp - (f) * (((r11 * (groundPoints[i, 0] - X0) + r21 * (groundPoints[i, 1] - Y0) + r31 * (
                    groundPoints[i, 2] - Z0)) / (r13 * (groundPoints[i, 0] - X0) + r23 * (
                    groundPoints[i, 1] - Y0) + r33 * (groundPoints[i, 2] - Z0))))
            y = yp - (f) * (((r12 * (groundPoints[i, 0] - X0) + r22 * (groundPoints[i, 1] - Y0) + r32 * (
                    groundPoints[i, 2] - Z0)) / (r13 * (groundPoints[i, 0] - X0) + r23 * (
                    groundPoints[i, 1] - Y0) + r33 * (groundPoints[i, 2] - Z0))))

            camPoints.append([x, y])

        return self.CameraToImage(np.array(camPoints))

    # ---------------------- Private methods ----------------------

    def lb(self):
        return self.ApproximateImg2Camera(self.__cp).reshape(np.size(self.__cp), 1)

    def ApproximateImg2Camera(self, image_points):
        """
        Compute approximate points in the camera system

        :param image_points: the points in the image system

        :type image_points: np.ndarray[nx2]

        :return: points in camera system [mm]
        """
        camera_points = []

        cols = self.__dim[0]
        rows = self.__dim[1]
        scale = self.camera.pixelSize

        for point in image_points:
            x = scale * (point[0] - cols / 2) - self.camera.principalPoint[0]
            y = -scale * (point[1] - rows / 2) - self.camera.principalPoint[1]
            camera_points.append([x, y])

        return np.array(camera_points)

    def ComputeObservationVector(self, groundPoints):
        """
        Compute observation vector for solving the exterior orientation parameters of a single image
        based on their approximate values

        :param groundPoints: Ground coordinates of the control points

        :type groundPoints: np.array nx3

        :return: Vector l0

        :rtype: np.array nx1
        """

        n = groundPoints.shape[0]  # number of points

        # Coordinates subtraction
        dX = groundPoints[:, 0] - self.exteriorOrientationParameters[0]
        dY = groundPoints[:, 1] - self.exteriorOrientationParameters[1]
        dZ = groundPoints[:, 2] - self.exteriorOrientationParameters[2]
        dXYZ = np.vstack([dX, dY, dZ])
        rotated_XYZ = np.dot(self.rotationMatrix.T, dXYZ).T

        l0 = np.zeros((n * 2, 1))
        lb = self.ApproximateImg2Camera(self.__cp).reshape(np.size(self.__cp), 1)
        j = 0
        for i in range(n):
            r = ((lb[j] - self.camera.parameters["xp"]) * (lb[j] - self.camera.parameters["xp"]) + (
                    lb[j + 1] - self.camera.parameters["yp"]) * (lb[j + 1] - self.camera.parameters["yp"])) ** 0.5
            l0[j] = self.camera.parameters["xp"] - self.camera.parameters["f"] * (
                    rotated_XYZ[i, 0] / rotated_XYZ[i, 2]) + \
                    lb[j] * (
                            self.camera.parameters["k1"] * r ** 2 + self.camera.parameters["k2"] * r ** 4 +
                            self.camera.parameters[
                                "k3"] * r ** 6) + self.camera.parameters["p1"] * (r ** 2 + 2 * lb[j] ** 2) + 2 * \
                    self.camera.parameters[
                        "p2"] * lb[j] * lb[j + 1] + self.camera.parameters["b1"] * lb[j] + self.camera.parameters[
                        "b2"] * lb[
                        j + 1]
            l0[j + 1] = self.camera.parameters["yp"] - self.camera.parameters["f"] * (rotated_XYZ[i, 1] / rotated_XYZ[
                i, 2]) + lb[
                            j + 1] * (self.camera.parameters["k1"] * r ** 2 + self.camera.parameters["k2"] * r ** 4 +
                                      self.camera.parameters[
                                          "k3"] * r ** 6) + 2 * self.camera.parameters["p1"] * lb[j] * lb[j + 1] + \
                        self.camera.parameters["p2"] * (
                                r ** 2 + 2 * lb[j + 1] * lb[j + 1])
            j += 2

        # Computation of the observation vector based on approximate exterior orientation parameters:
        return l0

    def ComputeDesignMatrix(self, groundPoints):
        """
            Compute the derivatives of the collinear law (design matrix)

            :param groundPoints: Ground coordinates of the control points

            :type groundPoints: np.array nx3

            :return: The design matrix

            :rtype: np.array nx6

        """
        # initialization for readability
        omega = self.exteriorOrientationParameters[3]
        phi = self.exteriorOrientationParameters[4]
        kappa = self.exteriorOrientationParameters[5]

        # Coordinates subtraction
        dX = groundPoints[:, 0] - self.exteriorOrientationParameters[0]
        dY = groundPoints[:, 1] - self.exteriorOrientationParameters[1]
        dZ = groundPoints[:, 2] - self.exteriorOrientationParameters[2]
        dXYZ = np.vstack([dX, dY, dZ])

        rotationMatrixT = self.rotationMatrix.T
        rotatedG = rotationMatrixT.dot(dXYZ)
        rT1g = rotatedG[0, :]
        rT2g = rotatedG[1, :]
        rT3g = rotatedG[2, :]

        focalBySqauredRT3g = self.camera.parameters["f"] / rT3g ** 2

        dxdg = rotationMatrixT[0, :][None, :] * rT3g[:, None] - rT1g[:, None] * rotationMatrixT[2, :][None, :]
        dydg = rotationMatrixT[1, :][None, :] * rT3g[:, None] - rT2g[:, None] * rotationMatrixT[2, :][None, :]

        dgdX0 = np.array([-1, 0, 0], 'f')
        dgdY0 = np.array([0, -1, 0], 'f')
        dgdZ0 = np.array([0, 0, -1], 'f')

        # Derivatives with respect to X0
        dxdX0 = -focalBySqauredRT3g * np.dot(dxdg, dgdX0)
        dydX0 = -focalBySqauredRT3g * np.dot(dydg, dgdX0)

        # Derivatives with respect to Y0
        dxdY0 = -focalBySqauredRT3g * np.dot(dxdg, dgdY0)
        dydY0 = -focalBySqauredRT3g * np.dot(dydg, dgdY0)

        # Derivatives with respect to Z0
        dxdZ0 = -focalBySqauredRT3g * np.dot(dxdg, dgdZ0)
        dydZ0 = -focalBySqauredRT3g * np.dot(dydg, dgdZ0)

        dRTdOmega = self.__Compute3DRotationDerivativeMatrix(omega, phi, kappa, 'omega').T
        dRTdPhi = self.__Compute3DRotationDerivativeMatrix(omega, phi, kappa, 'phi').T
        dRTdKappa = self.__Compute3DRotationDerivativeMatrix(omega, phi, kappa, 'kappa').T

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

        # all derivatives of x and y
        dd = np.array([np.vstack([dxdX0, dxdY0, dxdZ0, dxdOmega, dxdPhi, dxdKappa]).T,
                       np.vstack([dydX0, dydY0, dydZ0, dydOmega, dydPhi, dydKappa]).T])

        n = 2 * dd[0].shape[0]
        a = np.zeros((n, 6))
        a1 = np.zeros((n, 10))

        a[0::2] = dd[0]
        a[1::2] = dd[1]

        j = 0
        k = -1
        lb = self.ApproximateImg2Camera(self.__cp).reshape(np.size(self.__cp), 1)

        for i in range(int(n / 2)):
            k += 1
            r = ((lb[j] - self.camera.parameters["xp"]) * (lb[j] - self.camera.parameters["xp"]) + (
                    lb[j + 1] - self.camera.parameters["yp"]) * (lb[j + 1] - self.camera.parameters["yp"])) ** 0.5
            a1[j, 0] = -rT1g[k] / rT3g[k]
            a1[j, 1] = 1 + 2 * (lb[j] - self.camera.parameters["xp"]) * (lb[j] * (
                    self.camera.parameters["k1"] + 2 * self.camera.parameters["k2"] * r ** 2 + 3 *
                    self.camera.parameters["k3"] * r ** 4) + self.camera.parameters["p1"])
            a1[j, 2] = 2 * (lb[j + 1] - self.camera.parameters["yp"]) * (lb[j] * (
                    self.camera.parameters["k1"] + 2 * self.camera.parameters["k2"] * r ** 2 + 3 *
                    self.camera.parameters["k3"] * r ** 4) + self.camera.parameters["p1"])
            a1[j, 3] = lb[j] * r ** 2
            a1[j, 4] = lb[j] * r ** 4
            a1[j, 5] = lb[j] * r ** 6
            a1[j, 6] = r ** 2 + 2 * lb[j] ** 2
            a1[j, 7] = 2 * lb[j] * lb[j + 1]
            a1[j, 8] = lb[j]
            a1[j, 9] = lb[j + 1]
            a1[j + 1, 0] = -rT2g[k] / rT3g[k]
            a1[j + 1, 1] = 2 * (lb[j] - self.camera.parameters["xp"]) * (lb[j + 1] * (
                    self.camera.parameters["k1"] + 2 * self.camera.parameters["k2"] * r ** 2 + 3 *
                    self.camera.parameters["k3"] * r ** 4) + self.camera.parameters["p1"])
            a1[j + 1, 2] = 1 + 2 * (lb[j + 1] - self.camera.parameters["yp"]) * (lb[j + 1] * (
                    self.camera.parameters["k1"] + 2 * self.camera.parameters["k2"] * r ** 2 + 3 *
                    self.camera.parameters["k3"] * r ** 4) + self.camera.parameters["p2"])
            a1[j + 1, 3] = lb[j + 1] * r ** 2
            a1[j + 1, 4] = lb[j + 1] * r ** 4
            a1[j + 1, 5] = lb[j + 1] * r ** 6
            a1[j + 1, 6] = 2 * lb[j] * lb[j + 1]
            a1[j + 1, 7] = r ** 2 + 2 * lb[j + 1] * lb[j + 1]
            a1[j + 1, 8] = 0
            a1[j + 1, 9] = 0
            j += 2

        return np.hstack((a, a1))

    def __Compute3DRotationMatrix(self, omega, phi, kappa):
        """
        Computes a 3x3 rotation matrix defined by euler angles given in radians
        :param omega: Rotation angle around the x-axis (radians)
        :param phi: Rotation angle around the y-axis (radians)
        :param kappa: Rotation angle around the z-axis (radians)

        :type omega: float
        :type phi: float
        :type kappa: float


        :return: The corresponding 3D rotation matrix
        :rtype: array  (3x3)

        """
        # Rotation matrix around the x-axis
        rOmega = np.array([[1, 0, 0],
                           [0, np.cos(omega), -np.sin(omega)],
                           [0, np.sin(omega), np.cos(omega)]], 'f')

        # Rotation matrix around the y-axis
        rPhi = np.array([[np.cos(phi), 0, np.sin(phi)],
                         [0, 1, 0],
                         [-np.sin(phi), 0, np.cos(phi)]], 'f')

        # Rotation matrix around the z-axis
        rKappa = np.array([[np.cos(kappa), -np.sin(kappa), 0],
                           [np.sin(kappa), np.cos(kappa), 0],
                           [0, 0, 1]], 'f')

        return np.dot(np.dot(rOmega, rPhi), rKappa)

    def __Compute3DRotationDerivativeMatrix(self, omega, phi, kappa, var):
        r"""
        Computing the derivative of the 3D rotaion matrix defined by the euler angles according to one of the angles

        :param omega: Rotation angle around the x-axis (radians)
        :param phi: Rotation angle around the y-axis (radians)
        :param kappa: Rotation angle around the z-axis (radians)
        :param var: Name of the angle to compute the derivative by (omega/phi/kappa) (string)

        :type omega: float
        :type phi: float
        :type kappa: float
        :type var: str

        :return: The corresponding derivative matrix (3x3, ndarray). If var is not one of euler angles, the method will return None


        **Examples**

        1. Derivative matrix with respect to :math:\omega:

            .. code-block:: py

                from numpy import pi
                Compute3DRotationDerivativeMatrix(5 * pi / 180.0, 2 * pi / 180.0, 30 * pi / 180.0, 'omega')

        2. Derivative matrix with respect to :math:\varphi:

            .. code-block:: py

                from numpy import pi
                Compute3DRotationDerivativeMatrix(5 * pi / 180.0, 2 * pi / 180.0, 30 * pi / 180.0, 'phi')

        3. Derivative matrix with respect to :math:\kappa:

            .. code-block:: py

                from numpy import pi
                Compute3DRotationDerivativeMatrix(5 * pi / 180.0, 2 * pi / 180.0, 30 * pi / 180.0, 'kappa')

        """
        # Rotation matrix around the x-axis
        rOmega = np.array([[1, 0, 0],
                           [0, np.cos(omega), -np.sin(omega)],
                           [0, np.sin(omega), np.cos(omega)]], 'f')

        # Rotation matrix around the y-axis
        rPhi = np.array([[np.cos(phi), 0, np.sin(phi)],
                         [0, 1, 0],
                         [-np.sin(phi), 0, np.cos(phi)]], 'f')

        # Rotation matrix around the z-axis
        rKappa = np.array([[np.cos(kappa), -np.sin(kappa), 0],
                           [np.sin(kappa), np.cos(kappa), 0],
                           [0, 0, 1]], 'f')

        # Computing the derivative matrix based on requested parameter
        if (var == 'omega'):
            d = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], 'f')
            res = np.dot(d, np.dot(rOmega, np.dot(rPhi, rKappa)))
        elif (var == 'phi'):
            d = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], 'f')
            res = np.dot(rOmega, np.dot(d, np.dot(rPhi, rKappa)))
        elif (var == 'kappa'):
            d = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], 'f')
            res = np.dot(rOmega, np.dot(rPhi, np.dot(d, rKappa)))
        else:
            res = None

        return res


if __name__ == '__main__':
    pass
