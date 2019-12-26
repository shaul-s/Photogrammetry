import numpy as np
from Camera import Camera
from MatrixMethods import Compute3DRotationMatrix, Compute3DRotationDerivativeMatrix
from scipy import linalg as la


class SingleImage(object):

    def __init__(self, camera):
        """
        Initialize the SingleImage object

        :param camera: instance of the Camera class
        :param points: points in image space

        :type camera: Camera
        :type points: np.array

        """
        self.__camera = camera
        self.__innerOrientationParameters = None
        self.__isSolved = False
        self.__exteriorOrientationParameters = np.array([0, 0, 0, 0, 0, 0], 'f')
        self.__rotationMatrix = None

    @property
    def innerOrientationParameters(self):
        """
        Inner orientation parameters


        .. warning::

            Can be held either as dictionary or array. For your implementation and decision.

        .. note::

            Do not forget to decide how it is held and document your decision

        :return: inner orinetation parameters

        :rtype: **ADD**
        """
        return self.__innerOrientationParameters

    @property
    def camera(self):
        """
        The camera that took the image

        :rtype: Camera

        """
        return self.__camera

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

        R = Compute3DRotationMatrix(self.exteriorOrientationParameters[3], self.exteriorOrientationParameters[4],
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

    def ComputeInnerOrientation(self, imagePoints):
        r"""
        Compute inner orientation parameters

        :param imagePoints: coordinates in image space

        :type imagePoints: np.array nx2

        :return: Inner orientation parameters, their accuracies, and the residuals vector

        :rtype: dict

        .. note::

            - Don't forget to update the ``self.__innerOrinetationParameters`` member. You decide the type
            - The fiducial marks are held within the camera attribute of the object, i.e., ``self.camera.fiducialMarks``
            - return values can be a tuple of dictionaries and arrays.

        **Usage example**

        .. code-block:: py

            fMarks = np.array([[113.010, 113.011],
                       [-112.984, -113.004],
                       [-112.984, 113.004],
                       [113.024, -112.999]])
            img_fmarks = np.array([[-7208.01, 7379.35],
                        [7290.91, -7289.28],
                        [-7291.19, -7208.22],
                        [7375.09, 7293.59]])
            cam = Camera(153.42, np.array([0.015, -0.020]), None, None, fMarks)
            img = SingleImage(camera = cam, points = None)
            inner_parameters, accuracies, residuals = img.ComputeInnerOrientation(img_fmarks)
        """
        #  implementing observation vectors
        imagePoints = imagePoints.reshape(np.size(imagePoints), 1)

        fMarks = self.camera.fiducialMarks.reshape(np.size(self.camera.fiducialMarks), 1)

        n = int(len(imagePoints))  # number of observations
        u = 6  # 6 orientation parameters

        A = np.zeros((n, u))  # A matrix (n,u)

        j = 0
        for i in range(len(imagePoints)) :
            if i % 2 == 0 :
                A[i,0] = 1; A[i,1] = 0; A[i,2] = fMarks[j]; A[i,3] = fMarks[j+1];
                A[i, 4] = 0
                A[i, 5] = 0
            else :
                A[i, 0] = 0; A[i, 1] = 1; A[i, 2] = 0; A[i, 3] = 0;
                A[i, 4] = fMarks[j];
                A[i, 5] = fMarks[j+1]
                j += 2

        X = np.dot(la.inv(np.dot(np.transpose(A), A)), np.dot(np.transpose(A), imagePoints))
        v = np.dot(A, X) - imagePoints

        adjustment_results = {"params": X, "residuals": v, "N": np.dot(np.transpose(A), A)}

        self.__innerOrientationParameters = X  # updating the inner orientation params

        return adjustment_results

    def ComputeGeometricParameters(self):
        """
        Computes the geometric inner orientation parameters

        :return: geometric inner orientation parameters

        :rtype: dict

        .. warning::

           This function is empty, need implementation

        .. note::

            The algebraic inner orinetation paramters are held in ``self.innerOrientatioParameters`` and their type
            is according to what you decided when initialized them

        """
        # extracting inner orientation params
        a0 = self.innerOrientationParameters[0]
        b0 = self.innerOrientationParameters[1]
        a1 = self.innerOrientationParameters[2]
        a2 = self.innerOrientationParameters[3]
        b1 = self.innerOrientationParameters[4]
        b2 = self.innerOrientationParameters[5]

        # computing algebric params
        tx = a0; ty = b0
        theta = np.arctan(b1/b2)
        gamma = np.arctan((a1*np.sin(theta)+a2*np.cos(theta))/(b1*np.sin(theta)+b2*np.cos(theta)))
        sx = a1*np.cos(theta)-a2*np.sin(theta)
        sy = (a1*np.sin(theta)+a2*np.cos(theta))/np.sin(gamma)

        return {"translationX": tx, "translationY": ty, "rotationAngle": np.rad2deg(theta), "scaleFactorX": sx, "scaleFactorY": sy, "shearAngle": np.rad2deg(gamma)}

    def ComputeInverseInnerOrientation(self):
        """
        Computes the parameters of the inverse inner orientation transformation

        :return: parameters of the inverse transformation

        :rtype: dict

        .. warning::

            This function is empty, need implementation

        .. note::

            The inner orientation algebraic parameters are held in ``self.innerOrientationParameters``
            their type is as you decided when implementing
        """
        a0 = self.innerOrientationParameters[0]
        b0 = self.innerOrientationParameters[1]
        a1 = self.innerOrientationParameters[2]
        a2 = self.innerOrientationParameters[3]
        b1 = self.innerOrientationParameters[4]
        b2 = self.innerOrientationParameters[5]

        mat = np.array([[a1[0], a2[0]],[b1[0],b2[0]]])
        mat = la.inv(mat)

        return np.array([a0[0],b0[0],mat[0,0],mat[0,1],mat[1,0],mat[1,1]]).T

    def CameraToImage(self, cameraPoints):
        """
        Transforms camera points to image points

        :param cameraPoints: camera points

        :type cameraPoints: np.array nx2

        :return: corresponding Image points

        :rtype: np.array nx2


        .. warning::

            This function is empty, need implementation

        .. note::

            The inner orientation parameters required for this function are held in ``self.innerOrientationParameters``

        **Usage example**

        .. code-block:: py

            fMarks = np.array([[113.010, 113.011],
                       [-112.984, -113.004],
                       [-112.984, 113.004],
                       [113.024, -112.999]])
            img_fmarks = np.array([[-7208.01, 7379.35],
                        [7290.91, -7289.28],
                        [-7291.19, -7208.22],
                        [7375.09, 7293.59]])
            cam = Camera(153.42, np.array([0.015, -0.020]), None, None, fMarks)
            img = SingleImage(camera = cam, points = None)
            img.ComputeInnerOrientation(img_fmarks)
            pts_image = img.Camera2Image(fMarks)

        """
        #  setting up the required matrices
        a0 = self.innerOrientationParameters[0]
        b0 = self.innerOrientationParameters[1]
        a1 = self.innerOrientationParameters[2]
        a2 = self.innerOrientationParameters[3]
        b1 = self.innerOrientationParameters[4]
        b2 = self.innerOrientationParameters[5]

        R = np.array([[a1[0], a2[0]], [b1[0], b2[0]]])
        T = np.array([[a0[0]], [b0[0]]])

        cameraPoints = cameraPoints.T
        #  computing the transformation to the image system
        return (T + np.dot(R, cameraPoints)).T


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


        **Usage example**

        .. code-block:: py

            fMarks = np.array([[113.010, 113.011],
                       [-112.984, -113.004],
                       [-112.984, 113.004],
                       [113.024, -112.999]])
            img_fmarks = np.array([[-7208.01, 7379.35],
                        [7290.91, -7289.28],
                        [-7291.19, -7208.22],
                        [7375.09, 7293.59]])
            cam = Camera(153.42, np.array([0.015, -0.020]), None, None, fMarks)
            img = SingleImage(camera = cam, points = None)
            img.ComputeInnerOrientation(img_fmarks)
            pts_camera = img.Image2Camera(img_fmarks)

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

        **Usage Example**

        .. code-block:: py

            img = SingleImage(camera = cam)
            grdPnts = np.array([[201058.062, 743515.351, 243.987],
                        [201113.400, 743566.374, 252.489],
                        [201112.276, 743599.838, 247.401],
                        [201166.862, 743608.707, 248.259],
                        [201196.752, 743575.451, 247.377]])
            imgPnts3 = np.array([[-98.574, 10.892],
                         [-99.563, -5.458],
                         [-93.286, -10.081],
                         [-99.904, -20.212],
                         [-109.488, -20.183]])
            img.ComputeExteriorOrientation(imgPnts3, grdPnts, 0.3)


        """
        cameraPoints = self.ImageToCamera(imagePoints)
        self.__ComputeApproximateVals(cameraPoints, groundPoints)
        l0 = self.__ComputeObservationVector(groundPoints.T)
        l0 = np.reshape(l0, (-1,1))
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
        if (np.size(A,0)-np.size(deltaX)) != 0:
            sig = np.dot(v.T, v)/(np.size(A,0)-np.size(deltaX))
            sigmaX = sig[0]*la.inv(N)
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

        for i in range(groundPoints.shape[0]) :
            x = xp - (f) * (((r11 * (groundPoints[i, 0] - X0) + r21 * (groundPoints[i, 1] - Y0) + r31 * (
                    groundPoints[i, 2] - Z0)) / (r13 * (groundPoints[i, 0] - X0) + r23 * (
                    groundPoints[i, 1] - Y0) + r33 * (groundPoints[i, 2] - Z0))))
            y = yp - (f) * (((r12 * (groundPoints[i, 0] - X0) + r22 * (groundPoints[i, 1] - Y0) + r32 * (
                    groundPoints[i, 2] - Z0)) / (r13 * (groundPoints[i, 0] - X0) + r23 * (
                    groundPoints[i, 1] - Y0) + r33 * (groundPoints[i, 2] - Z0))))

            camPoints.append([x, y])

        return self.CameraToImage(np.array(camPoints))

    def ImageToRay(self, imagePoints):
        """
        Transforms Image point to a Ray in world system

        :param imagePoints: coordinates of an image point

        :type imagePoints: np.array nx2

        :return: Ray direction in world system

        :rtype: np.array nx3

        .. warning::

           This function is empty, need implementation

        .. note::

            The exterior orientation parameters needed here are called by ``self.exteriorOrientationParameters``
        """
        pass  # delete after implementations

    def ImageToGround_GivenZ(self, imagePoints, Z_values):
        """
        Compute corresponding ground point given the height in world system

        :param imagePoints: points in image space
        :param Z_values: height of the ground points


        :type Z_values: np.array nx1
        :type imagePoints: np.array nx2
        :type eop: np.ndarray 6x1

        :return: corresponding ground points

        :rtype: np.ndarray

        .. warning::

             This function is empty, need implementation

        .. note::

            - The exterior orientation parameters needed here are called by ``self.exteriorOrientationParameters``
            - The focal length can be called by ``self.camera.focalLength``

        **Usage Example**

        .. code-block:: py


            imgPnt = np.array([-50., -33.])
            img.ImageToGround_GivenZ(imgPnt, 115.)

        """
        cameraPoints = self.ImageToCamera(imagePoints)
        cameraPoints = cameraPoints.T
        pars = self.exteriorOrientationParameters
        X0 = pars[0]
        Y0 = pars[1]
        Z0 = pars[2]

        T = np.array([[X0], [Y0], [Z0]])

        omega = pars[3]
        phi = pars[4]
        kappa = pars[5]
        R = Compute3DRotationMatrix(omega, phi, kappa)

        f = self.camera.focalLength

        # allocating memory for return array
        groundPoints = []

        for i in range(len(cameraPoints[1])):
            camVec = np.insert(cameraPoints[:,i], np.size(cameraPoints), -f)
            lam = (Z_values - Z0)/(np.dot(R[2,:], camVec))

            X = X0 + lam*np.dot(R[0,:], camVec)
            Y = Y0 + lam * np.dot(R[1, :], camVec)

            xy = [X, Y, Z_values]
            groundPoints.append(xy)


        groundPoints = np.array(groundPoints)

        return groundPoints




    # ---------------------- Private methods ----------------------

    def __ComputeApproximateVals(self, cameraPoints, groundPoints):
        """
        Compute exterior orientation approximate values via 2-D conform transformation

        :param cameraPoints: points in image space (x y)
        :param groundPoints: corresponding points in world system (X, Y, Z)

        :type cameraPoints: np.ndarray [nx2]
        :type groundPoints: np.ndarray [nx3]

        :return: Approximate values of exterior orientation parameters
        :rtype: np.ndarray or dict

        .. note::

            - ImagePoints should be transformed to ideal camera using ``self.ImageToCamera(imagePoints)``. See code below
            - The focal length is stored in ``self.camera.focalLength``
            - Don't forget to update ``self.exteriorOrientationParameters`` in the order defined within the property
            - return values can be a tuple of dictionaries and arrays.

        .. warning::

           - This function is empty, need implementation
           - Decide how the exterior parameters are held, don't forget to update documentation
        """

        # Find approximate values
        cameraPoints = cameraPoints.reshape(np.size(cameraPoints), 1)
        groundPointsXY = groundPoints[0:2,:].T
        groundPointsXY = groundPointsXY.reshape(np.size(groundPointsXY), 1)
        groundPointsZ = groundPoints[2,:].T

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
        kappa = np.arctan2(-X[3],X[2])
        lam = np.sqrt(X[2]**2+X[3]**2)
        Z0 = np.average(groundPointsZ) + (lam)*self.camera.focalLength

        adjustment_results = {"X0" : X0[0], "Y0" : Y0[0], "Z0" : Z0[0] ,"omega" : 0, "phi" : 0, "kappa" : np.rad2deg(kappa[0])}

        self.__exteriorOrientationParameters = np.array([X0[0], Y0[0], Z0[0], 0, 0, kappa[0]]).T  # updating the exterior orientation params
        # self.__exteriorOrientationParameters = np.array([202225, 742447, 657.81, 0, 0, kappa[0]]).T
        return adjustment_results

    def __ComputeObservationVector(self, groundPoints):
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
        dX = groundPoints[:,0] - self.exteriorOrientationParameters[0]
        dY = groundPoints[:,1] - self.exteriorOrientationParameters[1]
        dZ = groundPoints[:,2] - self.exteriorOrientationParameters[2]
        dXYZ = np.vstack([dX, dY, dZ])
        rotated_XYZ = np.dot(self.rotationMatrix.T, dXYZ).T

        l0 = np.empty(n * 2)

        # Computation of the observation vector based on approximate exterior orientation parameters:
        l0[::2] = -self.camera.focalLength * rotated_XYZ[:, 0] / rotated_XYZ[:, 2]
        l0[1::2] = -self.camera.focalLength * rotated_XYZ[:, 1] / rotated_XYZ[:, 2]

        return l0

    def __ComputeDesignMatrix(self, groundPoints):
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

        focalBySqauredRT3g = self.camera.focalLength / rT3g ** 2

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

        # all derivatives of x and y
        dd = np.array([np.vstack([dxdX0, dxdY0, dxdZ0, dxdOmega, dxdPhi, dxdKappa]).T,
                       np.vstack([dydX0, dydY0, dydZ0, dydOmega, dydPhi, dydKappa]).T])

        a = np.zeros((2 * dd[0].shape[0], 6))
        a[0::2] = dd[0]
        a[1::2] = dd[1]

        return a


if __name__ == '__main__':
    fMarks = np.array([[113.010, 113.011],
                       [-112.984, -113.004],
                       [-112.984, 113.004],
                       [113.024, -112.999]])
    img_fmarks = np.array([[-7208.01, 7379.35],
                           [7290.91, -7289.28],
                           [-7291.19, -7208.22],
                           [7375.09, 7293.59]])
    cam = Camera(153.42, np.array([0.015, -0.020]), None, None, fMarks)
    img = SingleImage(camera = cam)
    print(img.ComputeInnerOrientation(img_fmarks))

    print(img.ImageToCamera(img_fmarks))

    print(img.CameraToImage(fMarks))

    GrdPnts = np.array([[5100.00, 9800.00, 100.00]])
    print(img.GroundToImage(GrdPnts))

    imgPnt = np.array([23.00, 25.00])
    print(img.ImageToRay(imgPnt))

    imgPnt2 = np.array([-50., -33.])
    print(img.ImageToGround_GivenZ(imgPnt2, 115.))

    # grdPnts = np.array([[201058.062, 743515.351, 243.987],
    #                     [201113.400, 743566.374, 252.489],
    #                     [201112.276, 743599.838, 247.401],
    #                     [201166.862, 743608.707, 248.259],
    #                     [201196.752, 743575.451, 247.377]])
    #
    # imgPnts3 = np.array([[-98.574, 10.892],
    #                      [-99.563, -5.458],
    #                      [-93.286, -10.081],
    #                      [-99.904, -20.212],
    #                      [-109.488, -20.183]])
    #
    # intVal = np.array([200786.686, 743884.889, 954.787, 0, 0, 133 * np.pi / 180])
    #
    # print img.ComputeExteriorOrientation(imgPnts3, grdPnts, intVal)
