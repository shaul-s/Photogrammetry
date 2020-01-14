from Camera import Camera
from SingleImage import SingleImage
from MatrixMethods import Compute3DRotationMatrix, Compute3DRotationDerivativeMatrix, ComputeSkewMatrixFromVector
import numpy as np
from scipy import linalg as la
import PhotoViewer as pv



class ImagePair(object) :

    def __init__(self, image1, image2) :
        """
        Initialize the ImagePair class

        :param image1: First image
        :param image2: Second image
        """
        self.__image1 = image1
        self.__image2 = image2
        self.__relativeOrientationImage1 = np.array([0, 0, 0, 0, 0, 0])  # The relative orientation of the first image
        self.__relativeOrientationImage2 = None  # The relative orientation of the second image
        self.__absoluteOrientation = None
        self.__isSolved = False  # Flag for the relative orientation

    @property
    def isSolved(self) :
        """
        Flag for the relative orientation
        returns True if the relative orientation is solved, otherwise it returns False

        :return: boolean, True or False values
        """
        return self.__isSolved

    @property
    def image1(self) :
        """
        first image in the pair

        :return: image

        :rtype: SingleImage
        """
        return self.__image1

    @property
    def image2(self) :
        """
        first image in the pair

        :return: image

        :rtype: SingleImage
        """
        return self.__image2

    @property
    def RotationMatrix_Image1(self) :
        """
        Rotation matrix of the first image

        :return: rotation matrix

        :rtype: np.array 3x3
        """
        return Compute3DRotationMatrix(self.__relativeOrientationImage1[3], self.__relativeOrientationImage1[4],
                                       self.__relativeOrientationImage1[5])

    @property
    def RotationMatrix_Image2(self) :
        """
        Rotation matrix of the second image

        :return: rotation matrix

        :rtype: np.array 3x3
        """
        return Compute3DRotationMatrix(self.__relativeOrientationImage2[3], self.__relativeOrientationImage2[4],
                                       self.__relativeOrientationImage2[5])

    @property
    def PerspectiveCenter_Image1(self) :
        """
        Perspective center of the first image

        :return: perspective center

        :rtype: np.array (3, )
        """
        return self.__relativeOrientationImage1[0 :3]

    @property
    def PerspectiveCenter_Image2(self) :
        """
        Perspective center of the second image

        :return: perspective center

        :rtype: np.array (3, )
        """
        return self.__relativeOrientationImage2[0 :3]

    def ImagesToGround(self, imagePoints1, imagePoints2, Method) :
        """
        Computes ground coordinates of homological points

        :param imagePoints1: points in image 1
        :param imagePoints2: corresponding points in image 2
        :param Method: method to use for the ray intersection, three options exist: geometric, vector, Collinearity

        :type imagePoints1: np.array nx2
        :type imagePoints2: np.array nx2
        :type Method: string

        :return: ground points, their accuracies.

        :rtype: dict

        **Usage example**

        .. code-block:: py

            camera = Camera(152, None, None, None, None)
            image1 = SingleImage(camera)
            image2 = SingleImage(camera)

            imagePoints1 = np.array([[-4.83,7.80],
                                [-4.64, 134.86],
                                [5.39,-100.80],
                                [4.58,55.13],
                                [98.73,9.59],
                                [62.39,128.00],
                                [67.90,143.92],
                                    [56.54,-85.76]])
            imagePoints2 = np.array([[-83.17,6.53],
                                 [-102.32,146.36],
                                 [-62.84,-102.87],
                                 [-97.33,56.40],
                                 [-3.51,14.86],
                                 [-27.44,136.08],
                                 [-23.70,152.90],
                                 [-8.08,-78.07]])

            new = ImagePair(image1, image2)

            new.ImagesToGround(imagePoints1, imagePoints2, 'geometric'))

        """
        #  defining perspective center in the world system and transforming to camera points
        o1 = np.array(self.__image1.exteriorOrientationParameters[0 :3])
        o2 = np.array(self.__image2.exteriorOrientationParameters[0 :3])
        camPoints1 = self.__image1.ImageToCamera(imagePoints1)
        camPoints2 = self.__image2.ImageToCamera(imagePoints2)

        groundPoints = []
        e = []
        #  what is the method ?
        if Method == 'geometric' :
            for i in range(camPoints1.shape[0]) :
                # following the geometric method for forward intersection:
                x1 = np.hstack((camPoints1[i, :], -self.__image1.camera.focalLength)) / 1000
                x2 = np.hstack((camPoints2[i, :], -self.__image2.camera.focalLength)) / 1000
                v1 = np.dot(self.__image1.rotationMatrix, x1)
                v1 = np.reshape((v1 / la.norm(v1)), (3, 1))
                v2 = np.dot(self.__image2.rotationMatrix, x2)
                v2 = np.reshape((v2 / la.norm(v2)), (3, 1))

                v1vt = np.dot(v1, v1.T)
                v2vt = np.dot(v2, v2.T)
                I = np.eye(v1.shape[0])

                A1 = I - v1vt
                A2 = I - v2vt

                l1 = np.dot(A1, o1)
                l2 = np.dot(A2, o2)

                A = np.vstack((A1, A2))
                l = np.hstack((l1, l2))
                # computing the point in the world system and the residuals
                X = np.dot(la.inv(np.dot(A.T, A)), np.dot(A.T, l))
                e1 = np.dot((I - v1vt), X - o1)
                e2 = np.dot((I - v2vt), X - o2)

                e.append((np.abs(e1) + np.abs(e2)) / 2)
                groundPoints.append(X)

            return np.array(groundPoints), np.array(e)


        elif Method == 'vector' :

            for i in range(camPoints1.shape[0]) :
                x1 = np.hstack((camPoints1[i, :], -self.__image1.camera.focalLength)) / 1000
                x2 = np.hstack((camPoints2[i, :], -self.__image2.camera.focalLength)) / 1000
                v1 = np.dot(self.__image1.rotationMatrix, x1)
                v1 = v1 / la.norm(v1)
                v2 = np.dot(self.__image2.rotationMatrix, x2)
                v2 = v2 / la.norm(v2)

                mat_inv = np.array([[np.dot(v1, v1), np.dot(-v1, v2)], [np.dot(v1, v2), np.dot(-v2, v2)]])
                mat = np.array([[np.dot((o2 - o1), v1)], [np.dot((o2 - o1), v2)]])

                lam = np.dot(la.inv(mat_inv), mat)

                f = o1 + np.dot(float(lam[0]), v1)
                g = o2 + np.dot(float(lam[1]), v2)
                groundPoints.append((f + g) / 2)
                e.append((f - g) / 2)

            return np.array(groundPoints), np.array(e)

    def ComputeDependentRelativeOrientation(self, imagePoints1, imagePoints2, initialValues) :
        """
         Compute relative orientation parameters

        :param imagePoints1: points in the first image [m"m]
        :param imagePoints2: corresponding points in image 2(homology points) nx2 [m"m]
        :param initialValues: approximate values of relative orientation parameters

        :type imagePoints1: np.array nx2
        :type imagePoints2: np.array nx2
        :type initialValues: np.array (6L,)

        :return: relative orientation parameters.

        :rtype: np.array 6x1 / ADD

        .. warning::

            Can be held either as dictionary or array. For your implementation and decision.

        .. note::

            Do not forget to decide how it is held and document your decision


        **Usage example**

        .. code-block:: py

            camera = Camera(152, None, None, None, None)
            image1 = SingleImage(camera)
            image2 = SingleImage(camera)

            imagePoints1 = np.array([[-4.83,7.80],
                                [-4.64, 134.86],
                                [5.39,-100.80],
                                [4.58,55.13],
                                [98.73,9.59],
                                [62.39,128.00],
                                [67.90,143.92],
                                [56.54,-85.76]])
            imagePoints2 = np.array([[-83.17,6.53],
                                 [-102.32,146.36],
                                 [-62.84,-102.87],
                                 [-97.33,56.40],
                                 [-3.51,14.86],
                                 [-27.44,136.08],
                                 [-23.70,152.90],
                                 [-8.08,-78.07]])
            new = ImagePair(image1, image2)

            new.ComputeDependentRelativeOrientation(imagePoints1, imagePoints2, np.array([1, 0, 0, 0, 0, 0])))

        """

        zs = np.full((1, len(imagePoints1)), -self.__image1.camera.focalLength)
        imagePoints1 = np.hstack((imagePoints1, np.transpose(zs)))
        imagePoints2 = np.hstack((imagePoints2, np.transpose(zs)))

        A, B, w = self.Build_A_B_W(imagePoints1, imagePoints2, np.reshape(initialValues, (initialValues.size, 1)))

        M = np.dot(B, B.T)
        N = np.dot(A.T, np.dot(la.inv(M), A))
        u = np.dot(A.T, np.dot(la.inv(M), w))

        dX = -np.dot(la.inv(N), u)
        initialValues = initialValues + dX

        while la.norm(dX) >= 1e-6 :
            A, B, w = self.Build_A_B_W(imagePoints1, imagePoints2, np.reshape(initialValues, (initialValues.size, 1)))

            M = np.dot(B, B.T)
            N = np.dot(A.T, np.dot(la.inv(M), A))
            u = np.dot(A.T, np.dot(la.inv(M), w))

            dX = -np.dot(la.inv(N), u)
            initialValues = initialValues + dX

        initialValues = np.insert(initialValues, 0, 1)
        self.__relativeOrientationImage2 = initialValues

        v = -np.dot(B.T, np.dot(la.inv(M), w))
        sig_squared = np.dot(v.T, v) / (A.shape[0] - 5)

        sigmaX = sig_squared * la.inv(N)

        self.__isSolved = True

        return {"Relative Orientation Parameters" : initialValues, "Variance-Covariance Matrix" : sigmaX}


    def Build_A_B_W(self, cameraPoints1, cameraPoints2, x) :
        """
        Function for computing the A and B matrices and vector w

        :param cameraPoints1: points in the first camera system
        :param cameraPoints2: corresponding homology points in the second camera system
        :param x: initialValues vector by, bz, omega, phi, kappa ( bx=1)

        :type cameraPoints1: np.array nx3
        :type cameraPoints2: np.array nx3
        :type x: np.array (5,1)

        :return: A ,B matrices, w vector

        :rtype: tuple
        """
        numPnts = cameraPoints1.shape[0]  # Number of points

        dbdy = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
        dbdz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])

        dXdx = np.array([1, 0, 0])
        dXdy = np.array([0, 1, 0])

        # Compute rotation matrix and it's derivatives
        rotationMatrix2 = Compute3DRotationMatrix(x[2, 0], x[3, 0], x[4, 0])
        dRdOmega = Compute3DRotationDerivativeMatrix(x[2, 0], x[3, 0], x[4, 0], 'omega')
        dRdPhi = Compute3DRotationDerivativeMatrix(x[2, 0], x[3, 0], x[4, 0], 'phi')
        dRdKappa = Compute3DRotationDerivativeMatrix(x[2, 0], x[3, 0], x[4, 0], 'kappa')

        # Create the skew matrix from the vector [bx, by, bz]
        bMatrix = ComputeSkewMatrixFromVector(np.array([1, x[0, 0], x[1, 0]]))

        # Compute A matrix; the coplanar derivatives with respect to the unknowns by, bz, omega, phi, kappa
        A = np.zeros((numPnts, 5))
        A[:, 0] = np.diag(
            np.dot(cameraPoints1,
                   np.dot(dbdy, np.dot(rotationMatrix2, cameraPoints2.T))))  # derivative in respect to by
        A[:, 1] = np.diag(
            np.dot(cameraPoints1,
                   np.dot(dbdz, np.dot(rotationMatrix2, cameraPoints2.T))))  # derivative in respect to bz
        A[:, 2] = np.diag(
            np.dot(cameraPoints1, np.dot(bMatrix, np.dot(dRdOmega, cameraPoints2.T))))  # derivative in respect to omega
        A[:, 3] = np.diag(
            np.dot(cameraPoints1, np.dot(bMatrix, np.dot(dRdPhi, cameraPoints2.T))))  # derivative in respect to phi
        A[:, 4] = np.diag(
            np.dot(cameraPoints1, np.dot(bMatrix, np.dot(dRdKappa, cameraPoints2.T))))  # derivative in respect to kappa

        # Compute B matrix; the coplanar derivatives in respect to the observations, x', y', x'', y''.
        B = np.zeros((numPnts, 4 * numPnts))
        k = 0
        for i in range(numPnts) :
            p1vec = cameraPoints1[i, :]
            p2vec = cameraPoints2[i, :]
            B[i, k] = np.dot(dXdx, np.dot(bMatrix, np.dot(rotationMatrix2, p2vec)))
            B[i, k + 1] = np.dot(dXdy, np.dot(bMatrix, np.dot(rotationMatrix2, p2vec)))
            B[i, k + 2] = np.dot(np.dot(p1vec, np.dot(bMatrix, rotationMatrix2)), dXdx)
            B[i, k + 3] = np.dot(np.dot(p1vec, np.dot(bMatrix, rotationMatrix2)), dXdy)
            k += 4

        # w vector
        w = np.diag(np.dot(cameraPoints1, np.dot(bMatrix, np.dot(rotationMatrix2, cameraPoints2.T))))

        return A, B, w

    def ImagesToModel(self, imagePoints1, imagePoints2, Method) :
        """
        Mapping points from image space to model space

        :param imagePoints1: points from the first image in pix
        :param imagePoints2: points from the second image in pix
        :param Method: method for intersection

        :type imagePoints1: np.array nx2
        :type imagePoints2: np.array nx2
        :type Method: string

        :return: corresponding model points
        :rtype: np.array nx3

        .. note::

            One of the images is a reference, orientation of this image must be set.

        """
        cameraPoints1 = self.__image1.ImageToCamera(imagePoints1)
        cameraPoints2 = self.__image2.ImageToCamera(imagePoints2)

        if Method == 'vector' :
            return self.vectorIntersction(cameraPoints1, cameraPoints2)
        elif Method == 'geometric' :
            return self.geometricIntersection(cameraPoints1, cameraPoints2)

    def GroundToImage(self, groundPoints) :
        """
        Transforming ground points to image points

        :param groundPoints: ground points [m]

        :type groundPoints: np.array nx3

        :return: corresponding Image points

        :rtype: np.array nx2

        """
        pass  # you need to know what image to use (?)

        for i in range(groundPoints.shape[0]) :
            x = xp - (f) * (((r11 * (groundPoints[i, 0] - X0) + r21 * (groundPoints[i, 1] - Y0) + r31 * (
                    groundPoints[i, 2] - Z0)) / (r13 * (groundPoints[i, 0] - X0) + r23 * (
                    groundPoints[i, 1] - Y0 + r33 * (groundPoints[i, 2] - Z0)))))
            y = yp - (f) * (((r12 * (groundPoints[i, 0] - X0) + r22 * (groundPoints[i, 1] - Y0) + r32 * (
                    groundPoints[i, 2] - Z0)) / (r13 * (groundPoints[i, 0] - X0) + r23 * (
                    groundPoints[i, 1] - Y0 + r33 * (groundPoints[i, 2] - Z0)))))

            camPoints.append([x, y])

        return np.array(camPoints)

    def RotationLevelModel(self, constrain1, constrain2) :

        """
        Compute rotation matrix from the current model coordinate system to the other coordinate system

        :param constrain1: constrain of the first axis
        :param constrain2: constrain of the second axis

        :type constrain1: tuple
        :type constrain2: tuple

        :return: rotation matrix

        :rtype: np.array 3x3

        .. note::

            The vector data included in the two constrains must be normalized

            The two constrains should be given to two different axises, if not return identity matrix

        """
        ax1, v1 = constrain1[0], constrain1[1]
        ax2, v2 = constrain2[0], constrain2[1]

        if la.norm(v1) != 1 or la.norm(v2) != 1:
            raise ValueError('Your vectors are not normalized !')

        if ax1 == ax2:
            return np.eye(3)

        if ax1 == 'z':
            final_axis = np.cross(v1, v2)
            x = np.reshape(v2, (3, 1))
            y = np.reshape(final_axis, (3, 1))
            z = np.reshape(v1, (3, 1))
        else:
            final_axis = np.cross(v2, v1)
            x = np.reshape(v1, (3, 1))
            y = np.reshape(final_axis, (3, 1))
            z = np.reshape(v2, (3, 1))

        return np.hstack((x, y, z))


    def ModelTransformation(self, modelPoints, rotation, scale):
        """
        Transform model from the current coordinate system to other coordinate system

        :param modelPoints: coordinates in current model space
        :param scale: scale between the two coordinate systems

        :type modelPoints: np.array nx3
        :type scale: float

        :return: corresponding coordinates in the other coordinate system

        :rtype: np.array nx3
        """
        worldPoints = []
        for i in range(modelPoints.shape[0]):
            worldPoints.append(scale * np.dot(rotation, modelPoints[i, :]))

        return np.array(worldPoints)

    def geometricIntersection(self, cameraPoints1, cameraPoints2) :
        """
        Ray Intersection based on geometric calculations.

        :param cameraPoints1: points in the first image
        :param cameraPoints2: corresponding points in the second image

        :type cameraPoints1: np.array nx3
        :type cameraPoints2: np.array nx3

        :return: modelPoints, e (accuracy's)

        :rtype: np.array nx3, nx1

        """
        o1 = self.PerspectiveCenter_Image1
        o2 = self.PerspectiveCenter_Image2
        modelPoints = []
        e = []
        for i in range(cameraPoints1.shape[0]) :
            # following the geometric method for forward intersection:
            x1 = np.hstack((cameraPoints1[i, :], -self.__image1.camera.focalLength)) / 1000
            x2 = np.hstack((cameraPoints2[i, :], -self.__image2.camera.focalLength)) / 1000
            v1 = np.dot(self.RotationMatrix_Image1, x1)
            v1 = np.reshape((v1 / la.norm(v1)), (3, 1))
            v2 = np.dot(self.RotationMatrix_Image2, x2)
            v2 = np.reshape((v2 / la.norm(v2)), (3, 1))

            v1vt = np.dot(v1, v1.T)
            v2vt = np.dot(v2, v2.T)
            I = np.eye(v1.shape[0])

            A1 = I - v1vt
            A2 = I - v2vt

            l1 = np.dot(A1, o1)
            l2 = np.dot(A2, o2)

            A = np.vstack((A1, A2))
            l = np.hstack((l1, l2))
            # computing the point in the world system and the residuals
            X = np.dot(la.inv(np.dot(A.T, A)), np.dot(A.T, l))
            e1 = np.dot((I - v1vt), X - o1)
            e2 = np.dot((I - v2vt), X - o2)

            e.append((np.abs(e1) + np.abs(e2)) / 2)
            modelPoints.append(X)

        return np.array(modelPoints), np.array(e)

    def vectorIntersction(self, cameraPoints1, cameraPoints2) :
        """
        Ray Intersection based on vector calculations.

        :param cameraPoints1: points in image space
        :param cameraPoints2: corresponding image points

        :type cameraPoints1: np.array nx
        :type cameraPoints2: np.array nx


        :return: groundPoints, e (the accuracy)

        :rtype: np.array nx3, nx1

        """
        o1 = self.PerspectiveCenter_Image1
        o2 = self.PerspectiveCenter_Image2
        modelPoints = []
        e = []
        for i in range(cameraPoints1.shape[0]) :
            x1 = np.hstack((cameraPoints1[i, :], -self.__image1.camera.focalLength)) / 1000
            x2 = np.hstack((cameraPoints2[i, :], -self.__image2.camera.focalLength)) / 1000
            v1 = np.dot(self.RotationMatrix_Image1, x1)
            v1 = v1 / la.norm(v1)
            v2 = np.dot(self.RotationMatrix_Image2, x2)
            v2 = v2 / la.norm(v2)

            mat_inv = np.array([[np.dot(v1, v1), np.dot(-v1, v2)], [np.dot(v1, v2), np.dot(-v2, v2)]])
            mat = np.array([[np.dot((o2 - o1), v1)], [np.dot((o2 - o1), v2)]])

            lam = np.dot(la.inv(mat_inv), mat)

            f = o1 + np.dot(float(lam[0]), v1)
            g = o2 + np.dot(float(lam[1]), v2)
            modelPoints.append((f + g) / 2)
            e.append((f - g) / 2)

        return np.array(modelPoints), np.array(e)

    def CollinearityIntersection(self, cameraPoints1, cameraPoints2) :
        """
        Ray intersection based on the collinearity principle

        :param cameraPoints1: points in image space
        :param cameraPoints2: corresponding image points

        :type cameraPoints1: np.array nx2
        :type cameraPoints2: np.array nx2

        :return: corresponding ground points

        :rtype: np.array nx3

        .. warning::

            This function is empty, needs implementation

        """
        pass  # didn't implement

    def drawImagePair(self, modelPoints, ax) :
        """
        Drawing the model images plane, perspective centers and ray intersection
        :param modelPoints: the points that were computed in the model system

        :type modelPoints: np.array nx3

        :return: None (plotting)
        """

        pv.drawOrientation(self.RotationMatrix_Image1,
                           np.reshape(self.PerspectiveCenter_Image1, (self.PerspectiveCenter_Image1.size, 1)), 100, ax)
        pv.drawOrientation(self.RotationMatrix_Image2,
                           np.reshape(self.PerspectiveCenter_Image2, (self.PerspectiveCenter_Image2.size, 1)), 100, ax)

        pv.drawImageFrame(5472 * 2.4e-3, 3648 * 2.4e-3, self.RotationMatrix_Image1,
                          np.reshape(self.PerspectiveCenter_Image1, (self.PerspectiveCenter_Image1.size, 1)),
                          4248.06 * 2.4e-3, 100, ax)
        pv.drawImageFrame(5472 * 2.4e-3, 3648 * 2.4e-3, self.RotationMatrix_Image2,
                          np.reshape(self.PerspectiveCenter_Image2, (self.PerspectiveCenter_Image2.size, 1)),
                          4248.06 * 2.4e-3, 100, ax)

        pv.drawRays(modelPoints * 1000,
                    np.reshape(self.PerspectiveCenter_Image1, (self.PerspectiveCenter_Image1.size, 1)), ax, 'gray')
        pv.drawRays(modelPoints * 1000,
                    np.reshape(self.PerspectiveCenter_Image2, (self.PerspectiveCenter_Image2.size, 1)), ax, 'gray')




if __name__ == '__main__' :
    camera = Camera(152, None, None, None, None)
    image1 = SingleImage(camera)
    image2 = SingleImage(camera)
    leftCamPnts = np.array([[-4.83, 7.80],
                            [-4.64, 134.86],
                            [5.39, -100.80],
                            [4.58, 55.13],
                            [98.73, 9.59],
                            [62.39, 128.00],
                            [67.90, 143.92],
                            [56.54, -85.76]])
    rightCamPnts = np.array([[-83.17, 6.53],
                             [-102.32, 146.36],
                             [-62.84, -102.87],
                             [-97.33, 56.40],
                             [-3.51, 14.86],
                             [-27.44, 136.08],
                             [-23.70, 152.90],
                             [-8.08, -78.07]])
    new = ImagePair(image1, image2)

    print(new.ComputeDependentRelativeOrientation(leftCamPnts, rightCamPnts, np.array([1, 0, 0, 0, 0, 0])))
