import numpy as np
from ImagePair import ImagePair
from SingleImage import SingleImage
from Camera import Camera
from matplotlib import pyplot as plt
import PhotoViewer as pv
from scipy import linalg as la


class ImageTriple(object):
    def __init__(self, imagePair1, imagePair2):
        """
        Inisialize the ImageTriple class

        :param imagePair1: first image pair
        :param imagePair2: second image pair

        .. warning::
            Check if the relative orientation is solved for each image pair
        """
        self.__imagePair1 = imagePair1
        self.__imagePair2 = imagePair2
        self.__o1 = np.array([0, 0, 0])
        self.__o2 = imagePair1.PerspectiveCenter_Image2
        self.__o3 = np.array([1.85391524, -0.54041259, 0.8277201])
        self.__R1 = np.eye(3)
        self.__R2 = imagePair1.RotationMatrix_Image2
        self.__R3 = np.dot(self.__R2, imagePair2.RotationMatrix_Image2)
        if self.__imagePair1.isSolved and self.__imagePair2.isSolved :
            self.__isSolved = True  # Flag for the relative orientation for both models
        else :
            self.__isSolved = False

        @property
        def isSolved(self):
            """
            Flag for the relative orientation of both models
            returns True if the relative orientation is solved, otherwise it returns False

            :return: boolean, True or False values
            """
            return self.__isSolved

    def ComputeScaleBetweenModels(self, cameraPoint1, cameraPoint2, cameraPoint3) :
        """
        Compute scale between two models given the relative orientation

        :param cameraPoints1: camera point in first camera space
        :param cameraPoints2: camera point in second camera space
        :param cameraPoints3:  camera point in third camera space

        :type cameraPoints1: np.array 1x3
        :type cameraPoints2: np.array 1x3
        """
        #  gathering data for vector intersection
        f = 4248.06 * 2.4e-3  # focal length is needed
        o1 = self.__imagePair1.PerspectiveCenter_Image1
        o2 = b12 = self.__imagePair1.PerspectiveCenter_Image2
        b23 = self.__imagePair2.PerspectiveCenter_Image2
        R2 = self.__imagePair1.RotationMatrix_Image2
        R3 = np.dot(R2, self.__imagePair2.RotationMatrix_Image2)
        o3 = o2 + np.dot(R2, b23)
        v1 = np.hstack((cameraPoint1, -f))
        v1 = np.reshape((v1 / la.norm(v1)), (v1.size, 1))
        v2 = np.dot(R2, np.hstack((cameraPoint2, -f)))
        v2 = np.reshape((v2 / la.norm(v2)), (v2.size, 1))
        v3 = np.dot(R3, np.hstack((cameraPoint3, -f)))
        v3 = np.reshape((v3 / la.norm(v3)), (v3.size, 1))


        #  first model
        d1 = np.reshape((np.cross(v1.T, v2.T)), (3, 1))
        A1 = np.hstack((v1, d1, -v2))
        X1 = np.dot(la.inv(A1), b12)

        #  second model
        d2 = np.reshape((np.cross(v2.T, v3.T)), (3, 1))
        A2 = np.hstack((np.reshape((np.dot(R2, b23)), (3, 1)), v3, -d2))
        X2 = np.dot(la.inv(A2), X1[2] * v2)

        #  returning the scale between models
        return X2[0], X2[1]


    def RayIntersection(self, cameraPoints1, cameraPoints2, cameraPoints3):
        """
        Compute coordinates of the corresponding model point

        :param cameraPoints1: points in camera1 coordinate system
        :param cameraPoints2: points in camera2 coordinate system
        :param cameraPoints3: points in camera3 coordinate system

        :type cameraPoints1 np.array nx3
        :type cameraPoints2: np.array nx3
        :type cameraPoints3: np.array nx3

        :return: point in model coordinate system
        :rtype: np.array nx3
        """
        points = []
        #  computing 3 ray geometric intersection
        for i in range(cameraPoints1.shape[0]):
            v1 = np.dot(self.__R1, cameraPoints1[i, :])
            v1 = np.reshape((v1 / la.norm(v1)), (3, 1))
            v2 = np.dot(self.__R2, cameraPoints2[i, :])
            v2 = np.reshape((v2 / la.norm(v2)), (3, 1))
            v3 = np.dot(self.__R3, cameraPoints3[i, :])
            v3 = np.reshape((v3 / la.norm(v3)), (3, 1))

            I = np.eye(v1.shape[0])
            v1vt = np.dot(v1, v1.T)
            v2vt = np.dot(v2, v2.T)
            v3vt = np.dot(v3, v3.T)

            A1 = I - v1vt
            A2 = I - v2vt
            A3 = I - v3vt

            l1 = np.dot(A1, self.__o1)
            l2 = np.dot(A2, self.__o2)
            l3 = np.dot(A3, self.__o3)

            A = np.vstack((A1, A2, A3))
            l = np.hstack((l1, l2, l3))

            X = np.dot(la.inv(np.dot(A.T, A)), np.dot(A.T, l))

            points.append(X)

        return np.array(points)

    def drawModles(self, imagePair1, imagePair2, modelPoints1, modelPoints2):
        """
        Draw two models in the same figure

        :param imagePair1: first image pair
        :param imagePair2:second image pair
        :param modelPoints1: points in the first model
        :param modelPoints2:points in the second model

        :type modelPoints1: np.array nx3
        :type modelPoints2: np.array nx3

        :return: None
        """
        fig_orthographic = plt.figure()
        ax = fig_orthographic.add_subplot(111, projection='3d')

        imagePair1.drawImagePair(modelPoints1, ax)
        imagePair2.drawImagePair(modelPoints2, ax)

        x1 = modelPoints1[:, 0] * 1000
        y1 = modelPoints1[:, 1] * 1000
        z1 = modelPoints1[:, 2] * 1000
        x2 = modelPoints2[:, 0] * 1000
        y2 = modelPoints2[:, 1] * 1000
        z2 = modelPoints2[:, 2] * 1000

        ax.scatter(x1, y1, z1, marker='o', c='r', s=50)
        ax.scatter(x2, y2, z2, marker='o', c='r', s=50)
        ax.plot(x1, y1, z1, 'b-')
        ax.plot(x2, y2, z2, 'b-')

    def drawImageTriple(self, modelPoints, ax):
        """
        Drawing the model images plane, perspective centers and ray intersection
        :param modelPoints: the points that were computed in the model system
        :param ax: the ax

        :type modelPoints: np.array nx3

        :return: None (plotting)
        """

        pv.drawOrientation(self.__R1,
                           np.reshape(self.__o1 * 1000, (self.__o1.size, 1)), 500, ax)
        pv.drawOrientation(self.__R2,
                           np.reshape(self.__o2 * 1000, (self.__o2.size, 1)), 500, ax)
        pv.drawOrientation(self.__R3,
                           np.reshape(self.__o3 * 1000, (self.__o3.size, 1)), 500, ax)

        pv.drawImageFrame(5472 * 2.4e-3, 3648 * 2.4e-3, self.__R1,
                          np.reshape(self.__o1 * 1000, (self.__o1.size, 1)),
                          4248.06 * 2.4e-3, 100, ax)
        pv.drawImageFrame(5472 * 2.4e-3, 3648 * 2.4e-3, self.__R2,
                          np.reshape(self.__o2 * 1000, (self.__o2.size, 1)),
                          4248.06 * 2.4e-3, 100, ax)
        pv.drawImageFrame(5472 * 2.4e-3, 3648 * 2.4e-3, self.__R3,
                          np.reshape(self.__o3 * 1000, (self.__o3.size, 1)),
                          4248.06 * 2.4e-3, 100, ax)

        pv.drawRays(modelPoints * 1000,
                    np.reshape(self.__o1 * 1000, (self.__o1.size, 1)), ax, 'gray')
        pv.drawRays(modelPoints * 1000,
                    np.reshape(self.__o2 * 1000, (self.__o2.size, 1)), ax, 'gray')
        pv.drawRays(modelPoints * 1000,
                    np.reshape(self.__o3 * 1000, (self.__o3.size, 1)), ax, 'gray')



if __name__ == '__main__':
    camera = Camera(152, None, None, None, None)
    image1 = SingleImage(camera)
    image2 = SingleImage(camera)
    image3 = SingleImage(camera)
    imagePair1 = ImagePair(image1, image2)
    imagePair2 = ImagePair(image2, image3)
    imageTriple1 = ImageTriple(imagePair11, imagePair22)
