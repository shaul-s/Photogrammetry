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
        v1 = v1 / la.norm(v1)
        v2 = np.dot(R2, np.hstack((cameraPoint2, -f)) - o2)
        v2 = v2 / la.norm(v2)
        v3 = np.dot(R3, np.hstack((cameraPoint3, -f)) - o3)
        v3 = v3 / la.norm(v3)

        #  first model
        d1 = np.cross(v1, v2)
        A1 = np.vstack((v1, d1, v2))
        X1 = np.dot(la.inv(A1), b12)

        #  second model
        d2 = np.cross(v2, v3)
        A2 = np.vstack((np.dot(R2, b23), v3, -d2))
        X2 = np.dot(la.inv(A2), X1[2] * v2)

        #  returning the scale between models
        return X2[0]


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

        .. warning::

            This function is empty' need implementation
        """

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



if __name__ == '__main__':
    camera = Camera(152, None, None, None, None)
    image1 = SingleImage(camera)
    image2 = SingleImage(camera)
    image3 = SingleImage(camera)
    imagePair1 = ImagePair(image1, image2)
    imagePair2 = ImagePair(image2, image3)
    imageTriple1 = ImageTriple(imagePair11, imagePair22)
