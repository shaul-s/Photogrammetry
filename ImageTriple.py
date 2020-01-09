import numpy as np
from ImagePair import ImagePair
from SingleImage import SingleImage
from Camera import Camera


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
        if self.imagePair1.isSolved and self.imagePair2.isSolved :
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

    def ComputeScaleBetweenModels(cameraPoint1, cameraPoint2, cameraPoint3):
        """
        Compute scale between two models given the relative orientation

        :param cameraPoints1: camera point in first camera space
        :param cameraPoints2: camera point in second camera space
        :param cameraPoints3:  camera point in third camera space

        :type cameraPoints1: np.array 1x3
        :type cameraPoints2: np.array 1x3
        :type cameraPoints3: np.array 1x3


        .. warning::

            This function is empty, need implementation
        """
        pass

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
        :param modelPoints1: points in the firt model
        :param modelPoints2:points in the second model

        :type modelPoints1: np.array nx3
        :type modelPoints2: np.array nx3

        :return: None

        .. warning::
            This function is empty, need implementation
        """


if __name__ == '__main__':
    camera = Camera(152, None, None, None, None)
    image1 = SingleImage(camera)
    image2 = SingleImage(camera)
    image3 = SingleImage(camera)
    imagePair1 = ImagePair(image1, image2)
    imagePair2 = ImagePair(image2, image3)
    imageTriple1 = ImageTriple(imagePair11, imagePair22)
