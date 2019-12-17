import numpy as np
import Reader as rd
import Camera as cam
import SingleImage as sg
import MatrixMethods as mm
from scipy import linalg as la


def ImagesToGround(image1, image2, imagePoints1, imagePoints2, Method):
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
    camPoints1 = image1.ImageToCamera(imagePoints1)
    camPoints2 = image2.ImageToCamera(imagePoints2)

    if Method == 'geometric':
        v1 = np.dot(self.__image1.rotationMatrix, cameraPoints1)
        v2 = np.dot(self.__image2.rotationMatrix, cameraPoints2)

        v1vt = np.dot(v1, v1.T)
        v2vt = np.dot(v2, v2.T)
        I = np.eye(v1vt.shape[1])

        A1 = I - v1vt
        A2 = I - v2vt

        o1 = np.hstack(self.__image1.camera.principalPoint, -self.__image1.camera.focalLength)
        o2 = np.hstack(self.__image2.camera.principalPoint, -self.__image2.camera.focalLength)
        l1 = np.dot(A1, o1)
        l2 = np.dot(A2, o2)

        A = np.hstack(A1, A2)
        l = np.hstack(l1, l2)

        X = np.dot(la.inv(np.dot(A.T, A)), np.dot(A.T, l))

        return X

    elif Method == 'vector':

        groundPoints = []

        o1 = np.array(image1.exteriorOrientationParameters[0:3]).T
        o2 = np.array(image2.exteriorOrientationParameters[0:3]).T

        for i in range(camPoints1.shape[0]):
            x1 = np.hstack((camPoints1[i, :], image1.camera.focalLength))/1000
            x2 = np.hstack((camPoints2[i, :], image2.camera.focalLength))/1000
            v1 = np.dot(image1.rotationMatrix, x1)
            v2 = np.dot(image1.rotationMatrix, x2)

            mat_inv = np.array([[np.dot(v1, v1), np.dot(-v1, v2)], [np.dot(v1, v2), np.dot(-v2, v2)]])
            mat = np.array([[np.dot((o2 - o1), v1)], [np.dot((o2 - o1), v2)]])

            lam = np.dot(la.inv(mat_inv), mat)

            groundPoints.append(((o1 + np.dot(float(lam[0]), v1)) + (o2 + np.dot(float(lam[1]), v2)))/2)

        return np.transpose(np.array(groundPoints))



if __name__ == "__main__":
    ### Reading data ###
    samples = rd.Reader.photoModXMLReader(r'Lab6-Photomod.xml')
    cam_pars = rd.Reader.ReadCamFile(r'rc30.cam')
    fiducialsImg3574 = rd.Reader.Readtxtfile(r'fiducialsImg3574.txt')
    fiducialsImg3575 = rd.Reader.Readtxtfile(r'fiducialsImg3575.txt')
    cam = cam.Camera(cam_pars["f"], np.array([[cam_pars["xp"]], [cam_pars["yp"]]]), None, None, cam_pars["fiducials"])
    image3574 = sg.SingleImage(cam)
    image3575 = sg.SingleImage(cam)
    control_p = rd.Reader.photoModXMLReader(r'lab5.xml')
    GCP = np.zeros((3, int(len(control_p[1]))))
    for i, row in enumerate(control_p[1]):
        GCP[0, i] = row[2]
        GCP[1, i] = row[3]
        GCP[2, i] = row[4]
    ICP3574 = []
    ICP3575 = []
    for row in control_p[2] :
        if row[0] == '3574' :
            ICP3574.append([row[1], row[2]])
        else:
            ICP3575.append([row[1], row[2]])

    ICP3574 = np.array(ICP3574).astype(np.float64)
    ICP3575 = np.array(ICP3575).astype(np.float64)

    ### PART A - PHOTOMOD ###
    image3574.ComputeInnerOrientation(fiducialsImg3574)
    image3575.ComputeInnerOrientation(fiducialsImg3575)
    image3574.ComputeExteriorOrientation(ICP3574, GCP, 1e-6)
    image3575.ComputeExteriorOrientation(ICP3575, GCP, 1e-6)

    fids3574 = []
    fids3575 = []

    for row in samples[2]:
        if row[0] == '3574':
            fids3574.append([row[1],row[2]])
        else:
            fids3575.append([row[1],row[2]])

    fids3574 = np.array(fids3574).astype(np.float64)
    fids3575 = np.array(fids3575).astype(np.float64)

    camPoints3574 = image3574.ImageToCamera(fids3574)
    camPoints3575 = image3575.ImageToCamera(fids3575)



    ### PART B - PYTHON ###

    gPoints = ImagesToGround(image3574, image3575, fids3574, fids3575, 'vector')

    print(samples)