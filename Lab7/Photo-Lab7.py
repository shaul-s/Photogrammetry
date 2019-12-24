import numpy as np
import Reader as rd
import Camera as cam
import SingleImage as sg
import ImagePair as ip
import MatrixMethods as mm
from scipy import linalg as la

if __name__ == "__main__" :
    ### loading camera parameters that were given in txt ###
    focal = 4248.06
    xp = 2628.916
    yp = 1834.855
    k1 = 0.07039488
    k2 = -0.17154803
    p1 = -0.00467987
    p2 = -0.00788186

    pixel_size = 2.4e-3  # [mm]

    # creating two camera objects
    cam1 = cam.Camera(focal * pixel_size, np.array([xp * pixel_size, yp * pixel_size]), None, None, None)
    image2008 = sg.SingleImage(cam1)
    image2009 = sg.SingleImage(cam1)
    image2010 = sg.SingleImage(cam1)

    img2008points1 = rd.Reader.ReadSampleFile(r'IMG_2008_1.json')
    img2009points1 = rd.Reader.ReadSampleFile(r'IMG_2009_1.json')
    img2009points2 = rd.Reader.ReadSampleFile(r'IMG_2009_2.json')
    img2010points2 = rd.Reader.ReadSampleFile(r'IMG_2010_2.json')

    img2008points1_tmp = img2008points1.copy()
    img2009points1_tmp = img2009points1.copy()
    img2009points2_tmp = img2009points2.copy()
    img2010points2_tmp = img2010points2.copy()

    #  adjusting to camera system in pixels
    T = np.array([xp, yp])
    for i in range(len(img2008points1_tmp)) :
        img2008points1_tmp[i, 1] = 3648 - img2008points1_tmp[i, 1]
        img2009points1_tmp[i, 1] = 3648 - img2009points1_tmp[i, 1]
    for i in range(len(img2009points2_tmp)) :
        img2009points2_tmp[i, 1] = 3648 - img2009points2_tmp[i, 1]
        img2010points2_tmp[i, 1] = 3648 - img2010points2_tmp[i, 1]

    #  turn pixels to mm's
    img2008points1_tmp = img2008points1_tmp - T
    img2009points1_tmp = img2009points1_tmp - T

    img2009points2_tmp = img2009points2_tmp - T
    img2010points2_tmp = img2010points2_tmp - T

    img2008points1mm = img2008points1_tmp * pixel_size
    img2009points1mm = img2009points1_tmp * pixel_size
    img2009points2mm = img2009points2_tmp * pixel_size
    img2010points2mm = img2010points2_tmp * pixel_size

    #
    imgPair_model1 = ip.ImagePair(image2008, image2009)
    imgPair_model2 = ip.ImagePair(image2009, image2010)
    relativeOrientation_model1 = imgPair_model1.ComputeDependentRelativeOrientation(img2008points1mm, img2009points1mm,
                                                                                    np.array([0, 0, 0, 0, 0]))
    relativeOrientation_model2 = imgPair_model2.ComputeDependentRelativeOrientation(img2009points2mm, img2010points2mm,
                                                                                    np.array([0, 0, 0, 0, 0]))


    print('hi')
