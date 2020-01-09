import numpy as np
import Reader as rd
import Camera as cam
import SingleImage as sg
import ImagePair as ip
from matplotlib import pyplot as plt
import MatrixMethods as mm
from scipy import linalg as la

if __name__ == "__main__" :
    #  computing model link without scale
    #  taking values from lab 7 --
    R2 = np.array([[0.93736404, 0.17563438, -0.30083427], [-0.1699276, 0.984417, 0.04525236],
                   [0.30409425, 0.00870211, 0.9526022]])
    R23 = np.array([[0.95362705, 0.14610846, -0.26314977], [-0.14700069, 0.9890003, 0.01640695],
                    [0.2626524, 0.02303708, 0.9646155]])
    o2 = np.array([1, -0.14151914, 0.22903829])
    b23 = np.array([1, -0.226124, 0.28123083])

    R3 = np.dot(R2, R23)
    o3 = o2 + np.dot(R2, b23)

    #  CODE COPIED FROM LAB 7 TO GET THE TWO MODELS
    ### loading camera parameters that were given in txt ###
    focal = 4248.06
    xp = 2628.916
    yp = 1834.855
    k1 = 0.07039488
    k2 = -0.17154803
    p1 = -0.00467987
    p2 = -0.00788186

    pixel_size = 2.4e-3  # [mm]

    #  reading homologue points
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

    # creating three camera objects for the purpose of inner orientation for every image
    cam1 = cam.Camera(focal * pixel_size, np.array([xp * pixel_size, yp * pixel_size]), None, None, img2008points1mm)
    cam2 = cam.Camera(focal * pixel_size, np.array([xp * pixel_size, yp * pixel_size]), None, None, img2009points1mm)
    cam3 = cam.Camera(focal * pixel_size, np.array([xp * pixel_size, yp * pixel_size]), None, None, img2010points2mm)
    image2008 = sg.SingleImage(cam1)
    image2009 = sg.SingleImage(cam2)
    image2010 = sg.SingleImage(cam3)

    image2008.ComputeInnerOrientation(img2008points1)
    image2009.ComputeInnerOrientation(img2009points1)
    image2010.ComputeInnerOrientation(img2010points2)

    #  creating ImagePair objects for each model and computing relative orientaiton !
    imgPair_model1 = ip.ImagePair(image2008, image2009)
    imgPair_model2 = ip.ImagePair(image2009, image2010)

    relativeOrientation_model1 = imgPair_model1.ComputeDependentRelativeOrientation(img2008points1mm, img2009points1mm,
                                                                                    np.array([0, 0, 0, 0, 0]))
    relativeOrientation_model2 = imgPair_model2.ComputeDependentRelativeOrientation(img2009points2mm, img2010points2mm,
                                                                                    np.array([0, 0, 0, 0, 0]))
    ### END OF COPIED CODE FROM LAB 7

    #  reading sampled points in image system and transforming them to model system
    img08model1 = rd.Reader.ReadSampleFile(r'IMG_2008_LAB8_MODEL1.json')
    img09model12 = rd.Reader.ReadSampleFile(r'IMG_2009_LAB8_MODEL1-2.json')
    img10model2 = rd.Reader.ReadSampleFile(r'IMG_2010_LAB8_MODEL2.json')

    #  computing points in both model systems
    model1uvw = imgPair_model1.ImagesToModel(img08model1, img09model12, 'vector')
    model2uvw = imgPair_model2.ImagesToModel(img09model12, img10model2, 'vector')


    print("hi")
