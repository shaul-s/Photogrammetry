import numpy as np
import Reader as rd
import Camera as cam
import SingleImage as sg
import ImagePair as ip
from matplotlib import pyplot as plt
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
    image2010.ComputeInnerOrientation(img2010points2mm)

    #  creating ImagePair objects for each model and computing relative orientaiton !
    imgPair_model1 = ip.ImagePair(image2008, image2009)
    imgPair_model2 = ip.ImagePair(image2009, image2010)

    relativeOrientation_model1 = imgPair_model1.ComputeDependentRelativeOrientation(img2008points1mm, img2009points1mm,
                                                                                    np.array([0, 0, 0, 0, 0]))
    relativeOrientation_model2 = imgPair_model2.ComputeDependentRelativeOrientation(img2009points2mm, img2010points2mm,
                                                                                    np.array([0, 0, 0, 0, 0]))

    #  reading sampled points in image system and transforming them to model system
    img2008cp = img2008points1.copy()
    img2009cp = img2009points1.copy()
    img2008framepoints = rd.Reader.ReadSampleFile(r'IMG_2008-frames.json')
    img2009framepoints = rd.Reader.ReadSampleFile(r'IMG_2009-frames.json')
    img2008box1 = rd.Reader.ReadSampleFile(r'IMG_2008-box1.json')
    img2009box1 = rd.Reader.ReadSampleFile(r'IMG_2009-box1.json')
    img2008box2 = rd.Reader.ReadSampleFile(r'IMG_2008-box2.json')
    img2009box2 = rd.Reader.ReadSampleFile(r'IMG_2009-box2.json')
    img2008box3 = rd.Reader.ReadSampleFile(r'IMG_2008-box3.json')
    img2009box3 = rd.Reader.ReadSampleFile(r'IMG_2009-box3.json')
    img2008triangle = rd.Reader.ReadSampleFile(r'IMG_2008-triangle.json')
    img2009triangle = rd.Reader.ReadSampleFile(r'IMG_2009triangle.json')

    model1_cpoints = imgPair_model1.ImagesToModel(img2008cp, img2009cp, 'vector')
    model1_framepoints = imgPair_model1.ImagesToModel(img2008framepoints, img2009framepoints, 'vector')
    model1_box1 = imgPair_model1.ImagesToModel(img2008box1, img2009box1, 'vector')
    model1_box2 = imgPair_model1.ImagesToModel(img2008box2, img2009box2, 'vector')
    model1_box3 = imgPair_model1.ImagesToModel(img2008box3, img2009box3, 'vector')
    model1_triangle = imgPair_model1.ImagesToModel(img2008triangle, img2009triangle, 'vector')

    fig_orthographic = plt.figure()
    ax = fig_orthographic.add_subplot(111, projection='3d')

    triangle = model1_triangle[0]
    box1 = model1_box1[0]
    box2 = model1_box2[0]
    box3 = model1_box3[0]
    cPoints = model1_cpoints[0]
    framePoints = model1_framepoints[0]
    imgPair_model1.drawImagePair(framePoints, ax)
    # ax.view_init(-1, 0)
    # plt.show()

    #  drawing wire frame model
    ax = fig_orthographic.add_subplot(111, projection='3d')
    x1 = framePoints[:, 0] * 10
    x2 = cPoints[0 :-5, 0] * 10
    x3 = box1[:, 0] * 10
    x4 = triangle[:, 0] * 10
    y1 = framePoints[:, 1] * 10
    y2 = cPoints[0 :-5, 1] * 10
    y3 = box1[:, 1] * 10
    y4 = triangle[:, 1] * 10
    z1 = framePoints[:, 2] * 10
    z2 = cPoints[0 :-5, 2] * 10
    z3 = box1[:, 2] * 10
    z4 = triangle[:, 2] * 10
    x5 = box2[:, 0] * 10
    x6 = box3[:, 0] * 10
    y5 = box2[:, 1] * 10
    y6 = box3[:, 1] * 10
    z5 = box2[:, 2] * 10
    z6 = box3[:, 2] * 10

    ax.scatter(x1, y1, z1, marker='o', c='r', s=50)
    ax.scatter(x2, y2, z2, marker='^', c='k', s=50)
    ax.scatter(x3, y3, z3, marker='o', c='r', s=50)
    ax.scatter(x4, y4, z4, marker='o', c='r', s=50)
    ax.scatter(x5, y5, z5, marker='o', c='r', s=50)
    ax.scatter(x6, y6, z6, marker='o', c='r', s=50)

    ax.plot(x1, y1, z1, 'b-')
    ax.plot(x3, y3, z3, 'b-')
    ax.plot(x4, y4, z4, 'b-')
    ax.plot(x5, y5, z5, 'b-')
    ax.plot(x6, y6, z6, 'b-')

    ax.view_init(-90, 270)
    plt.show()



    print('hi')
