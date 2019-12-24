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
    img2008points1 = rd.Reader.ReadSampleFile(r'IMG_2008_1.json')
    img2009points1 = rd.Reader.ReadSampleFile(r'IMG_2009_1.json')
    img2008points1_tmp = img2008points1
    img2009points1_tmp = img2009points1
    #  adjusting to camera system in pixels
    for i in range(len(img2008points1)) :
        img2008points1_tmp[i, 1] = 3648 - img2008points1_tmp[i, 1]
        img2009points1_tmp[i, 1] = 3648 - img2009points1_tmp[i, 1]
    #  turn pixels to mm's
    img2008points1mm = img2008points1_tmp * pixel_size
    img2009points1mm = img2009points1_tmp * pixel_size
    #  now we have our fiducials in mm's !

    print('hi')
