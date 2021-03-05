import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from part2_targets import hough_circles

if __name__ == '__main__':
    image = cv2.imread(r'cabinet_targets.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (608, 342))

    circles = hough_circles(resized, vote_threshold=200, inner_threshold=50, minRadius=7, maxRadius=10)

    circles = np.uint16(np.around(circles)).squeeze()
    circles = circles[circles[:, 1] != 150]  # filtering out the wrong detections

    # cimg = cv2.resize(image, (608, 342))
    # for i in circles:
    #     # draw the outer circle
    #     cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 1)
    #     # draw the center of the circle
    #     cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 1)
    #
    # cimg = cv2.cvtColor(cimg, cv2.COLOR_BGR2RGB)
    # orig = cv2.cvtColor(cv2.resize(image, (608, 342)), cv2.COLOR_BGR2RGB)
    #
    # plt.subplot(121)
    # plt.title('Original')
    # plt.axis('off')
    # plt.imshow(orig)
    # plt.subplot(122)
    # plt.title('Circles Detected')
    # plt.imshow(cimg)
    # plt.axis('off')
    # plt.show()

    focal_length = 35.  # [mm] from camera properties

    scale = (1.4e-6 * 7.5) / 0.02  # circle radius in real-world system - 2cm; pixel size - 1.4 micron

    external_pnts = (circles[:, :-1] - np.array(
        [304, 171])) * scale  # translating to image center and applying scale factor

    external_pnts[:, -1] *= -1 

    fig_orthographic = plt.figure()
    ax = fig_orthographic.add_subplot(111, projection='3d')

    ax.scatter(external_pnts[:, 0], external_pnts[:, 1], 0,
               marker='^')  # image points sit on the same plan so z=0 was chosen to reduce complexity

    plt.show()
