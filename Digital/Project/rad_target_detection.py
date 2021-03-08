import numpy as np
import itertools
import matplotlib.pyplot as plt
import cv2


def binarize_image(img):
    """

    :param img:
    :return:
    """
    blur = cv2.bilateralFilter(img, 10, 75, 75)  # d>=5, sigma values (10 is small, bigger the more effect)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 2)
    return thresh


def contour_image(binary, thresh=15):
    """

    :param binary:
    :param thresh:
    :return:
    """
    contours, hierarchy = cv2.findContours(binary,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    # filter out short ones
    contours = [cnt for cnt in contours if len(cnt) > thresh]
    return contours


def find_ellipses(conts):
    """

    :param conts:
    :return:
    """
    ellipses = []
    hulls = []
    # for each contour fit an ellipse
    for i, cnt in enumerate(conts):
        # get convex hull of contour
        hull = cv2.convexHull(cnt, returnPoints=True)
        # defects = cv2.convexityDefects(cnt, hull)

        # hulls must be greater or equal to 5 to be able to fit an ellipse to it
        if len(hull) >= 5:
            ellipse = cv2.fitEllipse(np.array(hull))
            ellipses.append(ellipse)
            hulls.append(hull)

    return ellipses, hulls


def draw_ellipse(
        img, center, axes, angle,
        startAngle, endAngle, color,
        thickness=3, lineType=cv2.LINE_8, shift=0):
    """

    :param img:
    :param center:
    :param axes:
    :param angle:
    :param startAngle:
    :param endAngle:
    :param color:
    :param thickness:
    :param lineType:
    :param shift:
    :return:
    """
    center = (
        int(round(center[0])),
        int(round(center[1]))
    )
    axes = (
        int(round(axes[0]) / 2),
        int(round(axes[1]) / 2)
    )
    # cv.ellipse(img, center, axes, angle, startAngle, endAngle, color[, thickness[, lineType[, shift]]])
    cv2.ellipse(
        img, center, axes, angle,
        startAngle, endAngle, color,
        thickness, lineType, shift)


def find_rad_targets(ellipses, epsilon=5, lower_thresh=3.5, upper_thresh=7):
    rad_targets = []
    ells = []
    # organize ellipses into one array
    for ell in ellipses:
        center = ell[0]
        axes = ell[1]
        angle = ell[-1]
        center = (
            int(round(center[0])),
            int(round(center[1]))
        )
        axes = (
            int(round(axes[0])),
            int(round(axes[1]))
        )
        ells.append([center[0], center[-1], axes[0], axes[-1], angle])

    ells = np.vstack(ells)

    # create an object to contain every possible pair of ellipses
    ell_pairs = itertools.combinations(ells, 2)

    for ell1, ell2 in ell_pairs:
        center1 = np.array([ell1[0], ell1[1]])
        center2 = np.array([ell2[0], ell2[1]])

        axes1 = np.array([ell1[2], ell1[3]])  # first is minor axis, second is major axis
        axes2 = np.array([ell2[2], ell2[3]])

        # check if in fact ellipses are concentric
        if np.linalg.norm(np.abs(center1 - center2)) < epsilon:
            # if they are concentric than check to see if correct ratio applies between the two
            if axes1[-1] < axes2[-1]:
                if lower_thresh < (axes2[-1] / axes1[-1]) < upper_thresh:
                    rad_targets.append(np.vstack((ell1, ell2)))
            else:
                if lower_thresh < (axes1[-1] / axes2[-1]) < upper_thresh:
                    rad_targets.append(np.vstack((ell2, ell1)))

    return rad_targets


if __name__ == '__main__':
    image = cv2.imread('small.jpg', 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # rgb image
    rgb_img = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # gray image

    # firstly we will threshold the image to make it binary
    binary = binarize_image(gray)

    # getting contours of the binary image
    conts = contour_image(binary, thresh=10)

    # next step is to fit ellipses to the contours and filter out ones that cannot be candidates for target
    ellipses, hulls = find_ellipses(conts)

    # find concentric ellipses, check to see if ratio applies between each pair of concentric ellipses
    rad_targets = find_rad_targets(ellipses, lower_thresh=3.5, upper_thresh=7.5)

    # drawing all ellipses on img
    # for ell in ellipses:
    #     draw_ellipse(rgb_img, ell[0], ell[1], ell[-1], 0, 360, (255, 0, 0))

    # drawing found targets on img
    for ell in rad_targets:
        ell1 = ell[0]
        ell2 = ell[1]
        draw_ellipse(rgb_img, (ell1[0], ell1[1]), (ell1[2], ell1[3]), ell1[-1], 0, 360, (255, 0, 0))
        draw_ellipse(rgb_img, (ell2[0], ell2[1]), (ell2[2], ell2[3]), ell2[-1], 0, 360, (0, 0, 255))

    # plotting and testing

    # plt.imshow(binary, cmap='gray')
    # plt.show()
    # plt.imshow(cv2.drawContours(image, hulls, -1, (255, 0, 0), 1))  # -1 for every contour and 1 for thickness
    # plt.show()
    plt.imshow(rgb_img)
    plt.show()
