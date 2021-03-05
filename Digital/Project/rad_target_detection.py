import numpy as np
import matplotlib.pyplot as plt
import cv2


def binarize_image(img):
    """

    :param img:
    :return:
    """
    blur = cv2.bilateralFilter(img, 5, 75, 75)  # d>=5, sigma values (10 is small, bigger the more effect)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, 2)
    return thresh


def contour_image(binary, thresh):
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
    ellipses = []
    hulls = []
    # for each contour fit an ellipse
    for i, cnt in enumerate(conts):
        # get convex hull of contour
        hull = cv2.convexHull(cnt, returnPoints=True)
        # defects = cv2.convexityDefects(cnt, hull)

        if len(hull) > 5:
            ellipse = cv2.fitEllipse(np.array(hull))
            ellipses.append(ellipse)
            hulls.append(hulls)

    return ellipses, hulls


if __name__ == '__main__':
    image = cv2.imread('small.jpg', 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # firstly we will threshold the image to make it binary
    binary = binarize_image(gray)

    # getting contours of the binary image
    conts = contour_image(binary, 20)

    # next step is to fit ellipses to the contours and filter out ones that cannot be candidates for target
    find_ellipses(conts)


    # plotting and testing
    plt.imshow(cv2.drawContours(image, conts, -1, (255, 0, 0), 1))  # -1 for every contour and 1 for thickness
    plt.show()

