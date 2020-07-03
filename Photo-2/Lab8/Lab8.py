import numpy as np
import random
from scipy import linalg as la
from scipy import matrix
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import cv2
from Camera import *
from ImagePair import *
from SingleImage import *
from PhotoViewer import *


def computeDesignMatrix_Fundamental(pts1, pts2):
    """
    return design matrix for Fundamental matrix computation given a set of homologic points
    :param pts1:
    :param pts2:
    :return: design matrix A for extracting the fundamental matrix
    """
    A = np.zeros((len(pts1), 9))
    A[:, 0] = pts1[:, 0] * pts2[:, 0]
    A[:, 1] = pts1[:, 1] * pts2[:, 0]
    A[:, 2] = pts2[:, 0]
    A[:, 3] = pts1[:, 0] * pts2[:, 1]
    A[:, 4] = pts1[:, 1] * pts2[:, 1]
    A[:, 5] = pts2[:, 1]
    A[:, 6] = pts1[:, 0]
    A[:, 7] = pts1[:, 1]
    A[:, -1] = np.ones(A[:, -1].shape)
    return A


def normalizePoints(img_shape, pts1, pts2):
    """
    return an array of the normalized points between -1, 1 given img and homologic points
    both images are presumed to be same size
    :param img:
    :param pts1:
    :param pts2:
    :return: the nomalizing matrix and normalized points
    """
    xmax = img_shape[0]
    ymax = img_shape[1]
    xm = 0.5 * (0 + xmax)
    ym = 0.5 * (0 + ymax)
    dx = xmax
    dy = ymax
    S = np.array([[2 / dx, 0, -2 * (xm / dx)], [0, 2 / dy, -2 * (ym / dy)], [0, 0, 1]])
    pts1_normalized = []
    pts2_normalized = []
    for i in range(len(pts1)):
        pts1_normalized.append(np.dot(S, pts1[i]))
        pts2_normalized.append(np.dot(S, pts2[i]))

    pts1_normalized = np.vstack(pts1_normalized)
    pts2_normalized = np.vstack(pts2_normalized)

    return S, pts1_normalized, pts2_normalized


def findHomologicPoints(img1, img2, draw_images=0):
    """
    use SIFT & opencv to locate points from img1 in img2
    :param img1: query image
    :param img2: train image
    :param draw_images: flag for drawing the homologic points found
    :return: pts1 & pts2
    """
    # find homologic points
    MIN_MATCH_COUNT = 10

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    good = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.5 * n.distance:
            matchesMask[i] = [1, 0]
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    if len(good) > MIN_MATCH_COUNT and draw_images:
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=0)

        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

        plt.imshow(img3, ), plt.show()

    return np.vstack(pts1), np.vstack(pts2)


def ransacFundamental(img1, pts1, pts2, tolerance=0.01, normalize=1):
    """

    :param img1:
    :param pts1:
    :param pts2:
    :param tolerance:
    :param normalize:
    :return:
    """

    def check_minCount(F, minCount, pts1, pts2):
        counter = 0
        for i in range(len(pts1)):
            if np.dot(np.dot(pts2[i].T, F), pts1[i]) <= tolerance:
                counter += 1
        print(counter, '/', len(pts1))
        if counter >= minCount:
            return False
        return True

    # setting desired count for matching points, in our case 85%
    n = len(pts1)
    good_pts1 = []
    good_pts2 = []
    MIN_MATCH_COUNT = np.int(0.85 * n)
    # initial value for the fundamental matrix
    F = np.ones((3, 3))
    while check_minCount(F, MIN_MATCH_COUNT, pts1, pts2):
        # select random 8 points - minimum for fundamental matrix extraction
        random_idx = []
        for i in range(8):
            random_idx.append(random.randrange(n))

        rand8_pts1 = np.vstack((pts1[random_idx[0]], pts1[random_idx[1]], pts1[random_idx[2]],
                                pts1[random_idx[3]], pts1[random_idx[4]], pts1[random_idx[5]],
                                pts1[random_idx[6]], pts1[random_idx[7]]))
        rand8_pts2 = np.vstack((pts2[random_idx[0]], pts2[random_idx[1]], pts2[random_idx[2]],
                                pts2[random_idx[3]], pts2[random_idx[4]], pts2[random_idx[5]],
                                pts2[random_idx[6]], pts2[random_idx[7]]))

        # normalizing pts between -1,1
        S, pts1_normalized, pts2_normalized = normalizePoints(img1.shape, rand8_pts1, rand8_pts2)
        # creating design matrix
        A = computeDesignMatrix_Fundamental(pts1_normalized, pts2_normalized)
        # solving homogeneous equation
        N = np.dot(A.T, A)
        egi_vals, egi_vect = np.linalg.eig(N)
        min_egi_val_index = np.argmin(egi_vals)
        v = egi_vect[:, min_egi_val_index]
        F = v.reshape((3, 3))
        # svd decomposition for setting one singular value to zero
        u, s, vh = la.svd(F)
        s[-1] = 0
        fixed_F = np.dot(np.dot(u, np.diag(s)), vh)
        # converting F back to the normal coordinates and normalizing to norm=1
        fixed_F = np.dot(np.dot(S.T, fixed_F), S)
        F = fixed_F / la.norm(fixed_F)

    # filtering out the bad points
    for i in range(len(pts1)):
        if np.dot(np.dot(pts2[i].T, F), pts1[i]) <= tolerance:
            good_pts1.append(pts1[i])
            good_pts2.append(pts2[i])
    # using rest of points to adjust the new and improved fundamental matrix
    # normalizing pts between -1,1
    S, pts1_normalized, pts2_normalized = normalizePoints(img1.shape, good_pts1, good_pts2)
    # creating design matrix
    A = computeDesignMatrix_Fundamental(pts1_normalized, pts2_normalized)
    # solving homogeneous equation
    N = np.dot(A.T, A)
    egi_vals, egi_vect = np.linalg.eig(N)
    min_egi_val_index = np.argmin(egi_vals)
    v = egi_vect[:, min_egi_val_index]
    F = v.reshape((3, 3))
    # svd decomposition for setting one singular value to zero
    u, s, vh = la.svd(F)
    s[-1] = 0
    fixed_F = np.dot(np.dot(u, np.diag(s)), vh)
    # converting F back to the normal coordinates and normalizing to norm=1
    fixed_F = np.dot(np.dot(S.T, fixed_F), S)
    F = fixed_F / la.norm(fixed_F)

    return F, np.vstack(good_pts1), np.vstack(good_pts2)


# METHOD FROM OPENCV
def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1.astype(int)), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2.astype(int)), 5, color, -1)
    return img1, img2


if __name__ == '__main__':
    # computing K camera calibration matrix using cv2
    K, rvecs, tvecs = Camera.calibrateCamera_checkers()
    # print(pd.DataFrame(K))

    # loading images
    img1 = cv2.imread('images/20200622_140804.jpg', 0)  # queryImage 'box'
    img2 = cv2.imread('images/20200622_140813.jpg', 0)  # trainImage 'box-in-scene'

    # computing fundamental matrix
    # locating homologic points
    pts1, pts2 = findHomologicPoints(img1, img2)

    ##### TRY TO USE OPENCV FOR GETTING FUNDAMENTAL MATRIX AND DRAW EPIPOLAR LINES
    # opencv drawing
    F1, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    F1 = F1 / la.norm(F1)

    # We select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    # lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F1)
    # lines1 = lines1.reshape(-1, 3)
    # img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)
    # # Find epilines corresponding to points in left image (first image) and
    # # drawing its lines on right image
    # lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F1)
    # lines2 = lines2.reshape(-1, 3)
    # img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)
    # plt.subplot(121),
    # # plt.scatter(pts1[:, 0], pts1[:, 1])
    # plt.imshow(img5), plt.axis('off')
    # plt.subplot(122), plt.imshow(img3), plt.axis('off')
    # plt.show()

    ####

    # converting points to homogeneous presentation
    # f = np.mean(np.array([K[0, 0], K[1, 1]]))
    pts1 = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2 = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    # correcting points to ideal camera
    K[0, 0] = -K[0, 0]
    K[1, 1] = -K[1, 1]
    xp = K[0, -1]
    yp = K[1, -1]
    ppa = np.array([xp, yp, 0])
    for i in range(len(pts1)):
        # pts1[i] = np.dot(la.inv(K), pts1[i])
        # pts2[i] = np.dot(la.inv(K), pts2[i])
        pts1[i] = pts1[i] - ppa
        pts2[i] = pts2[i] - ppa

    # ransac adjusting the fundamental matrix
    F, good_pts1, good_pts2 = ransacFundamental(img1, pts1, pts2, tolerance=0.01)

    # find epipole using null space
    # left null space is the 1st image epipole point
    epipole1 = la.null_space(matrix(F.T))
    # translate to image space
    epipole1 = (epipole1.T / epipole1.T[:, -1]) + ppa.T
    # show epipole on image
    # plt.imshow(img2, cmap='gray')
    # plt.scatter(epipole1.T[0], epipole1.T[1], s=200, c='g')
    # plt.axis('off')
    # plt.show()

    # computing epi-polar line for each homologic points and distance from it
    epi_lines = []
    distances = []
    for i, p in enumerate(good_pts2):
        epi_line = np.dot(p.T, F)
        epi_lines.append(epi_line)
        distances.append(np.abs(epi_line[0] * good_pts1[i, 0] + epi_line[1] * good_pts1[i, 1] + epi_line[-1]) / np.sqrt(
            epi_line[0] ** 2 + epi_line[1] ** 2))
    epi_lines = np.vstack(epi_lines)
    distances = np.vstack(distances)
    # filtering points above a certain distance from line
    idx = np.where(distances < 1)[0]
    good_pts1 = good_pts1[idx]
    good_pts2 = good_pts2[idx]
    good_epi_lines = epi_lines[idx]

    # for drawing purposes going back to image system
    good_pts1_image = []
    good_pts2_image = []
    for i in range(len(good_pts1)):
        good_pts1_image.append(good_pts1[i] + ppa)
        good_pts2_image.append(good_pts2[i] + ppa)

    good_pts1_image = np.vstack(good_pts1_image)
    good_pts2_image = np.vstack(good_pts2_image)

    # low_right_x = 2268.
    # upper_left_x = 0.
    # xs = (upper_left_x, low_right_x) # - ppa[0]
    # #
    #
    # plt.scatter(good_pts1[:, 0], good_pts1[:, 1], c='g')
    # for line in good_epi_lines:
    #     y1 = (-line[0] * upper_left_x - line[-1]) / line[1]
    #     y2 = (-line[0] * low_right_x - line[-1]) / line[1]
    #     ys = (y1, y2) #  - ppa[1]
    #     # plt.scatter(xs, ys)
    #     # plt.plot(xs, ys)
    #     cv2.line(img1, (0, 2268), (int(y1), int(y2)), (0, 255, 0), thickness=2)
    # plt.imshow(img1, cmap='gray')
    # plt.show()

    # plt.subplot(121)
    # plt.axis('off')
    # plt.imshow(img1, cmap='gray')
    # plt.scatter(good_pts1_image[:, 0], good_pts1_image[:, 1])
    # plt.subplot(122)
    # plt.axis('off')
    # plt.imshow(img2, cmap='gray')
    # plt.scatter(good_pts2_image[:, 0], good_pts2_image[:, 1])
    # plt.show()

    # compute essential matrix using F
    E = np.dot(np.dot(K.T, F), K)
    u, s, vh = la.svd(E)
    s = np.array([1, 1, 0])
    E = np.dot(np.dot(u, np.diag(s)), vh)
    E = E / la.norm(E)

    # extracting mutual orientation parameters
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    b1 = u[:, -1]
    b2 = -u[:, -1]
    R1 = np.dot(np.dot(u, W), vh.T)
    R2 = np.dot(np.dot(u, W.T), vh.T)

    # defining image pair
    cam = Camera(-K[0, 0], [K[0, -1], K[1, -1]], None, None, None, 0.5)

    image1 = SingleImage(cam)
    image2 = SingleImage(cam)

    image_pair = ImagePair(image1, image2)
    image_pair.RotationMatrix_Image1 = np.eye(3)

    fig_orthographic = plt.figure()
    ax1 = fig_orthographic.add_subplot(221, projection='3d')
    ax2 = fig_orthographic.add_subplot(222, projection='3d')
    ax3 = fig_orthographic.add_subplot(223, projection='3d')
    ax4 = fig_orthographic.add_subplot(224, projection='3d')

    # try1
    ax1.set_title('R1, b1')
    image_pair.RotationMatrix_Image2 = R1
    image_pair.PerspectiveCenter_Image2 = b1

    model_points = image_pair.ImagesToModel(good_pts1[5:10, 0:2], good_pts2[5:10, 0:2], 'vector')

    x1 = image_pair.PerspectiveCenter_Image1[:, None]
    x2 = image_pair.PerspectiveCenter_Image2[:, None]
    drawImageFrame(cam.sensorSize, cam.sensorSize, image_pair.RotationMatrix_Image1, x1, -cam.focalLength / 10000, 1,
                   ax1)
    drawImageFrame(cam.sensorSize, cam.sensorSize, image_pair.RotationMatrix_Image2, x2, -cam.focalLength / 10000, 1,
                   ax1)
    drawOrientation(image_pair.RotationMatrix_Image1, x1, 0.5, ax1)
    drawOrientation(image_pair.RotationMatrix_Image2, x2, 0.5, ax1)
    drawRays(model_points[0], x1, ax1, 'r')
    drawRays(model_points[0], x2, ax1, 'g')

    ax1.scatter(model_points[0][:, 0], model_points[0][:, 1], model_points[0][:, 2], marker='^')

    # try2
    ax2.set_title('R2, b2')
    image_pair.RotationMatrix_Image2 = R2
    image_pair.PerspectiveCenter_Image2 = b2

    model_points = image_pair.ImagesToModel(good_pts1[0:5, 0:2], good_pts2[0:5, 0:2], 'vector')

    x1 = image_pair.PerspectiveCenter_Image1[:, None]
    x2 = image_pair.PerspectiveCenter_Image2[:, None]
    drawImageFrame(cam.sensorSize, cam.sensorSize, image_pair.RotationMatrix_Image1, x1, -cam.focalLength / 10000, 1,
                   ax2)
    drawImageFrame(cam.sensorSize, cam.sensorSize, image_pair.RotationMatrix_Image2, x2, -cam.focalLength / 10000, 1,
                   ax2)
    drawOrientation(image_pair.RotationMatrix_Image1, x1, 0.5, ax2)
    drawOrientation(image_pair.RotationMatrix_Image2, x2, 0.5, ax2)
    drawRays(model_points[0], x1, ax2, 'r')
    drawRays(model_points[0], x2, ax2, 'g')

    ax2.scatter(model_points[0][:, 0], model_points[0][:, 1], model_points[0][:, 2], marker='^')

    # try3
    ax3.set_title('R1, b2')
    image_pair.RotationMatrix_Image2 = R1
    image_pair.PerspectiveCenter_Image2 = b2

    model_points = image_pair.ImagesToModel(good_pts1[0:5, 0:2], good_pts2[0:5, 0:2], 'vector')

    x1 = image_pair.PerspectiveCenter_Image1[:, None]
    x2 = image_pair.PerspectiveCenter_Image2[:, None]
    drawImageFrame(cam.sensorSize, cam.sensorSize, image_pair.RotationMatrix_Image1, x1, -cam.focalLength / 10000, 1,
                   ax3)
    drawImageFrame(cam.sensorSize, cam.sensorSize, image_pair.RotationMatrix_Image2, x2, -cam.focalLength / 10000, 1,
                   ax3)
    drawOrientation(image_pair.RotationMatrix_Image1, x1, 0.5, ax3)
    drawOrientation(image_pair.RotationMatrix_Image2, x2, 0.5, ax3)
    drawRays(model_points[0], x1, ax3, 'r')
    drawRays(model_points[0], x2, ax3, 'g')

    ax3.scatter(model_points[0][:, 0], model_points[0][:, 1], model_points[0][:, 2], marker='^')

    # try4
    ax4.set_title('R2, b1')
    image_pair.RotationMatrix_Image2 = R2
    image_pair.PerspectiveCenter_Image2 = b1

    model_points = image_pair.ImagesToModel(good_pts1[0:5, 0:2], good_pts2[0:5, 0:2], 'vector')

    x1 = image_pair.PerspectiveCenter_Image1[:, None]
    x2 = image_pair.PerspectiveCenter_Image2[:, None]
    drawImageFrame(cam.sensorSize, cam.sensorSize, image_pair.RotationMatrix_Image1, x1, -cam.focalLength / 10000, 1,
                   ax4)
    drawImageFrame(cam.sensorSize, cam.sensorSize, image_pair.RotationMatrix_Image2, x2, -cam.focalLength / 10000, 1,
                   ax4)
    drawOrientation(image_pair.RotationMatrix_Image1, x1, 0.5, ax4)
    drawOrientation(image_pair.RotationMatrix_Image2, x2, 0.5, ax4)
    drawRays(model_points[0], x1, ax4, 'r')
    drawRays(model_points[0], x2, ax4, 'g')

    ax4.scatter(model_points[0][:, 0], model_points[0][:, 1], model_points[0][:, 2], marker='^')

    # draw all model points
    fig_orthographic = plt.figure()
    ax = fig_orthographic.add_subplot(111, projection='3d')
    image_pair.RotationMatrix_Image2 = R2
    image_pair.PerspectiveCenter_Image2 = b2
    model_points = image_pair.ImagesToModel(good_pts1[:, 0:2], good_pts2[:, 0:2], 'vector')
    ax.scatter(model_points[0][:, 0], model_points[0][:, 1], model_points[0][:, 2], marker='^')

    # distances -> size of e vector we got from vectoric intersection
    es = la.norm(model_points[1], axis=1)

    plt.show()

    print(pd.DataFrame(F))
