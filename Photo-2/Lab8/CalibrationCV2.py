import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

cbrow = 9
cbcol = 6
plotChessboardImages = True

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((cbrow * cbcol, 3), np.float32)
objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob(r'C:\Users\Saul\PycharmProjects\Photogrammetry\Photo-2\Lab8\calibration_images/*.JPG')
print(images)
images = [images[3]]

for fname in images:
    print("Current Image: ", fname)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (cbcol, cbrow), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        cornersTemp = np.array(corners2).reshape(np.shape(corners2)[0], np.shape(corners2)[2])

        if plotChessboardImages:
            plt.imshow(img, cmap='gray', interpolation='bicubic')

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9, 6), corners2, ret)

            plt.scatter(x=cornersTemp[:, 0], y=cornersTemp[:, 1], c='r', s=10)
            plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            plt.show()

# reprojection_error, camera_matrix, distortion_coefficient, rotation_v, translation_v
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("\nreprojection_error: ", ret)
print("\ncamera_matrix: ")
print(mtx)
fx = mtx[0, 0]
fy = mtx[1, 1]
cx, cy = mtx[0, 2], mtx[1, 2]
print("Fx, Fy: ", fx, fy, "\t Average: ", (fx + fy) / 2.)
print("Cx, Cy:", cx, cy)

# print("\nDistortion Coefficients:")
# print(dist)
# print("\nRotation V:")
# print(rvecs)
# print("\nTranslation V:")
# print(tvecs)
