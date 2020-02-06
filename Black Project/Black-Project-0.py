import numpy as np
from scipy import linalg as la
import Camera
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename


def Compute3DRotationMatrix(omega, phi, kappa):
    """
    Computes a 3x3 rotation matrix defined by euler angles given in radians

    :param omega: Rotation angle around the x-axis (radians)
    :param phi: Rotation angle around the y-axis (radians)
    :param kappa: Rotation angle around the z-axis (radians)

    :type omega: float
    :type phi: float
    :type kappa: float


    :return: The corresponding 3D rotation matrix
    :rtype: array  (3x3)

    """
    # Rotation matrix around the x-axis
    rOmega = np.array([[1, 0, 0],
                       [0, np.cos(omega), -np.sin(omega)],
                       [0, np.sin(omega), np.cos(omega)]], 'f')

    # Rotation matrix around the y-axis
    rPhi = np.array([[np.cos(phi), 0, np.sin(phi)],
                     [0, 1, 0],
                     [-np.sin(phi), 0, np.cos(phi)]], 'f')

    # Rotation matrix around the z-axis
    rKappa = np.array([[np.cos(kappa), -np.sin(kappa), 0],
                       [np.sin(kappa), np.cos(kappa), 0],
                       [0, 0, 1]], 'f')

    return np.dot(np.dot(rOmega, rPhi), rKappa)


def approximateImg2Camera(image_points, flight_height, dimensions, camera):
    camera_points = []

    cols = dimensions[0]
    rows = dimensions[1]
    f = camera.focalLength()
    h = flight_height

    scale = h * 0.001 / f

    for point in range(image_points[0]):
        x = scale * (point[2] - cols / 2)
        y = scale * (point[3] - rows / 2)
        camera_points.append([point[0], point[1], x, y])

    return np.array(camera_points)


def ComputeApproximateVals(camera_points, ground_points):
    """
    Compute exterior orientation approximate values via 2-D conform transformation

    :param camera_points: points in image space (x y)
    :param groundPoints: corresponding points in world system (X, Y, Z)

    :type camera_points: np.ndarray [nx2]
    :type groundPoints: np.ndarray [nx3]

    :return: Approximate values of exterior orientation parameters
    :rtype: np.ndarray or dict
    """

    # Find approximate values
    camera_points = camera_points.reshape(np.size(camera_points), 1)
    groundPointsXY = ground_points[0:2, :].T
    groundPointsXY = groundPointsXY.reshape(np.size(groundPointsXY), 1)
    groundPointsZ = ground_points[2, :].T

    n = int(len(camera_points))  # number of observations
    u = 4  # 4 conform parameters

    A = np.zeros((n, u))  # A matrix (n,u)

    j = 0
    for i in range(len(camera_points)):
        if i % 2 == 0:
            A[i, 0] = 1
            A[i, 1] = 0
            A[i, 2] = camera_points[j]
            A[i, 3] = camera_points[j + 1]
        else:
            A[i, 0] = 0
            A[i, 1] = 1
            A[i, 2] = camera_points[j + 1]
            A[i, 3] = -camera_points[j]
            j += 2

    X = np.dot(la.inv(np.dot(np.transpose(A), A)), np.dot(np.transpose(A), groundPointsXY))

    #  now we can compute the rest of the params
    X0 = X[0]
    Y0 = X[1]
    kappa = np.arctan2(-X[3], X[2])
    lam = np.sqrt(X[2] ** 2 + X[3] ** 2)
    Z0 = np.average(groundPointsZ) + (lam) * camera.focalLength

    adjustment_results = {"X0": X0[0], "Y0": Y0[0], "Z0": Z0[0], "omega": 0, "phi": 0,
                          "kappa": np.rad2deg(kappa[0])}

    return adjustment_results


if __name__ == "__main__":
    # img dimensions #
    cols = 800  # pix
    rows = 600  # pix
    # flight height #
    flight_height = 20  # meter

    # camera object: focal, principal point #
    cam1 = Camera.Camera(100, [0, 0], None, None)

    # tkinter load data \
    # filename = tk.filedialog.askopenfilename()

    # image_points = np.loadtxt(tk.filedialog.askopenfile())

    txt_file = tk.filedialog.askopenfile(mode='r').readlines()

    image_points = np.stack([line.split() for line in txt_file[1:-1][:]])

    camera_points = approximateImg2Camera(image_points, flight_height, [cols, rows], cam1)

    print('hi')
