import numpy as np
from scipy import linalg as la
from Camera import Camera
from SingleImage import SingleImage
from Reader import Reader
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


def ComputeApproximateVals(camera_points, ground_points, focal, camera_num):
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
    groundPointsXY = ground_points[:, 0:2]
    groundPointsXY = groundPointsXY.reshape(np.size(groundPointsXY), 1)
    groundPointsZ = ground_points[:, 2].T

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
    Z0 = np.average(groundPointsZ) + lam * focal * 0.001

    adjustment_results = {"X0": X0[0], "Y0": Y0[0], "Z0": Z0[0], "omega": 0, "phi": 0,
                          "kappa": kappa[0], "Camera number": camera_num}

    return adjustment_results


def vectorIntersction(ext1, ext2, focal, cameraPoints1, cameraPoints2):
    """
    Ray Intersection based on vector calculations.

    :param cameraPoints1: points in image space
    :param cameraPoints2: corresponding image points
    :param ext1: exterior orientation values for cam1
    :param ext2: exterior orientation values for cam2

    :type cameraPoints1: np.array nx2
    :type cameraPoints2: np.array nx2
    :type ext1: dict {X0,Y0,Z0,OMEGA,PHI,KAPPA,CAM_NUM}
    :type ext2: dict

    :return: groundPoints, e (the accuracy)

    :rtype: np.array nx3, nx1

    """
    o1 = np.array([ext1["X0"], ext1["Y0"], ext1["Z0"]])
    o2 = np.array([ext2["X0"], ext2["Y0"], ext2["Z0"]])
    R1 = Compute3DRotationMatrix(ext1["omega"], ext1["phi"], ext1["kappa"])
    R2 = Compute3DRotationMatrix(ext2["omega"], ext2["phi"], ext2["kappa"])
    modelPoints = []
    e = []
    for i in range(1):
        x1 = np.hstack((cameraPoints1[i:], -focal)) / 1000
        x2 = np.hstack((cameraPoints2[i:], -focal)) / 1000
        v1 = np.dot(R1, x1)
        v1 = np.reshape(v1 / la.norm(v1), (3, 1))
        v2 = np.dot(R2, x2)
        v2 = np.reshape(v2 / la.norm(v2), (3, 1))

        mat_inv = np.reshape(np.array([[np.dot(v1.T, v1), np.dot(-v1.T, v2)], [np.dot(v1.T, v2), np.dot(-v2.T, v2)]]),
                             (2, 2))
        mat = np.reshape(np.array([[np.dot((o2 - o1), v1)], [np.dot((o2 - o1), v2)]]), (2, 1))

        lam = np.dot(la.inv(mat_inv), mat)

        f = np.reshape(o1, (3, 1)) + np.dot(float(lam[0]), v1)
        g = np.reshape(o2, (3, 1)) + np.dot(float(lam[1]), v2)
        modelPoints.append((f + g) / 2)
        e.append((f - g) / 2)

    return np.reshape(np.array(modelPoints), (1, 3))  # , np.array(e)


def ComputeObservationVector(lb, img, groundPoints, camera_parameters):
    """
    Compute observation vector for solving the exterior orientation parameters of a single image
    based on their approximate values

    :param groundPoints: Ground coordinates of the control points
    :param camera_parameters

    :type groundPoints: np.array nx3

    :return: Vector l0

    :rtype: np.array nx1
    """

    n = groundPoints.shape[0]  # number of points

    # Coordinates subtraction
    dX = groundPoints[:, 0] - img.exteriorOrientationParameters[0]
    dY = groundPoints[:, 1] - img.exteriorOrientationParameters[1]
    dZ = groundPoints[:, 2] - img.exteriorOrientationParameters[2]
    dXYZ = np.vstack([dX, dY, dZ])
    rotated_XYZ = np.dot(img.rotationMatrix.T, dXYZ).T

    l0 = np.empty(n * 2)
    j = 0
    for i in range(n):
        r = ((lb[j] - camera_parameters["xp"]) * (lb[j] - camera_parameters["xp"]) + (
                lb[j + 1] - camera_parameters["yp"]) * (lb[j + 1] - camera_parameters["yp"])) ** 0.5
        l0[j] = camera_parameters["xp"] - camera_parameters["f"] * rotated_XYZ[i, 0] / rotated_XYZ[i, 2] + lb[j] * (
                camera_parameters["k1"] * r ** 2 + camera_parameters["k2"] * r ** 4 + camera_parameters[
            "k3"] * r ** 6) + camera_parameters["p1"] * (r ** 2 + 2 * lb[j] ** 2) + 2 * camera_parameters["p2"] * \
                lb[j] * lb[j + 1] + camera_parameters["b1"] * lb[j] + camera_parameters["b2"] * lb[j + 1]
        l0[j + 1] = camera_parameters["yp"] - camera_parameters["f"] * rotated_XYZ[i, 1] / rotated_XYZ[i, 2] + lb[
            j + 1] * (camera_parameters["k1"] * r ** 2 + camera_parameters["k2"] * r ** 4 + camera_parameters[
            "k3"] * r ** 6) + 2 * camera_parameters["p1"] * lb[j] * lb[j + 1] + camera_parameters["p2"] * (
                            r ** 2 + 2 * lb[j + 1] * lb[j + 1])

    # Computation of the observation vector based on approximate exterior orientation parameters:
    r = ()
    l0[::2] = camera_parameters["xp"] - camera_parameters[0] * rotated_XYZ[:, 0] / rotated_XYZ[:, 2]
    l0[1::2] = camera_parameters["yp"] - camera_parameters[0] * rotated_XYZ[:, 1] / rotated_XYZ[:, 2]

    return l0


if __name__ == "__main__":
    camera_parameters = {"f": 0, "xp": 0, "yp": 0, "k1": 0, "k2": 0, "k3": 0, "p1": 0, "p2": 0, "b1": 0, "b2": 0}

    # img dimensions #
    cols = 1280  # pix
    rows = 720  # pix

    # camera object: focal, principal point #
    focal = 26  # mm

    camera_parameters["f"] = focal

    cam = Camera(focal, [0, 0], 1.4e-3, camera_parameters)

    # creating 6 images for calibration
    img0deg = SingleImage(cam, np.array([cols, rows]), Reader.ReadSampleFile(r'raw data\0deg.json'))
    img30deg = SingleImage(cam, np.array([cols, rows]), Reader.ReadSampleFile(r'raw data\30deg.json'))
    img45deg = SingleImage(cam, np.array([cols, rows]), Reader.ReadSampleFile(r'raw data\45deg.json'))
    img60deg = SingleImage(cam, np.array([cols, rows]), Reader.ReadSampleFile(r'raw data\60deg.json'))
    img90deg = SingleImage(cam, np.array([cols, rows]), Reader.ReadSampleFile(r'raw data\90deg.json'))
    img225deg = SingleImage(cam, np.array([cols, rows]), Reader.ReadSampleFile(r'raw data\225deg.json'))

    ground_cp = np.array(
        [[0, 0, 0], [0, 3, 0], [0, 6, 0], [0, 9, 0], [0, -3, 0], [0, -6, 0], [0, -9, 0], [3, 0, 0], [6, 0, 0],
         [9, 0, 0], [12, 0, 0], [-3, 0, 0], [-6, 0, 0], [-9, 0, 0], [-12, 0, 0]])

    img0deg.ComputeApproximateVals(ground_cp)
    img30deg.ComputeApproximateVals(ground_cp)
    img45deg.ComputeApproximateVals(ground_cp)
    img60deg.ComputeApproximateVals(ground_cp)
    img90deg.ComputeApproximateVals(ground_cp)
    img225deg.ComputeApproximateVals(ground_cp)

    lb0deg = img0deg.lb()
    l0_0deg = img0deg.ComputeObservationVector(ground_cp)
    lb30deg = img30deg.lb()
    l0_30deg = img30deg.ComputeObservationVector(ground_cp)
    lb45deg = img45deg.lb()
    l0_45deg = img45deg.ComputeObservationVector(ground_cp)
    lb60deg = img60deg.lb()
    l0_60deg = img60deg.ComputeObservationVector(ground_cp)
    lb90deg = img90deg.lb()
    l0_90deg = img90deg.ComputeObservationVector(ground_cp)
    lb225deg = img225deg.lb()
    l0_225deg = img225deg.ComputeObservationVector(ground_cp)

    lb = np.vstack((lb0deg, lb30deg, lb45deg, lb60deg, lb90deg, lb225deg))
    l0 = np.vstack((l0_0deg, l0_30deg, l0_45deg, l0_60deg, l0_90deg, l0_225deg))

    l = lb - l0

    a_0deg = img0deg.ComputeDesignMatrix(ground_cp)
    a_30deg = img30deg.ComputeDesignMatrix(ground_cp)
    a_45deg = img45deg.ComputeDesignMatrix(ground_cp)
    a_60deg = img60deg.ComputeDesignMatrix(ground_cp)
    a_90deg = img90deg.ComputeDesignMatrix(ground_cp)
    a_225deg = img225deg.ComputeDesignMatrix(ground_cp)

    A = np.zeros((180, 46))
    aa = la.block_diag(a_0deg[:, 0:6], a_30deg[:, 0:6], a_45deg[:, 0:6], a_60deg[:, 0:6], a_90deg[:, 0:6],
                       a_225deg[:, 0:6])
    A[:, 0:36] = aa
    A[0:30, 36:46] = a_0deg[:, 6:16]
    A[30:60, 36:46] = a_30deg[:, 6:16]
    A[60:90, 36:46] = a_45deg[:, 6:16]
    A[90:120, 36:46] = a_60deg[:, 6:16]
    A[120:150, 36:46] = a_90deg[:, 6:16]
    A[150:180, 36:46] = a_225deg[:, 6:16]

    N = np.dot(A.T, A)
    u = np.dot(A.T, l)
    deltaX = np.dot(la.inv(N), u)

    # update EOF and camera pars
    img0deg.exteriorOrientationParameters = np.reshape(img0deg.exteriorOrientationParameters + deltaX[0:6].T, (6, 1))
    img30deg.exteriorOrientationParameters = np.reshape(img30deg.exteriorOrientationParameters + deltaX[6:12].T, (6, 1))
    img45deg.exteriorOrientationParameters = np.reshape(img45deg.exteriorOrientationParameters + deltaX[12:18].T,
                                                        (6, 1))
    img60deg.exteriorOrientationParameters = np.reshape(img60deg.exteriorOrientationParameters + deltaX[18:24].T,
                                                        (6, 1))
    img90deg.exteriorOrientationParameters = np.reshape(img90deg.exteriorOrientationParameters + deltaX[24:30].T,
                                                        (6, 1))
    img225deg.exteriorOrientationParameters = np.reshape(img225deg.exteriorOrientationParameters + deltaX[30:36].T,
                                                         (6, 1))

    camera_parameters["f"] += deltaX[36]
    camera_parameters["xp"] += deltaX[37]
    camera_parameters["yp"] += deltaX[38]
    camera_parameters["k1"] += deltaX[39]
    camera_parameters["k2"] += deltaX[40]
    camera_parameters["k3"] += deltaX[41]
    camera_parameters["p1"] += deltaX[42]
    camera_parameters["p2"] += deltaX[43]
    camera_parameters["b1"] += deltaX[44]
    camera_parameters["b2"] += deltaX[45]

    cam = Camera(camera_parameters["f"], [camera_parameters["xp"], camera_parameters["yp"]], 1.4e-3, camera_parameters)

    img0deg.camera = img30deg.camera = img45deg.camera = img60deg.camera = img90deg.camera = img225deg.camera = cam

    while la.norm(deltaX[36:46]) > 1e-6:
        lb0deg = img0deg.lb()
        l0_0deg = img0deg.ComputeObservationVector(ground_cp)
        lb30deg = img30deg.lb()
        l0_30deg = img30deg.ComputeObservationVector(ground_cp)
        lb45deg = img45deg.lb()
        l0_45deg = img45deg.ComputeObservationVector(ground_cp)
        lb60deg = img60deg.lb()
        l0_60deg = img60deg.ComputeObservationVector(ground_cp)
        lb90deg = img90deg.lb()
        l0_90deg = img90deg.ComputeObservationVector(ground_cp)
        lb225deg = img225deg.lb()
        l0_225deg = img225deg.ComputeObservationVector(ground_cp)

        lb = np.vstack((lb0deg, lb30deg, lb45deg, lb60deg, lb90deg, lb225deg))
        l0 = np.vstack((l0_0deg, l0_30deg, l0_45deg, l0_60deg, l0_90deg, l0_225deg))

        l = lb - l0

        a_0deg = img0deg.ComputeDesignMatrix(ground_cp)
        a_30deg = img30deg.ComputeDesignMatrix(ground_cp)
        a_45deg = img45deg.ComputeDesignMatrix(ground_cp)
        a_60deg = img60deg.ComputeDesignMatrix(ground_cp)
        a_90deg = img90deg.ComputeDesignMatrix(ground_cp)
        a_225deg = img225deg.ComputeDesignMatrix(ground_cp)

        A = np.zeros((180, 46))
        aa = la.block_diag(a_0deg[:, 0:6], a_30deg[:, 0:6], a_45deg[:, 0:6], a_60deg[:, 0:6], a_90deg[:, 0:6],
                           a_225deg[:, 0:6])
        A[:, 0:36] = aa
        A[0:30, 36:46] = a_0deg[:, 6:16]
        A[30:60, 36:46] = a_30deg[:, 6:16]
        A[60:90, 36:46] = a_45deg[:, 6:16]
        A[90:120, 36:46] = a_60deg[:, 6:16]
        A[120:150, 36:46] = a_90deg[:, 6:16]
        A[150:180, 36:46] = a_225deg[:, 6:16]

        N = np.dot(A.T, A)
        u = np.dot(A.T, l)
        deltaX = np.dot(la.inv(N), u)
        # update EOF and camera pars
        img0deg.exteriorOrientationParameters = np.reshape(img0deg.exteriorOrientationParameters + deltaX[0:6],
                                                           (6, 1))
        img30deg.exteriorOrientationParameters = np.reshape(img30deg.exteriorOrientationParameters + deltaX[6:12],
                                                            (6, 1))
        img45deg.exteriorOrientationParameters = np.reshape(img45deg.exteriorOrientationParameters + deltaX[12:18],
                                                            (6, 1))
        img60deg.exteriorOrientationParameters = np.reshape(img60deg.exteriorOrientationParameters + deltaX[18:24],
                                                            (6, 1))
        img90deg.exteriorOrientationParameters = np.reshape(img90deg.exteriorOrientationParameters + deltaX[24:30],
                                                            (6, 1))
        img225deg.exteriorOrientationParameters = np.reshape(img225deg.exteriorOrientationParameters + deltaX[30:36],
                                                             (6, 1))

        camera_parameters["f"] += deltaX[36]
        camera_parameters["xp"] += deltaX[37]
        camera_parameters["yp"] += deltaX[38]
        camera_parameters["k1"] += deltaX[39]
        camera_parameters["k2"] += deltaX[40]
        camera_parameters["k3"] += deltaX[41]
        camera_parameters["p1"] += deltaX[42]
        camera_parameters["p2"] += deltaX[43]
        camera_parameters["b1"] += deltaX[44]
        camera_parameters["b2"] += deltaX[45]

        cam = Camera(camera_parameters["f"], [camera_parameters["xp"], camera_parameters["yp"]], 1.4e-3,
                     camera_parameters)

        img0deg.camera = img30deg.camera = img45deg.camera = img60deg.camera = img90deg.camera = img225deg.camera = cam

    # tkinter load data \
    # filename = tk.filedialog.askopenfilename()

    # image_points = np.loadtxt(tk.filedialog.askopenfile())

    control_points = tk.filedialog.askopenfile(mode='r', title='Select CONTROL POINTS file',
                                               filetypes=[('Text File', '*.txt')]).readlines()
    cp_samples = tk.filedialog.askopenfile(mode='r', title='Select CONTROL POINTS SAMPLE file',
                                           filetypes=[('Text File', '*.txt')]).readlines()
    tp_samples = tk.filedialog.askopenfile(mode='r', title='Select TIE POINTS file',
                                           filetypes=[('Text File', '*.txt')]).readlines()

    control_points = np.stack([line.split() for line in control_points[1:-1][:]]).astype(float)
    cp_samples = np.stack([line.split() for line in cp_samples[1:-1][:]]).astype(float)
    tp_samples = np.stack([line.split() for line in tp_samples[1:-1][:]]).astype(float)

    # organizing points by image #
    img1_cp = []
    img1_tp = []
    img2_cp = []
    img2_tp = []
    img3_cp = []
    img3_tp = []
    img4_cp = []
    img4_tp = []
    img5_cp = []
    img5_tp = []
    img6_cp = []
    img6_tp = []

    for point in cp_samples:
        if point[0] == 1.0:
            img1_cp.append(point)
        elif point[0] == 2.0:
            img2_cp.append(point)
        elif point[0] == 3.0:
            img3_cp.append(point)
        elif point[0] == 4.0:
            img4_cp.append(point)
        elif point[0] == 5.0:
            img5_cp.append(point)
        else:
            img6_cp.append(point)

    imgs_cp = [np.array(img1_cp), np.array(img2_cp), np.array(img3_cp), np.array(img4_cp), np.array(img5_cp),
               np.array(img6_cp)]

    for point in tp_samples:
        if point[0] == 1.0:
            img1_tp.append(point)
        elif point[0] == 2.0:
            img2_tp.append(point)
        elif point[0] == 3.0:
            img3_tp.append(point)
        elif point[0] == 4.0:
            img4_tp.append(point)
        elif point[0] == 5.0:
            img5_tp.append(point)
        else:
            img6_tp.append(point)

    imgs_tp = [np.array(img1_tp), np.array(img2_tp), np.array(img3_tp), np.array(img4_tp), np.array(img5_tp),
               np.array(img6_tp)]

    # computing appx exterior orientaiton vals for every img
    appx_vals = []
    for img in imgs_cp:
        camera_points = []
        ground_points = []
        for point in img:
            for cp in control_points:
                if point[1] == cp[0]:
                    camera_points.append(point[2:])
                    ground_points.append(cp[1:])

        appx_vals.append(ComputeApproximateVals(np.array(camera_points), np.array(ground_points), focal, point[0]))

    # computing appx values for every tie point
    tiePoints = []
    cmptd_points = []
    for img1 in imgs_tp:
        for img2 in imgs_tp:
            for tp1 in img1:
                for tp2 in img2:
                    if tp1[0] != tp2[0] and tp1[1] == tp2[1] and tp1[1] not in cmptd_points:
                        tiePoints.append(np.hstack((np.reshape(tp1[1], (1, 1)),
                                                    vectorIntersction(appx_vals[int(tp1[0] - 1)],
                                                                      appx_vals[int(tp2[0] - 1)], focal, tp1[2:4],
                                                                      tp2[2:4]))))
                        cmptd_points.append(tp1[1])
                        break

    tiePoints = np.reshape(np.array(tiePoints), (len(tiePoints), 4))

    print('hiya')
