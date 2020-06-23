import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np
import PhotoViewer as pv


def cubeCorners(edgeSize):
    """
    get edge size and return 8 cube corners
    when the center of the cube is on (0,0,0)
    :param edgeSize:
    :return: cube corners
    """
    a = [edgeSize / 2, -edgeSize / 2, edgeSize / 2]
    b = [-edgeSize / 2, -edgeSize / 2, edgeSize / 2]
    c = [-edgeSize / 2, edgeSize / 2, edgeSize / 2]
    d = [edgeSize / 2, edgeSize / 2, edgeSize / 2]
    e = [edgeSize / 2, -edgeSize / 2, -edgeSize / 2]
    f = [-edgeSize / 2, -edgeSize / 2, -edgeSize / 2]
    g = [-edgeSize / 2, edgeSize / 2, -edgeSize / 2]
    h = [edgeSize / 2, edgeSize / 2, -edgeSize / 2]

    return np.array([a, b, c, d, e, f, g, h])


"""
def createCalibrationField(layers, depth, space):

    returns an array of 3d points as an optimal calibration field.
    you need to specify # of layers and depth between layers
    :param layers:
    :param depth:
    :return: np.array nx3

    points = []
    layer = layers
    num_of_pts = layers
    for i in range(layers):
        for j in range(num_of_pts ** 2 - 2 * i):
            points.append([j * space + i, i * space, layer * depth])
            points.append([j * space + i, (i + 1) * space, layer * depth])
            points.append([j * space + i, (i + 2) * space, layer * depth])
        layer -= 1
        space = space / 2

    return np.reshape(np.array(points), (len(points), 3))
"""


def createCalibrationField(layers, space, scale):
    """
    returns an array of 3d points as an optimal calibration field.
    you need to specify # of layers and depth between layers
    :param layers:
    :param space:
    :param scale:
    :return: np.array nx3
    """
    pts = []
    zs = []
    with open(r'star_points.txt') as file:
        lines = file.readlines()
        for line in lines:
            line = line.split(',')
            pts.append(line)
            zs.append(0)

    pts = np.array(pts).astype(float)
    zs = np.reshape(np.array(zs).astype(float), (len(zs), 1))
    pts = (scale ** 2) * np.array(np.hstack((pts, zs)))

    pts_copy = pts.copy()
    for i in range(1, layers):
        deeperpts = (scale ** i) * pts_copy
        deeperpts[:, 2] = -space * i
        pts = np.vstack((pts, deeperpts))

    return pts


if __name__ == '__main__':
    pts = createCalibrationField(3, 2, 5)

    fig_orthographic = plt.figure()
    ax = fig_orthographic.add_subplot(111, projection='3d')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
    plt.show()

    z = np.array([0, 0, 0, 0, 0, 0], 'f')
    print('hi')
    """
    
    corners = cubeCorners(5)
    
    side1 = np.vstack((corners[0, :], corners[3, :]))
    side2 = np.vstack((corners[4, :], corners[7, :]))
    side3 = np.vstack((corners[0, :], corners[4, :]))
    side4 = np.vstack((corners[1, :], corners[5, :]))
    side5 = np.vstack((corners[2, :], corners[6, :]))
    side6 = np.vstack((corners[3, :], corners[7, :]))
    
    ax.scatter(corners[:, 0], corners[:, 1], corners[:, 2])
    ax.plot(corners[0:4, 0], corners[0:4, 1], corners[0:4, 2])
    ax.plot(corners[4:8, 0], corners[4:8, 1], corners[4:8, 2])
    ax.plot(side1[:, 0], side1[:, 1], side1[:, 2])
    ax.plot(side2[:, 0], side2[:, 1], side2[:, 2])
    ax.plot(side3[:, 0], side3[:, 1], side3[:, 2])
    ax.plot(side4[:, 0], side4[:, 1], side4[:, 2])
    ax.plot(side5[:, 0], side5[:, 1], side5[:, 2])
    ax.plot(side6[:, 0], side6[:, 1], side6[:, 2])
    
    print('hi')
    """
