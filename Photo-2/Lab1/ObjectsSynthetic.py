import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np


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


fig_orthographic = plt.figure()
ax = fig_orthographic.add_subplot(111, projection='3d')
corners = cubeCorners(5)

side1 = np.vstack((corners[0, :], corners[3, :]))
side2 = np.vstack((corners[4, :], corners[7, :]))
side3 = np.vstack((corners[0, :], corners[4, :]))
side4 = np.vstack((corners[1, :], corners[5, :]))
side5 = np.vstack((corners[2, :], corners[6, :]))
side6 = np.vstack((corners[3, :], corners[7, :]))

ax.scatter(corners[:, 0], corners[:, 1], corners[:, 2], c='r')
ax.plot(corners[0:4, 0], corners[0:4, 1], corners[0:4, 2], c='r')
ax.plot(corners[4:8, 0], corners[4:8, 1], corners[4:8, 2], c='r')
ax.plot(side1[:, 0], side1[:, 1], side1[:, 2])
ax.plot(side2[:, 0], side2[:, 1], side2[:, 2])
ax.plot(side3[:, 0], side3[:, 1], side3[:, 2])
ax.plot(side4[:, 0], side4[:, 1], side4[:, 2])
ax.plot(side5[:, 0], side5[:, 1], side5[:, 2])
ax.plot(side6[:, 0], side6[:, 1], side6[:, 2])

plt.show()

print('hi')
