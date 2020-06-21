import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Camera import *
from SingleImage import *
from Reader import *

if __name__ == '__main__':
    # reading sampled points
    x_axis = Reader.ReadSampleFile(r"x_axis.json")
    y_axis = Reader.ReadSampleFile(r"y_axis.json")
    z_axis = Reader.ReadSampleFile(r"z_axis.json")

    # applying homogeneous presentation
    x_axis = np.hstack((x_axis, np.ones((max(x_axis.shape), 1))))
    y_axis = np.hstack((y_axis, np.ones((max(y_axis.shape), 1))))
    z_axis = np.hstack((z_axis, np.ones((max(z_axis.shape), 1))))

    # camera object + finding vanishing points
    cam = Camera(None, None, {'K1': -0.5104e-8, 'K2': 0.1150e-12},
                 {'P1': -0.8776e-7, 'P2': 0.1722e-7}, None, None)
    img = SingleImage(cam)

    vp_x = img.findVanishingPoint(x_axis)
    vp_y = img.findVanishingPoint(y_axis)
    vp_z = img.findVanishingPoint(z_axis)

    # computing calibration matrix K
    K = cam.compute_CalibrationMatrix(np.reshape(vp_x, 3), np.reshape(vp_y, 3), np.reshape(vp_z, 3))

    # compute rotation matrix
    img.rotationMatrix_vanishingPoints(vp_x, vp_y, vp_z)

    # check if what we did is correct - >
    e1 = np.dot(np.dot(img.rotationMatrix.T, la.inv(K)), vp_x.T)
    e1 = e1 / la.norm(e1)

    # try to reconstruct faces
    face1 = Reader.ReadSampleFile(r"face1.json")
    face2 = Reader.ReadSampleFile(r"face2.json")
    face3 = Reader.ReadSampleFile(r"face3.json")
    # fix shared points
    face2[0] = face1[1]
    face2[-1] = face1[-2]
    face3[0] = face1[0]
    face3[1] = face1[1]
    face3[-2] = face2[1]

    face1 = np.hstack((face1, np.full((max(face1.shape), 1), -cam.focalLength)))
    face2 = np.hstack((face2, np.full((max(face2.shape), 1), -cam.focalLength)))
    face3 = np.hstack((face3, np.full((max(face3.shape), 1), -cam.focalLength)))

    # correcting to 'ideal' camera system
    for i in range(len(face1)):
        face1[i] = np.dot(la.inv(K), face1[i])
        face2[i] = np.dot(la.inv(K), face2[i])
        face3[i] = np.dot(la.inv(K), face3[i])

    # computing vanishing points of faces
    vp1face1 = np.cross(face1[0], face2[0])
    vp1face1 = vp1face1 / vp1face1[-1]
    vp2face1 = np.cross(face1[0], face1[-1])
    vp2face1 = vp2face1 / vp2face1[-1]

    vp1face2 = np.cross(face1[1], face1[2])
    vp1face2 = vp1face2 / vp1face2[-1]
    vp2face2 = np.cross(face1[1], face2[1])
    vp2face2 = vp2face2 / vp2face2[-1]

    vp1face3 = np.cross(face1[0], face1[1])
    vp1face3 = vp1face3 / vp1face3[-1]
    vp2face3 = np.cross(face1[0], face3[-1])
    vp2face3 = vp2face3 / vp2face3[-1]

    # computing face normals
    normal_face1 = img.faceNormal_imageSpace(vp1face1, vp2face1)
    normal_face1 = img.faceNormal_objectSpace(normal_face1)

    normal_face2 = img.faceNormal_imageSpace(vp1face2, vp2face2)
    normal_face2 = img.faceNormal_objectSpace(normal_face2)

    normal_face3 = img.faceNormal_imageSpace(vp1face3, vp2face3)
    normal_face3 = img.faceNormal_objectSpace(normal_face3)

    normals = [normal_face1, normal_face2, normal_face3]

    os_points1 = []
    os_points2 = []
    os_points3 = []

    roh1 = img.scale_firstFace(normal_face1, 5, face1[0], face1[1])
    roh2 = img.scale_firstFace(normal_face2, 10, face2[0], face2[1])
    roh3 = img.scale_firstFace(normal_face3, 5, face3[0], face3[1])

    for pnt in face1:
        pnt = np.dot(img.rotationMatrix, pnt)
        li = roh1 / np.dot(normal_face1, pnt)
        os_points1.append(np.dot(li, pnt))

    for pnt in face2:
        pnt = np.dot(img.rotationMatrix, pnt)
        li = roh2 / np.dot(normal_face2, pnt)
        os_points2.append(np.dot(li, pnt))

    for pnt in face3:
        pnt = np.dot(img.rotationMatrix, pnt)
        li = roh3 / np.dot(normal_face3, pnt)
        os_points3.append(np.dot(li, pnt))

    fig_orthographic = plt.figure()
    ax = fig_orthographic.add_subplot(111, projection='3d')
    os_points1 = np.vstack((os_points1, os_points1[0]))
    os_points2 = np.vstack((os_points2, os_points2[0]))
    os_points3 = np.vstack((os_points3, os_points3[0]))
    ax.plot(os_points1[:, 0], os_points1[:, 1], os_points1[:, 2], marker='o')
    # ax.plot(os_points2[:, 0], os_points2[:, 1], os_points2[:, 2], marker='^')
    ax.plot(os_points3[:, 0], os_points3[:, 1], os_points3[:, 2], marker='*')

    plt.show()

    print('hi')
