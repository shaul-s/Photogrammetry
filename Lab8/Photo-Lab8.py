import numpy as np
import Reader as rd
import Camera as cam
import SingleImage as sg
import ImagePair as ip
from matplotlib import pyplot as plt
import MatrixMethods as mm
from scipy import linalg as la

if __name__ == "__main__" :
    #  computing model link without scale
    #  taking values from lab 7 --
    R2 = np.array([[0.93736404, 0.17563438, -0.30083427], [-0.1699276, 0.984417, 0.04525236],
                   [0.30409425, 0.00870211, 0.9526022]])
    R23 = np.array([[0.95362705, 0.14610846, -0.26314977], [-0.14700069, 0.9890003, 0.01640695],
                    [0.2626524, 0.02303708, 0.9646155]])
    o2 = np.array([1, -0.14151914, 0.22903829])
    b23 = np.array([1, -0.226124, 0.28123083])

    R3 = np.dot(R2, R23)
    o3 = o2 + np.dot(R2, b23)

    print("hi")
