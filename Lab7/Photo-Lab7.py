import numpy as np
import Reader as rd
import Camera as cam
import SingleImage as sg
import ImagePair as ip
import MatrixMethods as mm
from scipy import linalg as la

if __name__ == "__main__":
    cam = cam.Camera(cam_pars["f"], np.array([[cam_pars["xp"]], [cam_pars["yp"]]]), None, None, cam_pars["fiducials"])
