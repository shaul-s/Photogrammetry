import numpy as np
import Reader as rd
import Camera as cam
import SingleImage as sg
import MatrixMethods as mm
from scipy import linalg as la


if __name__ == "__main__":

    cam_pars = rd.Reader.ReadCamFile(r'rc30.cam')
    fiducialsImg = rd.Reader.Readtxtfile(r'fiducialsImg.txt')

    cam = cam.Camera(cam_pars["f"], np.array([[cam_pars["xp"]], [cam_pars["yp"]]]), None, None, cam_pars["fiducials"])
    image = sg.SingleImage(cam)

    inner_pars = image.ComputeInnerOrientation(fiducialsImg)
    sig = np.dot(np.transpose(inner_pars["residuals"]), inner_pars["residuals"])/(len(inner_pars["residuals"])-6)





    # printing answers
    mm.PrintMatrix(inner_pars["params"], 'Inner Orientation Parameters')
    mm.PrintMatrix(inner_pars["residuals"], 'Inner Orientation Parameters')
    print(sig, '\n')

    print(image.ComputeInverseInnerOrientation())
    mm.PrintMatrix(sig * la.inv(inner_pars["N"]), 'var-cov', 8)
