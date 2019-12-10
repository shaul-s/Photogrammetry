import numpy as np
import Reader as rd
import Camera as cam
import SingleImage as sg
import MatrixMethods as mm
from scipy import linalg as la


if __name__ == "__main__":

    cam_pars = rd.Reader.ReadCamFile(r'rc30.cam')
    fiducialsImg = rd.Reader.Readtxtfile(r'fiducialsImg.txt')
    control_p = rd.Reader.photoModXMLReader(r'lab5.xml')

    cam = cam.Camera(cam_pars["f"], np.array([[cam_pars["xp"]], [cam_pars["yp"]]]), None, None, cam_pars["fiducials"])
    image = sg.SingleImage(cam)

    inner_pars = image.ComputeInnerOrientation(fiducialsImg)
    sig = np.dot(np.transpose(inner_pars["residuals"]), inner_pars["residuals"])/(len(inner_pars["residuals"])-6)

    points = image.ImageToCamera(fiducialsImg)

    ###   PART B - EXTERIOR ORIENTATION  ###
    fiducialsImg3574 = np.zeros((2, int(len(control_p[2])/2)))  # initializing the fiducials array for img 3574
    GCP = np.zeros((3, int(len(control_p[1]))))
    j = 0
    for row in control_p[2]:
        if row[0] == '3574':
            fiducialsImg3574[0,j] = row[1]
            fiducialsImg3574[1, j] = row[2]
        else:
            j -= 1

    for i, row in enumerate(control_p[1]):
        GCP[0, i] = row[2]
        GCP[1, i] = row[3]
        GCP[2, i] = row[4]

    fiducialsImg3574 = fiducialsImg3574.T
    fiducialsCamera3574 = image.ImageToCamera(fiducialsImg3574)

    appx_vals = image.ComputeExteriorOrientation(fiducialsCamera3574, GCP, 0.001)





    # printing answers
    mm.PrintMatrix(fiducialsCamera3574, 'Image 3574 Control Points in Camera System')
    """
    mm.PrintMatrix(inner_pars["params"], 'Inner Orientation Parameters')
    mm.PrintMatrix(inner_pars["residuals"], 'Inner Orientation Parameters')
    print(image.ComputeInverseInnerOrientation())
    mm.PrintMatrix(sig * la.inv(inner_pars["N"]), 'var-cov', 8)
    """
