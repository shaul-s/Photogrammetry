import numpy as np
import Reader as rd
import Camera as cam
import SingleImage as sg
import ImagePair as ip
import MatrixMethods as mm
from scipy import linalg as la

def poly_area(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

if __name__ == "__main__":
    ### Reading data ###
    samples = rd.Reader.photoModXMLReader(r'Lab6-Photomod.xml')
    cam_pars = rd.Reader.ReadCamFile(r'rc30.cam')
    fiducialsImg3574 = rd.Reader.Readtxtfile(r'fiducialsImg3574.txt')
    fiducialsImg3575 = rd.Reader.Readtxtfile(r'fiducialsImg3575.txt')
    cam = cam.Camera(cam_pars["f"], np.array([[cam_pars["xp"]], [cam_pars["yp"]]]), None, None, cam_pars["fiducials"])
    image3574 = sg.SingleImage(cam)
    image3575 = sg.SingleImage(cam)
    control_p = rd.Reader.photoModXMLReader(r'lab5.xml')
    GCP = np.zeros((3, int(len(control_p[1]))))
    for i, row in enumerate(control_p[1]):
        GCP[0, i] = row[2]
        GCP[1, i] = row[3]
        GCP[2, i] = row[4]
    ICP3574 = []
    ICP3575 = []
    for row in control_p[2] :
        if row[0] == '3574' :
            ICP3574.append([row[1], row[2]])
        else:
            ICP3575.append([row[1], row[2]])

    ICP3574 = np.array(ICP3574).astype(np.float64)
    ICP3575 = np.array(ICP3575).astype(np.float64)

    ### PART A - PHOTOMOD ###
    image3574.ComputeInnerOrientation(fiducialsImg3574)
    image3575.ComputeInnerOrientation(fiducialsImg3575)
    image3574.ComputeExteriorOrientation(ICP3574, GCP, 1e-6)
    image3575.ComputeExteriorOrientation(ICP3575, GCP, 1e-6)

    fids3574 = []
    fids3575 = []

    for row in samples[2]:
        if row[0] == '3574':
            fids3574.append([row[1],row[2]])
        else:
            fids3575.append([row[1],row[2]])

    fids3574 = np.array(fids3574).astype(np.float64)
    fids3575 = np.array(fids3575).astype(np.float64)

    camPoints3574 = image3574.ImageToCamera(fids3574)
    camPoints3575 = image3575.ImageToCamera(fids3575)



    ### PART B - PYTHON ###

    imagePair1 = ip.ImagePair(image3574, image3575)

    gPoints, acc = imagePair1.ImagesToGround(fids3574, fids3575, 'vector')

    imgPoints3574 = image3574.GroundToImage(gPoints)
    imgPoints3575 = image3575.GroundToImage(gPoints)

    roof_area_world = poly_area(gPoints[:,0], gPoints[:,1])
    roof_area_camera = poly_area(camPoints3574[:,0], camPoints3574[:,1])

