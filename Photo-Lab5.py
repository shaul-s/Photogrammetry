import reader as rd

cam = rd.Reader.ReadCamFile(r'C:\Users\Saul\Desktop\lab5-shaul&ariel\PythonFiles\PythonFiles\rc30.cam')
fiducialsImg = rd.Reader.Readtxtfile(r'C:\Users\Saul\Desktop\lab5-shaul&ariel\PythonFiles\PythonFiles\fiducialsImg.txt')

print(cam,'\n')
print(fiducialsImg)