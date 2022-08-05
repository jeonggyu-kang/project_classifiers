import numpy as np
import pydicom
import cv2
import matplotlib.pylab as plt
import os



root_dir = "/media/compu/ssd2t/cxr_all/cxr"
dest_dir = "./data_check"    
rsize = (244,244)
crop_ratio = 0

all_files = []

for (path, dirs, files) in os.walk(root_dir):

    for file in files:
        path_file = path[len(root_dir):] + '/' + file

        if not os.path.exists(dest_dir + "/" + path[len(root_dir):]):
            os.makedirs(dest_dir + "/" + path[len(root_dir):])
        if path_file[-3:] == "dcm":
            all_files.append(path_file)

for file in all_files:
    print (file)    
   
    dco = pydicom.dcmread(root_dir + file)       # dco: dicom_content
    arr = dco.pixel_array

    arr = arr / (2 ** (dco.BitsStored - 8))

    if(('RescaleSlope' in dco) and ('RescaleIntercept' in dco)):
        arr = (arr * dco.RescaleSlope) + dco.RescaleIntercept
    resized_arr = cv2.resize(arr, dsize=rsize)

    cv2.imwrite(dest_dir + "/" + file[1:-4] + ".jpg", resized_arr)
    # np.save(dest_dir + "/" + file[1:-4], resized_arr)
