import numpy as np
import pydicom
import cv2
import matplotlib.pylab as plt
import os
from pathlib import Path
import pandas as pd 
from PIL import Image

import pydicom

from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut


root_dir = "/media/compu/ssd2t/cxr_all/cxr"
dest_dir = "./data/image_samples"    

rsize = (896,896)
crop_ratio = 0.15

sample_rate = 0.01

all_file_size = []

for (path, dirs, files) in os.walk(root_dir):

    for file in files:
        path_file = root_dir + path[len(root_dir):] + '/' + file


        if path_file[-3:] == "dcm":
            size = Path(path_file).stat().st_size
            all_file_size.append((path_file, size))


df = pd.DataFrame(all_file_size, columns=['file', 'size'])

print (len(df))

df = df.sort_values('size')

df1 = df.loc[0:20, 'file']
df2 = df.sample(n=round(len(df)*sample_rate)).loc[:, 'file']



df_sample = pd.concat([df1,df2], ignore_index=True)
df_sample.drop_duplicates()
print (len(df_sample))

all_files = df_sample.values.tolist()

sett_all = []

for file in all_files:
  
    dco = pydicom.dcmread(file)       # dco: dicom_content
    arr = dco.pixel_array

    arr = arr[round(arr.shape[0]*crop_ratio):arr.shape[0]- round(arr.shape[0]*crop_ratio),
                round(arr.shape[1]*crop_ratio):arr.shape[1]- round(arr.shape[1]*crop_ratio)]     
    # print (arr.shape)
    diff = arr.shape[0] - arr.shape[1]

    if diff > 0:    
        if round(diff/2) * 2 != diff:
            arr = np.delete(arr, arr.shape[0]-1, axis = 0)
            arr = np.delete(arr, range(0, int((diff/2) - 0.5)), axis = 0)
            arr = np.delete(arr, range(arr.shape[0] - int((diff/2) - 0.5), arr.shape[0]), axis = 0)
        else:
            arr = np.delete(arr, range(0, int(diff/2)), axis = 0)
            arr = np.delete(arr, range(arr.shape[0] - int(diff/2), arr.shape[0]), axis = 0)

    elif diff < 0:
        diff = (-diff) 
        if round(diff/2) * 2 != diff:
            arr = np.delete(arr, arr.shape[1]-1, axis = 1)
            arr = np.delete(arr, range(0, int((diff/2) - 0.5)), axis = 1)
            arr = np.delete(arr, range(arr.shape[1] - int((diff/2) - 0.5), arr.shape[1]), axis = 1)
        else:
            arr = np.delete(arr, range(0, int(diff/2)), axis = 1)
            arr = np.delete(arr, range(arr.shape[1] - int(diff/2), arr.shape[1]), axis = 1)

    bits = dco.BitsStored

    window_center = dco.WindowCenter
    window_width = dco.WindowWidth


    # m = np.max(arr[int(round(arr.shape[0]*0.3)) :- int(round(arr.shape[0]*0.3)), : ])
    
    # n = np.min(arr)

    l = window_width/256

    # sett_all.append((file, m, n))

    arr = arr - (window_center - window_width/2)
    
    arr = np.divide(arr, l)

    resized_arr = cv2.resize(arr, dsize=rsize)


    resized_arr = np.trunc(resized_arr)
    resized_arr = np.asarray(resized_arr, dtype = int)

    img_2 = Image.fromarray(resized_arr.astype(np.uint8)) # NumPy array to PIL image
   
   
    img_2.save(dest_dir + "/" + file[len(root_dir)+1:-4].replace("/", "_") + '.png','png')


