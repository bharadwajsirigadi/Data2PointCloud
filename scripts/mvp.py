import os
import cv2
import h5py
import numpy as np 
from pathlib import Path
import open3d as o3d


DATASET_DIR = "/home/bharadwajsirigadi/Documents/Data_preprocess/Datasets/MVP_Datasets"

TYPE = "Completion"
SUB_TYPE = "Test"

if (TYPE == 'completion' or TYPE == 'Completion') and (SUB_TYPE == 'test' or SUB_TYPE == 'Test'):
    hdf5_file = 'MVP_Test_CP.h5'
elif (TYPE == 'completion' or TYPE == 'Completion') and (SUB_TYPE == 'train'or SUB_TYPE == 'Train'):
    hdf5_file = 'MVP_Train_CP.h5'
elif (TYPE == 'registration' or TYPE == 'Registration') and (SUB_TYPE == 'test' or SUB_TYPE == 'Test'):
    hdf5_file = 'MVP_Test_RG.h5'
elif (TYPE == 'registration' or TYPE == 'Registration') and (SUB_TYPE == 'train' or SUB_TYPE == 'Train'):
    hdf5_file = 'MVP_Train_RG.h5'
else:
    print("Invalid Type!")
    exit()
print(hdf5_file)

hdf5_file_path = os.path.join(DATASET_DIR, hdf5_file)


# reading hdf5 file

file = h5py.File(hdf5_file_path,'r+')
print("Keys in the HDF5 file:", list(file.keys()))


dataset = file[list(file.keys())[1]]
print(dataset[0].shape)
point_cloud_o3d = o3d.geometry.PointCloud()
point_cloud_o3d.points = o3d.utility.Vector3dVector(dataset[100])
o3d.visualization.draw_geometries([point_cloud_o3d])
