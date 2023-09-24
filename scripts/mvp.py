import os
import cv2
import h5py
import numpy as np 
from pathlib import Path
import open3d as o3d


DATASET_DIR = "/home/bharadwajsirigadi/Documents/Data_preprocess/Datasets/MVP_Datasets"

TYPE = "Completion"
SUB_TYPE = "Test"

class MVPDataset():
    def __init__(self, data_dir: Path, type, sub_type) -> None:
        self.dataset_dir = data_dir
        self.file_path = self.get_file_path(data_dir, type, sub_type)
        complete_pcds, incomplete_pcds, labels = self.load_data(self.file_path)
        # point_cloud_o3d = o3d.geometry.PointCloud()
        # point_cloud_o3d.points = o3d.utility.Vector3dVector(complete_pcds[0])
        # o3d.visualization.draw_geometries([point_cloud_o3d])

    def get_file_path(self, data_dir: Path, type, sub_type):
        if (type == 'completion' or type == 'Completion') and (sub_type == 'test' or sub_type == 'Test'):
            hdf5_file = 'MVP_Test_CP.h5'
        elif (type == 'completion' or type == 'Completion') and (sub_type == 'train'or sub_type == 'Train'):
            hdf5_file = 'MVP_Train_CP.h5'
        elif (type == 'registration' or type == 'Registration') and (sub_type == 'test' or sub_type == 'Test'):
            hdf5_file = 'MVP_Test_RG.h5'
        elif (type == 'registration' or type == 'Registration') and (sub_type == 'train' or sub_type == 'Train'):
            hdf5_file = 'MVP_Train_RG.h5'
        else:
            print("Invalid Type!")
            exit()
        hdf5_file_path = os.path.join(data_dir, hdf5_file)
        return hdf5_file_path
    
    def load_data(self, file_path:Path):
        file = h5py.File(file_path,'r+')
        keys_list = list(file.keys())
        complete_pcds = file[keys_list[0]][:]
        incomplete_pcds = file[keys_list[1]][:]
        labels = file[keys_list[2]][:]
        return complete_pcds, incomplete_pcds, labels
    
a = MVPDataset(DATASET_DIR, TYPE, SUB_TYPE)