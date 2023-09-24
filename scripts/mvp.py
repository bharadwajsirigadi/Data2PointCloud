import os
import cv2
import h5py
import numpy as np 
from pathlib import Path
import open3d as o3d

# Specific to the requirement
DATASET_DIR = "/home/bharadwajsirigadi/Documents/Data_preprocess/Datasets/MVP_Datasets"
TYPE = "Completion"
SUB_TYPE = "Test"

# Default
DIRECTORY = "point_cloud_data"

class MVPDataset():
    def __init__(self, data_dir: Path, type, sub_type) -> None:
        self.dataset_dir = data_dir
        self.file_path = self.get_file_path(data_dir, type, sub_type)
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
    
    def save_numpy_array(self, points, file_name):
        path = os.path.join(self.dataset_dir, DIRECTORY)
        if os.path.exists(path):
            file_path = os.path.join(path, file_name)
            np.save(file_path, points, allow_pickle=True, fix_imports=True)
        else:
            os.mkdir(path)
            file_path = os.path.join(path, file_name)
            np.save(file_path, points, allow_pickle=True, fix_imports=True)
        return
    
    def get_points(self, complete_pcd, incomplete_pcd):
        points = np.hstack((complete_pcd, incomplete_pcd))
        return points
    
    def extract_data(self):
        file = h5py.File(self.file_path,'r+')
        keys_list = list(file.keys())
        complete_pcds = file[keys_list[0]][:]
        incomplete_pcds = file[keys_list[1]][:]
        labels = file[keys_list[2]][:]
        # points = self.get_points(complete_pcds[0], incomplete_pcds[0])
        # print(points.shape)
        # point_cloud_o3d = o3d.geometry.PointCloud()
        # point_positions = np.array(points[:, 3:], dtype=np.float64)
        # point_cloud_o3d.points = o3d.utility.Vector3dVector(point_positions) 
        # o3d.visualization.draw_geometries([point_cloud_o3d])
        pc_length = len(complete_pcds)
        counter = 0
        for i, j, k in zip(complete_pcds, incomplete_pcds, labels):
            points = self.get_points(i, j)
            file_name = f"frame-{counter:02d}-{k}.txt"
            self.save_numpy_array(points, file_name)
            print(f"Generated point cloud for image {counter+1}/{pc_length}")
            counter += 1
        return

def main():
    dataset = MVPDataset(DATASET_DIR, TYPE, SUB_TYPE)
    dataset.extract_data()
    return

if __name__ == "__main__":
    main()