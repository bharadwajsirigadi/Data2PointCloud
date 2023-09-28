import os
import numpy as np 
from pathlib import Path
import open3d as o3d

DATASET_DIR = "/home/bharadwajsirigadi/Documents/Data_preprocess/Datasets/ShapeNet_Datasets/02691156"

DIRECTORY = "point_cloud_data"
FILE_NAME = str()

class ShapeNetDataset():
    def __init__(self, data_dir:Path, *_, **__) -> None:
        self.dataset_dir = data_dir
        self.file_name = str()
        return

    def extract_vertices_from_obj(self, file_path):
        vertices = []
        with open(file_path, 'r') as file:
            obj_data = file.readlines()
        for line in obj_data:
            if line.startswith('v '):
                parts = line.split()
                x, y, z = map(float, parts[1:4])
                vertices.append([x, y, z])
        return vertices

    def get_files(self, dir:Path):
        files = os.listdir(dir)
        filtered_files = []
        for file in files:
            path = os.path.join(dir, file, "models/model_normalized.obj")
            filtered_files.append(path)
        return filtered_files

    def write_text(self, dir:Path, files_list, file_name):
        file_name = f"file_{file_name}.txt"
        self.file_name = file_name
        file_path = os.path.join(dir, file_name)
        try:
            with open(file_path, 'w') as file:
                for item in files_list:
                    file.write(f"{item}\n")
            print(f"File '{file_name}' has been created and written to '{file_path}'.")
        except Exception as e:
            print(f"An error occurred: {e}")
        return
    
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
    
    def extract_data(self):
        file_dirs = self.get_files(DATASET_DIR)
        self.write_text(DATASET_DIR, file_dirs, "directories")
        directories_file = open(os.path.join(DATASET_DIR, self.file_name))
        directories = directories_file.readlines()
        directories_file.close()
        counter = 0
        for file in directories:
            vertices = self.extract_vertices_from_obj(file.strip())
            file_name = str(f"frame-{counter:02d}")
            self.save_numpy_array(vertices, file_name)
            print(f"Generated point cloud for image {counter+1}/{len(directories)}")
            counter += 1
        return
            
c = ShapeNetDataset(DATASET_DIR)
c.extract_data()

# filtered_files = get_files(DATASET_DIR)
# write_text(filtered_files, "directories")

# print(files)
# point_cloud_o3d = o3d.geometry.PointCloud()
# point_cloud_o3d.points = o3d.utility.Vector3dVector(vertices) 
# o3d.visualization.draw_geometries([point_cloud_o3d])
# print(vertices)