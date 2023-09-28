import os
import cv2
import h5py
import numpy as np 
from pathlib import Path
import open3d as o3d

# DATASET_DIR = " "

def extract_vertices_from_obj(file_path):
    vertices = []
    with open(file_path, 'r') as file:
        obj_data = file.readlines()

    for line in obj_data:
        if line.startswith('v '):
            parts = line.split()
            x, y, z = map(float, parts[1:4])
            vertices.append([x, y, z])

    return vertices

# Usage example:
# file_path = '/home/bharadwajsirigadi/Documents/Data_preprocess/Datasets/ShapeNet_Datasets/02691156/1a04e3eab45ca15dd86060f189eb133/models/model_normalized.obj'
file_path = "/home/bharadwajsirigadi/Documents/Data_preprocess/Datasets/ShapeNet_Datasets/02691156/1a6ad7a24bb89733f412783097373bdc/models/model_normalized.obj"
vertices = extract_vertices_from_obj(file_path)
point_cloud_o3d = o3d.geometry.PointCloud()
point_cloud_o3d.points = o3d.utility.Vector3dVector(vertices) 
o3d.visualization.draw_geometries([point_cloud_o3d])
# print(vertices)