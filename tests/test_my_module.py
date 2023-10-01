# -*- coding: utf-8 -*-
# @Author: Sai Bharadwaj Sirigadi
# @Date:   2023-09-30 10:39:16
# @Last Modified by:   Your name
# @Last Modified time: 2023-09-30 17:54:47
# Example test_my_module.py
# test_my_module.py

from Data2PointCloud.src.datasets.three_d_match import ThreeDMatchDataset2

DATASET_FOLDER_PATH = "/Users/bharadwajsirigadi/Documents/Dataset-Preprocess/custom_dataset" #"/home/bharadwajsirigadi/Documents/Data_preprocess/custom_dataset"
SEQUENCE = 1

def test_three_d_match():
    a = ThreeDMatchDataset2(DATASET_FOLDER_PATH, SEQUENCE)
    points = a[2]
    print(points)
    return 
    


