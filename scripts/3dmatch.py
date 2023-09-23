import os
import cv2
import numpy as np 
from pathlib import Path
import open3d as o3d

DATASET_FOLDER_PATH = "/home/bharadwajsirigadi/Documents/Data_preprocess/custom_dataset"
SEQUENCE = 1
# IMG_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, SEQUENCE)
RGB_EXTENSION = ".color.png"
DEPTH_EXTENSION = ".depth.png"
EXTRINSIC_EXTENSION = ".pose.txt"

class ThreeDMatchDataset():
    def __init__(self, data_dir: Path, sequence: int, *_, **__):
        self.sequence_id = os.path.basename(data_dir)
        # print("sequence_ID:", self.sequence_id)
        print(str(f"seq-{sequence:02d}"))
        self.sequence_dir = os.path.join(data_dir, str(f"seq-{sequence:02d}"))

        self.dataset_dir = data_dir
        self.intrinsic_matrix = np.loadtxt(os.path.join(self.dataset_dir, "camera-intrinsics.txt"))
        self.points, self.colors = self.load_data()
    
    def get_files(self, extension):
        files = os.listdir(self.sequence_dir)
        filtered_files = []
        for file in files:
            if file.startswith("frame-") and file.endswith(extension):
                filtered_files.append(file)
        filtered_files.sort()
        return filtered_files
    
    def read_images(self, image_files, conversion_code):
        image_list = []
        for image_file in image_files:
            image_path = os.path.join(self.sequence_dir, image_file)
            print(image_path)
            image = cv2.imread(image_path, conversion_code)
            # print(image)
            image_list.append(image)
        return np.array(image_list)
    
    def read_extrinsic(self, extrinsic_files):
        extrinsic_list = []
        for extrinsic_file in extrinsic_files:
            extrinsic_path = os.path.join(self.sequence_dir, extrinsic_file)
            extrinsic_matrix = np.loadtxt(extrinsic_path)
            # print(image)
            extrinsic_list.append(extrinsic_matrix)
        return np.array(extrinsic_list)
      
    def get_points(self, RGB_img_list, depth_img_list, extrinsic_list):
        points = []
        colors = []
        first_image = depth_img_list[0]
        height, width = first_image.shape
        num_images = len(RGB_img_list)
        for i in range(num_images):
            rgb_img = RGB_img_list[i]
            # print("rgb_image", rgb_img)
            depth_img = depth_img_list[i]
            # print("depth_img ", depth_img)
            extrinsic_matrix = extrinsic_list[i]
            height, width = depth_img.shape
            points_1 = []
            colors_1 = []
            for v in range(height):
                for u in range(width):
                    depth = depth_img[v, u]
                    # print(depth)
                    if depth > 0:
                        # Convert 2D pixel coordinates to 3D world coordinates
                        projection_mtx = self.intrinsic_matrix @ extrinsic_matrix[:3, :] # 3*4 matrix
                        homo_projection_mtx = np.vstack([projection_mtx, [0, 0, 0, 1]])
                        point_3d = np.dot(np.linalg.inv(homo_projection_mtx), np.array([u* depth, v* depth, depth, 1]) )
                        point_3d = point_3d[:3] / point_3d[3]
                        points.append(point_3d)
                        # points_1.append(point_3d)
                        color = rgb_img[v, u]
                        colors_1.append(color) 
            # print(i)        
            points.append(points_1)
            colors.append(colors_1)
        return points, colors
    
    def load_data(self):
        # Reading RGB Images
        RGB_img_files = self.get_files(RGB_EXTENSION)
        RGB_img_list = self.read_images(RGB_img_files, cv2.IMREAD_COLOR)
        # print(RGB_img_list)
        # # Reading Depth Images
        depth_img_files = self.get_files(DEPTH_EXTENSION)
        depth_img_list = self.read_images(depth_img_files, cv2.IMREAD_UNCHANGED)
        # Reading Pose Information(extrinsic files)
        extrinsic_text_files = self.get_files(EXTRINSIC_EXTENSION)
        extrinsic_list = self.read_extrinsic(extrinsic_text_files)
        # print(extrinsic_list)
        points, colors = self.get_points(RGB_img_list, depth_img_list, extrinsic_list)   
        # data_dict = {
        #     "points": points,
        #     "colors": colors
        # }

        # Save data to a .npy file
        # np.save(os.path.join(self.dataset_dir, "data.npy"), data_dict) 
        return points, colors    
    
    def __getitem__(self, idx):
        return self.points[0][idx], self.colors[0][idx]
    
    def __len__(self):
        return len(self.points)
    

    
def main():
    a = ThreeDMatchDataset(DATASET_FOLDER_PATH, SEQUENCE)
    print(a)
    # points, clouds = a.__getitem__(0)
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points.append(points)
    # o3d.visualization.draw_geometries([point_cloud])   
    # points, colors = a.load_data()
    # print(points)
    return

if __name__ == "__main__":
    main()
    