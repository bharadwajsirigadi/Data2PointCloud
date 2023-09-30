import os
import cv2
import numpy as np 
from pathlib import Path
import open3d as o3d

# Specific to the requirement
DATASET_FOLDER_PATH = "/home/bharadwajsirigadi/Documents/Data_preprocess/Datasets/3DMatch_Datasets/custom_dataset" #"/home/bharadwajsirigadi/Documents/Data_preprocess/custom_dataset"
SEQUENCE = 1

# Default
RGB_EXTENSION = ".color.png"
DEPTH_EXTENSION = ".depth.png"
EXTRINSIC_EXTENSION = ".pose.txt"
POINT_DATA_EXTENSION = ".npy"
DIRECTORY = "point_cloud_data"

class ThreeDMatchDataset2(): 
    def __init__(self, data_dir: Path, sequence: int, *_, **__) -> None:
        self.data_loaded = False
        self.sequence_id = os.path.basename(data_dir)
        # print("sequence_ID:", self.sequence_id)
        print(str(f"seq-{sequence:02d}"))
        self.dataset_dir = data_dir
        self.sequence_dir = os.path.join(data_dir, str(f"seq-{sequence:02d}"))
        self.intrinsic_matrix = np.loadtxt(os.path.join(self.dataset_dir, "camera-intrinsics.txt"))
        # self.extract_data()

    def get_files(self, extension, dir:Path):
        files = os.listdir(dir)
        filtered_files = []
        for file in files:
            if file.startswith("frame-") and file.endswith(extension):
                filtered_files.append(file)
        filtered_files.sort()
        return filtered_files
    
    def read_image(self, file_name, conversion_code):
        image_path = os.path.join(self.sequence_dir, file_name)
        image = cv2.imread(image_path, conversion_code)
        return np.array(image)
    
    def write_text(self, files_list, file_name):
        file_name = f"file_{file_name}.txt"
        file_path = os.path.join(self.sequence_dir, file_name)
        try:
            with open(file_path, 'w') as file:
                for item in files_list:
                    file.write(f"{item}\n")
            print(f"File '{file_name}' has been created and written to '{file_path}'.")
        except Exception as e:
            print(f"An error occurred: {e}")
        return
    
    def read_extrinsic(self, extrinsic_files):
        extrinsic_list = []
        for extrinsic_file in extrinsic_files:
            extrinsic_path = os.path.join(self.sequence_dir, extrinsic_file)
            extrinsic_matrix = np.loadtxt(extrinsic_path)
            # print(image)
            extrinsic_list.append(extrinsic_matrix)
        return np.array(extrinsic_list)
    
    def get_points(self, rgb_img, depth_img, extrinsic_matrix):
        points = np.zeros((rgb_img.shape[0], rgb_img.shape[1], 6), dtype=np.float32)  # (X, Y, Z, R, G, B)
        point_index = 0
        height, width = depth_img.shape
        for i in range(height):
            for j in range(width):
                # Extract RGB and depth values for the current pixel
                rgb = rgb_img[i, j]
                depth = depth_img[i, j]
                if depth > 0:
                        # Convert 2D pixel coordinates to 3D world coordinates
                        projection_mtx = self.intrinsic_matrix @ extrinsic_matrix[:3, :] # 3*4 matrix
                        homo_projection_mtx = np.vstack([projection_mtx, [0, 0, 0, 1]])
                        point = np.dot(np.linalg.inv(homo_projection_mtx), np.array([j* depth, i* depth, depth, 1]) )
                        point_3d = list(point[:3] / point[3])
                        # point_3d.extend(rgb)
                        # points[point_index, 0:6] = point_3d
                        points[i, j, 0:3] = point_3d
                        points[i, j, 3:6] = rgb
                        point_index += 1
        return points
    
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
        RGB_img_files = self.get_files(RGB_EXTENSION, self.sequence_dir)
        self.write_text(RGB_img_files, "rgb_images")
        # print(f"Sorted & Retrieved RGB-Image Files")

        depth_img_files = self.get_files(DEPTH_EXTENSION, self.sequence_dir)
        self.write_text(depth_img_files, "depth_images")
        # print(f"Sorted & Retrieved Depth-Image Files")
        self.data_loaded = True
        extrinsic_text_files = self.get_files(EXTRINSIC_EXTENSION, self.sequence_dir)
        extrinsic_list = self.read_extrinsic(extrinsic_text_files)

        rgb_file = open(os.path.join(self.sequence_dir, "file_rgb_images.txt"))
        depth_file = open(os.path.join(self.sequence_dir, "file_depth_images.txt"))
        rgb_file_names = rgb_file.readlines()
        depth_file_names = depth_file.readlines()
        rgb_file.close()
        depth_file.close()
        counter = 0
        for rgb_file_name, depth_file_name, extrinsic_matrix in zip(rgb_file_names, depth_file_names, extrinsic_list):
            rgb_img_array = self.read_image(rgb_file_name.strip(), cv2.IMREAD_COLOR)
            depth_img_array = self.read_image(depth_file_name.strip(), cv2.IMREAD_UNCHANGED)
            points = self.get_points(rgb_img_array, depth_img_array, extrinsic_matrix)
            point_cloud_o3d = o3d.geometry.PointCloud()
            point_positions = np.array(points[:, :, :3], dtype=np.float64).reshape(-1, 3)
            point_colors = np.array(points[:, :, 3:], dtype=np.uint8).reshape(-1, 3)
            point_cloud_o3d.points = o3d.utility.Vector3dVector(point_positions) 
            point_cloud_o3d.colors = o3d.utility.Vector3dVector(point_colors/ 255.0)
            np.save("file", points, allow_pickle=True, fix_imports=True)
            file_name = str(f"frame-{counter:02d}")
            self.save_numpy_array(points, file_name)
            print(f"Generated point cloud for image {counter+1}/{len(rgb_file_names)}")
            # o3d.visualization.draw_geometries([point_cloud_o3d])
            counter += 1
        return
    
    def __getitem__(self, idx):
        data_path = os.path.join(self.dataset_dir, DIRECTORY)
        if os.path.exists(data_path):
            files = self.get_files(POINT_DATA_EXTENSION, data_path)
            data_file_path = os.path.join(data_path, files[idx].strip())
            data = np.load(data_file_path)
        else:
            print(f"Data isn't extracted yet!")
            print(f"Extracting Data")
            self.extract_data()
            files = self.get_files(POINT_DATA_EXTENSION, data_path)
            data_file_path = os.path.join(data_path, files[idx].strip())
            data = np.load(data_file_path)
        return data
    
    def __len__(self) -> int:
        rgb_file_path = os.path.join(self.sequence_dir, "file_rgb_images.txt")
        if os.path.exists(rgb_file_path):
            rgb_file = open(rgb_file_path)
            rgb_file_names = rgb_file.readlines()
        else:
            print(f"Data isn't extracted yet!")
            print(f"Extracting Data")
            self.extract_data()
            rgb_file = open(rgb_file_path)
            rgb_file_names = rgb_file.readlines()
        return len(rgb_file_names)
        
def main():
    a = ThreeDMatchDataset2(DATASET_FOLDER_PATH, SEQUENCE)
    # points = a[2]
    # point_cloud_o3d = o3d.geometry.PointCloud()
    # point_positions = np.array(points[:, :, :3], dtype=np.float64).reshape(-1, 3)
    # point_colors = np.array(points[:, :, 3:], dtype=np.uint8).reshape(-1, 3)
    # point_cloud_o3d.points = o3d.utility.Vector3dVector(point_positions) 
    # point_cloud_o3d.colors = o3d.utility.Vector3dVector(point_colors/ 255.0)
    # o3d.visualization.draw_geometries([point_cloud_o3d])
    return

if __name__ == "__main__":
    main()