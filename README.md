# Data2PointCloud

### Supported Datasets
1. 3DMatch
2. MVP
3. ShapeNet

#### 3DMatch File Structure
```
├── Dataset Directory
│   ├── seq-01
│   │   ├── frame-000000.color.png
|   |   ├── frame-000000.depth.png
|   |   ├── frame-000000.pose.txt
|   |   |        |
│   ├── seq-02
│   │   ├── frame-000000.color.png
|   |   ├── frame-000000.depth.png
|   |   ├── frame-000000.pose.txt
|   |   |        |
├── ├── camera-intrinsics.txt
```
#### Dataset Structure after pre-processing
```
├── Dataset Directory
│   ├── point_cloud_data
│   │   ├── frame-00.npy
│   │   ├── frame-01.npy
|   |   |        |
│   ├── seq-01
│   │   ├── file_depth_images.txt
│   │   ├── file_rgb_images.txt
│   │   ├── frame-000000.color.png
|   |   ├── frame-000000.depth.png
|   |   ├── frame-000000.pose.txt
|   |   |        |
│   ├── seq-02
│   │   ├── file_depth_images.txt
│   │   ├── file_rgb_images.txt
│   │   ├── frame-000000.color.png
|   |   ├── frame-000000.depth.png
|   |   ├── frame-000000.pose.txt
|   |   |        |
├── ├── camera-intrinsics.txt
```

#### MVP File Structure
```
├── Dataset Directory
│   ├── MVP_Test_CP.h5
│   ├── MVP_Train_CP.h5
```
#### Dataset Structure after pre-processing
```
├── Dataset Directory
│   ├── point_cloud_data
│   │   ├── frame-00.npy
│   │   ├── frame-01.npy
|   |   |        |
│   ├── MVP_Test_CP.h5
│   ├── MVP_Train_CP.h5
```

#### ShapeNet File Structure
```
├── Dataset Directory
│   ├── <synsetId> 
│   │   ├── <modelId>
|   |   |   ├── models
|   |   |   |   ├── model_normalized.json
|   |   |   |   ├── model_normalized.obj
|   |   |   |   ├── model_normalized.mtl
|   |   |   |   ├── model_normalized.solid.binvox
|   |   |   |   ├── model_normalized.surface.binvox
|   |   |   ├── images
|   |   |   |   ├── jpg, png
|   |   |   ├── screenshots
|   |   |   |   ├── png
```
#### Dataset Structure after pre-processing

```
├── Dataset Directory
│   ├── <synsetId> 
│   │   ├── <modelId>
│   │   ├── point_cloud_data
│   │   |   ├── frame-00.npy
│   │   |   ├── frame-01.npy
```