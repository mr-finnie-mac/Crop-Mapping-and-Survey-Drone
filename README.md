# Crop-Mapping-and-Survey-Drone
## Description
Repository for Farming drone project - building on previous work with drones. See [meadeor-drone](https://github.com/mr-finnie-mac/meadeor-drone) (Fire rescue drone) and [corona-drone](https://github.com/mr-finnie-mac/corona-drone) (test/vaccine transport drone)

### Stereo camera
Stereo camera rig using two Coral cameras and 3d printed frame. with a custom built library for handling synced image capture and video capture. Python based library that uses gstreamer and opencv to manage image caputure and saving. Uses threading to reduce delay between left/right image pairs. Calibration, undistoriton and rectification all calculated using opencv and matlab.

### Depth sensing
Using opencv to make depth maps from stereo images using SGBM and BM. Alternatively, MiDaS and Keras ML models also used for preditcting depth for when using stereo camera as a monocular system. 

### Pointclouds
Converted from depth imagery. Used RGBD data structures to make 3D meshes, voxel grids and pointcloud analysis also done using open3d.
