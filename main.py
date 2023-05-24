# Include configuration header
from config import *

# Include necessary libraries
import cv2 as cv
import torch
import matplotlib.pyplot as plt
import threading
import numpy as np
import time
import open3d as o3d
import glob
import os
# Import camera hardware interfacing module
from camera import (Camera, getCurrentMS, gstreamer_pipeline, testAllFeatures, gstreamer_pipeline, 
                    runMonoCamera, recordMonoVideo, captureMono, monoCalibrationSequence, 
                    runStereoCamera, recordStereoVideo, captureStereo, stereoCalibrationSequence, capture_stereo_frame)

# Import depth map module
from depthmap import (load_midas_model, compute_depth_map, show_depth_map, compare_depth_map,
                      stereo_depthmap_compute, compute_stereo_mission_folder_to_image_and_depth,
                      intrinsic_stereo_depthmap_compute, blender_specific_depthmap)

#Import pointcloud module
from pointcloud import (make_rgbd, convert_to_ply, observe_pointcloud, convert_ply_to_voxel, 
                        condition_pointcloud, display_inlier_outlier, downsample_pointcloud, 
                        radial_outlier_inlier, evaluate_ICP, draw_registration_result, 
                        point_to_point_ICP, plot_images, plot_rgbd_image_images,
                        make_rgbd_from_np, export_pointcloud, generate_rgbd_pack_from_mission,
                        generate_pointclouds_from_rgbd_list, pointcloud_clustering_basic, 
                        pointcloud_clustering_segmentation, visualise_segmented_pointcloud_list,
                        cluster_statistical_outlier_removal, process_pointcloud, open_ply, analyse_cluster)

# Import calibration module
from calibration import make_calibration_params, make_new_calibration_params

def convert_folder_to_depthmaps(pathToFolder):
    """Convert a folder of images to depth maps"""

    # load model
    load_model =  load_midas_model(MIDAS_MODEL_TYPE)
    transform = load_model[0]
    midas = load_model[1]
    device = load_model[2]

    # make new folder for depthmaps
    images = glob.glob(pathToFolder + '*.jpeg', recursive = False)
    depthmapFolder = pathToFolder + "/depthmaps/"
    os.makedirs(depthmapFolder, exist_ok=True)
    # go through all images in folder and make depthmaps for them, save to new folder
    for image in images:
        filename_ext = os.path.basename(image)
        filename = filename_ext.replace('.jpeg','')
        print("Generating depth map of %s"%(filename_ext))
        img = cv.imread(image)
        depth_map = compute_depth_map(transform, midas, device, img, depthmapFolder+filename+MAP_EXT)



def stereo_to_pointcloud_demo(left_camera=Camera(LEFT_ID, LEFT_DEVICE_ADDRESS, LEFT_WIDTH, LEFT_HEIGHT, LEFT_FRAMERATE, IMAGE_DESTINATION), 
                    right_camera=Camera(RIGHT_ID, RIGHT_DEVICE_ADDRESS, RIGHT_WIDTH, RIGHT_HEIGHT, RIGHT_FRAMERATE, IMAGE_DESTINATION)):
    """Take a stereo pair of images and convert them to pointclouds via master prep-pipeline"""

    # leftRGB, rightRGB = capture_stereo_frame(left_camera, right_camera)
    leftRGB = cv.imread('./active_data/images/guitar_L.jpeg')
    rightRGB = cv.imread('./active_data/images/guitar_R.jpeg')

    disparity, filteredDisparity = stereo_depthmap_compute(leftImage=leftRGB, rightImage=rightRGB)
    rgbd = make_rgbd_from_np(leftRGB, filteredDisparity)
    
    pcd = convert_to_ply(rgbd)
    pcd = condition_pointcloud(pcd)
    refined_pcd = downsample_pointcloud(pcd)
    refined_pcd = radial_outlier_inlier(refined_pcd, remove=True)

    observe_pointcloud(refined_pcd)
    export_pointcloud(refined_pcd)

def map_environment(missionCode=getCurrentMS(),
                    left_camera=Camera(LEFT_ID, LEFT_DEVICE_ADDRESS, LEFT_WIDTH, LEFT_HEIGHT, LEFT_FRAMERATE, IMAGE_DESTINATION), 
                    right_camera=Camera(RIGHT_ID, RIGHT_DEVICE_ADDRESS, RIGHT_WIDTH, RIGHT_HEIGHT, RIGHT_FRAMERATE, IMAGE_DESTINATION)):


    image_list, depth_list = compute_stereo_mission_folder_to_image_and_depth(missionCode=missionCode, path="./active_data/", sgbm_preset=0, wls=True)
    
    
    rgbd_list = generate_rgbd_pack_from_mission(missionCode=missionCode, path="./active_data/")#, image_list=image_list, depth_list=depth_list)
    # print(type(rgbd_list[0][0]))
    # print(rgbd_list[1][0])
    # print(rgbd_list[2][0])
    
    # generate pointcloud list (as voxels)
    pcd_list = generate_pointclouds_from_rgbd_list(rgbd_list=rgbd_list, use_voxel=False)

    o3d.visualization.draw_geometries(pcd_list)
    o3d.visualization.draw_geometries([pcd_list[10]])
    test_pcd = pcd_list[10]
    print(type(test_pcd))
    test_pcd.remove_non_finite_points()
    # observe_pointcloud(test_pcd)
    # observe_pointcloud(pcd_list[0])
    # test_pcd = radial_outlier_inlier(test_pcd, remove=True)
    # display_inlier_outlier(test_pcd, ind)
    # observe_pointcloud(test_pcd)


    # print("----------FLAG")
    # pcd = convert_to_ply(rgbd_list[0][1])
    # pcd = condition_pointcloud(pcd)
    # refined_pcd = downsample_pointcloud(pcd)
    # refined_pcd = radial_outlier_inlier(refined_pcd, remove=True)
    # observe_pointcloud(refined_pcd)
    # pointcloud_clustering_basic(test_pcd)
    # seg_pcd = pointcloud_clustering_segmentation(test_pcd)
    # visualise_segmented_pointcloud_list(seg_pcd)

    # trying to normalise values
    # pcd_coords = np.asarray(test_pcd.points)
    # pcd_coords_norm = np.linalg.norm(pcd_coords, axis=1,keepdims=True)
    # pcd_coords_norm = pcd_coords / pcd_coords_norm.max()

    # pcd_norm = o3d.geometry.PointCloud()
    # # pcd_norm = o3d.utility.Vector3dVector(pcd_coords_norm)
    # o3d.visualization.draw_geometries([pcd_norm])
    process_pointcloud(test_pcd)

    # pcd2 = convert_to_ply(rgbd_list[2][1])
    # pcd2 = condition_pointcloud(pcd2)
    # refined_pcd2 = downsample_pointcloud(pcd2)
    # # refined_pcd = radial_outlier_inlier(refined_pcd, remove=True)
    
    # observe_pointcloud(refined_pcd2)

    # point_to_point_ICP(refined_pcd, refined_pcd2)
    


    # # leftRGB, rightRGB = capture_stereo_frame(left_camera, right_camera)
    # leftRGB = cv.imread('./active_data/images/guitar_L.jpeg')
    # rightRGB = cv.imread('./active_data/images/guitar_R.jpeg')

    # disparity, filteredDisparity = stereo_depthmap_compute(leftImage=leftRGB, rightImage=rightRGB)
    # rgbd = make_rgbd_from_np(leftRGB, filteredDisparity)
    
    # pcd = convert_to_ply(rgbd)
    # pcd = condition_pointcloud(pcd)
    # refined_pcd = downsample_pointcloud(pcd)
    # refined_pcd = radial_outlier_inlier(refined_pcd, remove=True)

    # observe_pointcloud(refined_pcd)
    # export_pointcloud(refined_pcd)

if __name__ == "__main__":    
    # stereo_to_pointcloud_demo()
    # imgL = cv.imread('./blender/images/left.png')
    # imgR = cv.imread('./blender/images/right.png')
    # depth, wslDepth = stereo_depthmap_compute(leftImage = imgL, rightImage = imgR, name="blender_overhead_test")
    # rgbd = make_rgbd_from_np(colourImage = imgL, depthImage=wslDepth)
    # pcd = convert_to_ply(rgbd_image=rgbd)
    # process_pointcloud(pcd)
    # pcd = condition_pointcloud(pcd)
    # pcd = downsample_pointcloud(pcd)
    # observe_pointcloud(pcd)
    # segmented_pcd = pointcloud_clustering_segmentation(pcd)
    # visualise_segmented_pointcloud_list(segmented_pcd)

    # Stereo image to pointcloud pipeline
    # stereo_to_pointcloud_demo()

    # Blender Sample
    # disparity = intrinsic_stereo_depthmap_compute(leftImage='./blender/images/left.png', rightImage='./blender/images/right.png', outputPath="./", name="blender_perp")
    # disparity = blender_specific_depthmap(cv.imread('./blender/images/left.png'), cv.imread('./blender/images/right.png'))
    # rgbd_blender = make_rgbd_from_np(cv.imread('./blender/images/left.png'), disparity)
    # pcd_blender = convert_to_ply(rgbd_image=rgbd_blender)
    # process_pointcloud(pcd_blender)

    # Garden dataset sample
    map_environment("garden")

    # Use mock pointcloud data
    mock_pcd = open_ply(MOCK_DATA_SRC+"alt/blendercrops_alt.ply")
    o3d.visualization.draw_geometries([mock_pcd])
    process_pointcloud(mock_pcd)
    # analyse_cloud(pcd=mock_pcd)

    


    # left_camera = Camera(LEFT_ID, LEFT_DEVICE_ADDRESS, LEFT_WIDTH, LEFT_HEIGHT, LEFT_FRAMERATE, IMAGE_DESTINATION)
    # right_camera = Camera(RIGHT_ID, RIGHT_DEVICE_ADDRESS, RIGHT_WIDTH, RIGHT_HEIGHT, RIGHT_FRAMERATE, IMAGE_DESTINATION)
    # left_camera.cameraInfo()
    # right_camera.cameraInfo()

    # make pointcloud from generated rgbd (via depth map)
    # filename = "rendered_guitar_wls"

    # rgbd_1 = make_rgbd('./active_data/images/guitar_L.jpeg', './active_data/filtered_disparity_image_test8.png')
    # pcd_1 = convert_to_ply(rgbd_1)
    # pcd_1 = condition_pointcloud(pcd_1)
    # # observe_pointcloud(pcd_1)
    # # sownsample pointcloud
    # print("Downsampling pcd")
    # downsampled_pcd_1 = downsample_pointcloud(pcd_1)
    # o3d.visualization.draw_geometries([downsampled_pcd_1])

    # # make mesh from pointcloud
    # alpha = 0.05
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(downsampled_pcd_1, alpha)
    # mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    
    # map_environment()


    # update calibration params
    # make_calibration_params()
    # make_new_calibration_params()

    # make depth map using given image and latest intrinsics
    # model = load_midas_model(MIDAS_MODEL_TYPE)
    # transform = model[0]
    # midas = model[1]
    # device = model[2]

    # filename = "rendered_guitar_wls"
    # # current_image1 = cv.imread(filename=('%s%s%s'%(SRC_IMAGE_DATA_PATH, filename, IMAGE_TYPE)))
    # # depth_map = compute_depth_map(transform, midas, device, current_image1,"./active_data/depth_maps/depth_map_%s_%s_.png"%(TEST_NAME, filename))

    # rgbd_1 = make_rgbd('./active_data/images/guitar_L.jpeg', './active_data/filtered_disparity_image_test8.png')
    # pcd_1 = convert_to_ply(rgbd_1)
    # pcd_1 = condition_pointcloud(pcd_1)
    # # observe_pointcloud(pcd_1)

    # # filename = "bbq_2"
    # # current_image2 = cv.imread(filename=('%s%s%s'%(SRC_IMAGE_DATA_PATH, "bbq_2", IMAGE_TYPE)))
    # # depth_map = compute_depth_map(transform, midas, device, current_image2,"./active_data/depth_maps/depth_map_%s_%s_.png"%(TEST_NAME, filename))

    # # rgbd_2 = make_rgbd('%s%s%s'%(SRC_IMAGE_DATA_PATH, filename, IMAGE_TYPE), "./active_data/depth_maps/depth_map_%s_%s_.png"%(TEST_NAME, filename))
    # # pcd_2 = convert_to_ply(rgbd_2)
    # # pcd_2 = condition_pointcloud(pcd_2)
    # # observe_pointcloud(pcd_2)

    # print("Downsampling pcd")
    # downsampled_pcd_1 = downsample_pointcloud(pcd_1)
    # o3d.visualization.draw_geometries([downsampled_pcd_1])



    # alpha = 0.05
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(downsampled_pcd_1, alpha)
    # mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    # trying better mesh generation
    # radii = [0.1, 0.1, 0.1, 0.1]
    # downsampled_pcd_1.estimate_normals()
    # rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(downsampled_pcd_1, o3d.utility.DoubleVector(radii))
    # o3d.visualization.draw_geometries([downsampled_pcd_1, rec_mesh])

    # print("Removing outliers from downsampled pcd")
    # outlier_pcd = radial_outlier_inlier(downsampled_pcd_1)
    # cl = outlier_pcd[0]
    # ind = outlier_pcd[1]
    # display_inlier_outlier(downsampled_pcd_1, ind)
    # o3d.visualization.draw_geometries([radial_outlier_inlier(downsampled_pcd_1, True)]) # viewing pointcloud with outliers removed

    # print("Downsampling pcd")
    # downsampled_pcd_2 = downsample_pointcloud(pcd_2)
    # o3d.visualization.draw_geometries([downsampled_pcd_2])
    # print("Removing outliers from downsampled pcd")
    # outlier_pcd = radial_outlier_inlier(downsampled_pcd_2)
    # cl = outlier_pcd[0]
    # ind = outlier_pcd[1]
    # display_inlier_outlier(downsampled_pcd_2, ind)
    # o3d.visualization.draw_geometries([radial_outlier_inlier(downsampled_pcd_2, True)]) # viewing pointcloud with outliers removed

    # plot_images(current_image1, current_image2)

    # # ICP
    # point_to_point_ICP(downsampled_pcd_1, downsampled_pcd_2)

    # evaluate_ICP(downsampled_pcd_1, downsampled_pcd_2)

    # convert_ply_to_voxel(pcd)



    # Test Depth Map module

    # pathToImage = './img_sets/square_dataset/images/'
    # model_type = "MiDaS"
    # src1 = cv.imread('%s%s%s'%(pathToImage, "1677066767959_L", IMAGE_TYPE))
    # src2 = cv.imread('%s%s%s'%(pathToImage, "1677066757881_L", IMAGE_TYPE))

    # load_model =  load_midas_model(model_type)
    # transform = load_model[0]
    # midas = load_model[1]
    # device = load_model[2]

    # depth_map = compute_depth_map(transform, midas, device, src1)
    # show_depth_map(depth_map, "right")

    # depth_map2 = compute_depth_map(transform, midas, device, src2)
    # show_depth_map(depth_map2, "left")

    # compare_depth_map(depth_map, depth_map2, "right_left")


    # Test Camera module

    # testAllFeatures()


