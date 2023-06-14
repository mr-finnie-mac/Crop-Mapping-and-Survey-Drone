import glob
import time
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as plt_colors
import cv2 as cv
import os
import sys
from config import *
import copy
from sous import *
import random
from typing import List
from scipy.spatial.distance import cdist

from depthmap import *

# Create a custom colormap with 100 colors
custom_cmap = ['#8B4513', '#3F3F3F', '#7F7F7F', '#BFBFBF', '#FFFFFF', '#FF0000', '#FFA500', '#FFFF00', '#008000', '#0000FF',
          '#4B0082', '#9400D3', '#FF00FF', '#00FFFF', '#ADD8E6', '#F08080', '#90EE90', '#D3D3D3', '#FFC0CB', '#00FF00',
          '#800000', '#FFD700', '#FFA07A', '#7CFC00', '#DC143C', '#00BFFF', '#FF1493', '#FF8C00', '#48D1CC', '#B0C4DE',
          '#00CED1', '#9400D3', '#FF00FF', '#800080', '#FFC0CB', '#00FF00', '#F0E68C', '#ADFF2F', '#6A5ACD', '#FFE4C4',
          '#B8860B', '#FA8072', '#87CEFA', '#BA55D3', '#AFEEEE', '#FFDAB9', '#DA70D6', '#FF7F50', '#00FA9A', '#D8BFD8',
          '#4169E1', '#C71585', '#FF4500', '#FF69B4', '#ADFF2F', '#CD5C5C', '#EEE8AA', '#8A2BE2', '#556B2F', '#FFA07A',
          '#F5DEB3', '#00FFFF', '#00CED1', '#6B8E23', '#DB7093', '#191970', '#FAEBD7', '#FFB6C1', '#00FA9A', '#8B0000',
          '#008B8B', '#F08080', '#FF6347', '#008000', '#000080', '#FF8C00', '#9932CC', '#FF69B4', '#8B008B', '#FFA500',
          '#0000CD', '#800080', '#FFD700', '#98FB98', '#9400D3', '#20B2AA', '#FFE4E1', '#2E8B57', '#FF00FF', '#FF1493',
          '#FFC0CB', '#4169E1', '#B22222', '#FF4500', '#87CEEB', '#228B22', '#8B4513', '#FA8072', '#4B0082', '#ADD8E6',
          '#DAA520', '#1E90FF', '#FF69B4', '#9370DB', '#CD853F', '#FFB6C1', '#FAFAD2', '#90EE90', '#808000', '#BA55D3',
          '#8B0000', '#008080', '#F5DEB3', '#EEE8AA', '#FF7F50', '#00BFFF', '#483D8B', '#FF7F50', '#00BFFF', '#483D8B',
          '#4169E1', '#C71585', '#FF4500', '#FF69B4', '#ADFF2F', '#CD5C5C', '#EEE8AA', '#8A2BE2', '#556B2F', '#FFA07A',
          '#F5DEB3', '#00FFFF', '#00CED1', '#6B8E23', '#DB7093', '#191970', '#FAEBD7', '#FFB6C1', '#00FA9A', '#8B0000',
          '#008B8B', '#F08080', '#FF6347', '#008000', '#000080', '#FF8C00', '#9932CC', '#FF69B4', '#8B008B', '#FFA500',
          '#0000CD', '#800080', '#FFD700', '#98FB98', '#9400D3', '#20B2AA', '#FFE4E1', '#2E8B57', '#FF00FF', '#FF1493',
          '#FFC0CB', '#4169E1', '#B22222', '#FF4500', '#87CEEB', '#228B22', '#8B4513', '#FA8072', '#4B0082', '#ADD8E6',
          '#DAA520', '#1E90FF', '#FF69B4', '#9370DB', '#CD853F', '#FFB6C1', '#FAFAD2', '#90EE90', '#808000', '#BA55D3',
          '#B8860B', '#FA8072', '#87CEFA', '#BA55D3', '#AFEEEE', '#FFDAB9', '#DA70D6', '#FF7F50', '#00FA9A', '#D8BFD8',
          '#4169E1', '#C71585', '#FF4500', '#FF69B4', '#ADFF2F', '#CD5C5C', '#EEE8AA', '#8A2BE2', '#556B2F', '#FFA07A',
          '#F5DEB3', '#00FFFF', '#00CED1', '#6B8E23', '#DB7093', '#191970', '#FAEBD7', '#FFB6C1', '#00FA9A', '#8B0000',
          '#008B8B', '#F08080', '#FF6347', '#008000', '#000080', '#FF8C00', '#9932CC', '#FF69B4', '#8B008B', '#FFA500',
          '#0000CD', '#800080', '#FFD700', '#98FB98', '#9400D3', '#20B2AA', '#FFE4E1', '#2E8B57', '#FF00FF', '#FF1493',
          '#FFC0CB', '#4169E1', '#B22222', '#FF4500', '#87CEEB', '#228B22', '#8B4513', '#FA8072', '#4B0082', '#ADD8E6',
          '#DAA520', '#1E90FF', '#FF69B4', '#9370DB', '#CD853F', '#FFB6C1', '#FAFAD2', '#90EE90', '#808000', '#BA55D3',
          '#8B0000', '#008080', '#F5DEB3', '#EEE8AA', '#FF7F50', '#00BFFF', '#483D8B', '#FF7F50', '#00BFFF', '#483D8B',
          '#4169E1', '#C71585', '#FF4500', '#FF69B4', '#ADFF2F', '#CD5C5C', '#EEE8AA', '#8A2BE2', '#556B2F', '#FFA07A',
          '#F5DEB3', '#00FFFF', '#00CED1', '#6B8E23', '#DB7093', '#191970', '#FAEBD7', '#FFB6C1', '#00FA9A', '#8B0000',
          '#008B8B', '#F08080', '#FF6347', '#008000', '#000080', '#FF8C00', '#9932CC', '#FF69B4', '#8B008B', '#FFA500',
          '#0000CD', '#800080', '#FFD700', '#98FB98', '#9400D3', '#20B2AA', '#FFE4E1', '#2E8B57', '#FF00FF', '#FF1493',
          '#FFC0CB', '#4169E1', '#B22222', '#FF4500', '#87CEEB', '#228B22', '#8B4513', '#FA8072', '#4B0082', '#ADD8E6',
          '#DAA520', '#1E90FF', '#FF69B4', '#9370DB', '#CD853F', '#FFB6C1', '#FAFAD2', '#90EE90', '#808000', '#BA55D3',
          '#B8860B', '#FA8072', '#87CEFA', '#BA55D3', '#AFEEEE', '#FFDAB9', '#DA70D6', '#FF7F50', '#00FA9A', '#D8BFD8',
          '#4169E1', '#C71585', '#FF4500', '#FF69B4', '#ADFF2F', '#CD5C5C', '#EEE8AA', '#8A2BE2', '#556B2F', '#FFA07A',
          '#F5DEB3', '#00FFFF', '#00CED1', '#6B8E23', '#DB7093', '#191970', '#FAEBD7', '#FFB6C1', '#00FA9A', '#8B0000',
          '#008B8B', '#F08080', '#FF6347', '#008000', '#000080', '#FF8C00', '#9932CC', '#FF69B4', '#8B008B', '#FFA500',
          '#0000CD', '#800080', '#FFD700', '#98FB98', '#9400D3', '#20B2AA', '#FFE4E1', '#2E8B57', '#FF00FF', '#FF1493',
          '#FFC0CB', '#4169E1', '#B22222', '#FF4500', '#87CEEB', '#228B22', '#8B4513', '#FA8072', '#4B0082', '#ADD8E6',
          '#DAA520', '#1E90FF', '#FF69B4', '#9370DB', '#CD853F', '#FFB6C1', '#FAFAD2', '#90EE90', '#808000', '#BA55D3',
          '#8B0000', '#008080', '#F5DEB3', '#EEE8AA', '#FF7F50', '#00BFFF', '#483D8B', '#FF7F50', '#00BFFF', '#483D8B',
          '#4169E1', '#C71585', '#FF4500', '#FF69B4', '#ADFF2F', '#CD5C5C', '#EEE8AA', '#8A2BE2', '#556B2F', '#FFA07A',
          '#F5DEB3', '#00FFFF', '#00CED1', '#6B8E23', '#DB7093', '#191970', '#FAEBD7', '#FFB6C1', '#00FA9A', '#8B0000',
          '#008B8B', '#F08080', '#FF6347', '#008000', '#000080', '#FF8C00', '#9932CC', '#FF69B4', '#8B008B', '#FFA500',
          '#0000CD', '#800080', '#FFD700', '#98FB98', '#9400D3', '#20B2AA', '#FFE4E1', '#2E8B57', '#FF00FF', '#FF1493',
          '#FFC0CB', '#4169E1', '#B22222', '#FF4500', '#87CEEB', '#228B22', '#8B4513', '#FA8072', '#4B0082', '#ADD8E6',
          '#DAA520', '#1E90FF', '#FF69B4', '#9370DB', '#CD853F', '#FFB6C1', '#FAFAD2', '#90EE90', '#808000', '#BA55D3',
          '#B8860B', '#FA8072', '#87CEFA', '#BA55D3', '#AFEEEE', '#FFDAB9', '#DA70D6', '#FF7F50', '#00FA9A', '#D8BFD8',
          '#4169E1', '#C71585', '#FF4500', '#FF69B4', '#ADFF2F', '#CD5C5C', '#EEE8AA', '#8A2BE2', '#556B2F', '#FFA07A',
          '#F5DEB3', '#00FFFF', '#00CED1', '#6B8E23', '#DB7093', '#191970', '#FAEBD7', '#FFB6C1', '#00FA9A', '#8B0000',
          '#008B8B', '#F08080', '#FF6347', '#008000', '#000080', '#FF8C00', '#9932CC', '#FF69B4', '#8B008B', '#FFA500',
          '#0000CD', '#800080', '#FFD700', '#98FB98', '#9400D3', '#20B2AA', '#FFE4E1', '#2E8B57', '#FF00FF', '#FF1493',
          '#FFC0CB', '#4169E1', '#B22222', '#FF4500', '#87CEEB', '#228B22', '#8B4513', '#FA8072', '#4B0082', '#ADD8E6',
          '#DAA520', '#1E90FF', '#FF69B4', '#9370DB', '#CD853F', '#FFB6C1', '#FAFAD2', '#90EE90', '#808000', '#BA55D3',
          '#8B0000', '#008080', '#F5DEB3', '#EEE8AA', '#FF7F50', '#00BFFF', '#483D8B', '#FF7F50', '#00BFFF', '#483D8B',
          '#4169E1', '#C71585', '#FF4500', '#FF69B4', '#ADFF2F', '#CD5C5C', '#EEE8AA', '#8A2BE2', '#556B2F', '#FFA07A',
          '#F5DEB3', '#00FFFF', '#00CED1', '#6B8E23', '#DB7093', '#191970', '#FAEBD7', '#FFB6C1', '#00FA9A', '#8B0000',
          '#008B8B', '#F08080', '#FF6347', '#008000', '#000080', '#FF8C00', '#9932CC', '#FF69B4', '#8B008B', '#FFA500',
          '#0000CD', '#800080', '#FFD700', '#98FB98', '#9400D3', '#20B2AA', '#FFE4E1', '#2E8B57', '#FF00FF', '#FF1493',
          '#FFC0CB', '#4169E1', '#B22222', '#FF4500', '#87CEEB', '#228B22', '#8B4513', '#FA8072', '#4B0082', '#ADD8E6',
          '#DAA520', '#1E90FF', '#FF69B4', '#9370DB', '#CD853F', '#FFB6C1', '#FAFAD2', '#90EE90', '#808000', '#BA55D3',
          '#4B0082', '#9400D3', '#FF00FF', '#00FFFF', '#ADD8E6', '#F08080', '#90EE90', '#D3D3D3', '#FFC0CB', '#00FF00',
          '#800000', '#FFD700', '#FFA07A', '#7CFC00', '#DC143C', '#00BFFF', '#FF1493', '#FF8C00', '#48D1CC', '#B0C4DE',
          '#00CED1', '#9400D3', '#FF00FF', '#800080', '#FFC0CB', '#00FF00', '#F0E68C', '#ADFF2F', '#6A5ACD', '#FFE4C4',
          '#B8860B', '#FA8072', '#87CEFA', '#BA55D3', '#AFEEEE', '#FFDAB9', '#DA70D6', '#FF7F50', '#00FA9A', '#D8BFD8',
          '#4169E1', '#C71585', '#FF4500', '#FF69B4', '#ADFF2F', '#CD5C5C', '#EEE8AA', '#8A2BE2', '#556B2F', '#FFA07A',
          '#F5DEB3', '#00FFFF', '#00CED1', '#6B8E23', '#DB7093', '#191970', '#FAEBD7', '#FFB6C1', '#00FA9A', '#8B0000',
          '#008B8B', '#F08080', '#FF6347', '#008000', '#000080', '#FF8C00', '#9932CC', '#FF69B4', '#8B008B', '#FFA500',
          '#0000CD', '#800080', '#FFD700', '#98FB98', '#9400D3', '#20B2AA', '#FFE4E1', '#2E8B57', '#FF00FF', '#FF1493',
          '#FFC0CB', '#4169E1', '#B22222', '#FF4500', '#87CEEB', '#228B22', '#8B4513', '#FA8072', '#4B0082', '#ADD8E6',
          '#DAA520', '#1E90FF', '#FF69B4', '#9370DB', '#CD853F', '#FFB6C1', '#FAFAD2', '#90EE90', '#808000', '#BA55D3',
          '#8B0000', '#008080', '#F5DEB3', '#EEE8AA', '#FF7F50', '#00BFFF', '#483D8B', '#FF7F50', '#00BFFF', '#483D8B',
          '#4169E1', '#C71585', '#FF4500', '#FF69B4', '#ADFF2F', '#CD5C5C', '#EEE8AA', '#8A2BE2', '#556B2F', '#FFA07A',
          '#F5DEB3', '#00FFFF', '#00CED1', '#6B8E23', '#DB7093', '#191970', '#FAEBD7', '#FFB6C1', '#00FA9A', '#8B0000',
          '#008B8B', '#F08080', '#FF6347', '#008000', '#000080', '#FF8C00', '#9932CC', '#FF69B4', '#8B008B', '#FFA500',
          '#0000CD', '#800080', '#FFD700', '#98FB98', '#9400D3', '#20B2AA', '#FFE4E1', '#2E8B57', '#FF00FF', '#FF1493',
          '#FFC0CB', '#4169E1', '#B22222', '#FF4500', '#87CEEB', '#228B22', '#8B4513', '#FA8072', '#4B0082', '#ADD8E6',
          '#DAA520', '#1E90FF', '#FF69B4', '#9370DB', '#CD853F', '#FFB6C1', '#FAFAD2', '#90EE90', '#808000', '#BA55D3',
          '#B8860B', '#FA8072', '#87CEFA', '#BA55D3', '#AFEEEE', '#FFDAB9', '#DA70D6', '#FF7F50', '#00FA9A', '#D8BFD8',
          '#4169E1', '#C71585', '#FF4500', '#FF69B4', '#ADFF2F', '#CD5C5C', '#EEE8AA', '#8A2BE2', '#556B2F', '#FFA07A',
          '#F5DEB3', '#00FFFF', '#00CED1', '#6B8E23', '#DB7093', '#191970', '#FAEBD7', '#FFB6C1', '#00FA9A', '#8B0000',
          '#008B8B', '#F08080', '#FF6347', '#008000', '#000080', '#FF8C00', '#9932CC', '#FF69B4', '#8B008B', '#FFA500',
          '#0000CD', '#800080', '#FFD700', '#98FB98', '#9400D3', '#20B2AA', '#FFE4E1', '#2E8B57', '#FF00FF', '#FF1493',
          '#FFC0CB', '#4169E1', '#B22222', '#FF4500', '#87CEEB', '#228B22', '#8B4513', '#FA8072', '#4B0082', '#ADD8E6',
          '#DAA520', '#1E90FF', '#FF69B4', '#9370DB', '#CD853F', '#FFB6C1', '#FAFAD2', '#90EE90', '#808000', '#BA55D3',
          '#8B0000', '#008080', '#F5DEB3', '#EEE8AA', '#FF7F50', '#00BFFF', '#483D8B', '#FF7F50', '#00BFFF', '#483D8B',
          '#4169E1', '#C71585', '#FF4500', '#FF69B4', '#ADFF2F', '#CD5C5C', '#EEE8AA', '#8A2BE2', '#556B2F', '#FFA07A',
          '#F5DEB3', '#00FFFF', '#00CED1', '#6B8E23', '#DB7093', '#191970', '#FAEBD7', '#FFB6C1', '#00FA9A', '#8B0000',
          '#008B8B', '#F08080', '#FF6347', '#008000', '#000080', '#FF8C00', '#9932CC', '#FF69B4', '#8B008B', '#FFA500',
          '#0000CD', '#800080', '#FFD700', '#98FB98', '#9400D3', '#20B2AA', '#FFE4E1', '#2E8B57', '#FF00FF', '#FF1493',
          '#FFC0CB', '#4169E1', '#B22222', '#FF4500', '#87CEEB', '#228B22', '#8B4513', '#FA8072', '#4B0082', '#ADD8E6',
          '#DAA520', '#1E90FF', '#FF69B4', '#9370DB', '#CD853F', '#FFB6C1', '#FAFAD2', '#90EE90', '#808000', '#BA55D3',
          '#B8860B', '#FA8072', '#87CEFA', '#BA55D3', '#AFEEEE', '#FFDAB9', '#DA70D6', '#FF7F50', '#00FA9A', '#D8BFD8',
          '#4169E1', '#C71585', '#FF4500', '#FF69B4', '#ADFF2F', '#CD5C5C', '#EEE8AA', '#8A2BE2', '#556B2F', '#FFA07A',
          '#F5DEB3', '#00FFFF', '#00CED1', '#6B8E23', '#DB7093', '#191970', '#FAEBD7', '#FFB6C1', '#00FA9A', '#8B0000',
          '#008B8B', '#F08080', '#FF6347', '#008000', '#000080', '#FF8C00', '#9932CC', '#FF69B4', '#8B008B', '#FFA500',
          '#0000CD', '#800080', '#FFD700', '#98FB98', '#9400D3', '#20B2AA', '#FFE4E1', '#2E8B57', '#FF00FF', '#FF1493',
          '#FFC0CB', '#4169E1', '#B22222', '#FF4500', '#87CEEB', '#228B22', '#8B4513', '#FA8072', '#4B0082', '#ADD8E6',
          '#DAA520', '#1E90FF', '#FF69B4', '#9370DB', '#CD853F', '#FFB6C1', '#FAFAD2', '#90EE90', '#808000', '#BA55D3',
          '#8B0000', '#008080', '#F5DEB3', '#EEE8AA', '#FF7F50', '#00BFFF', '#483D8B', '#FF7F50', '#00BFFF', '#483D8B',
          '#4169E1', '#C71585', '#FF4500', '#FF69B4', '#ADFF2F', '#CD5C5C', '#EEE8AA', '#8A2BE2', '#556B2F', '#FFA07A',
          '#F5DEB3', '#00FFFF', '#00CED1', '#6B8E23', '#DB7093', '#191970', '#FAEBD7', '#FFB6C1', '#00FA9A', '#8B0000',
          '#008B8B', '#F08080', '#FF6347', '#008000', '#000080', '#FF8C00', '#9932CC', '#FF69B4', '#8B008B', '#FFA500',
          '#0000CD', '#800080', '#FFD700', '#98FB98', '#9400D3', '#20B2AA', '#FFE4E1', '#2E8B57', '#FF00FF', '#FF1493',
          '#FFC0CB', '#4169E1', '#B22222', '#FF4500', '#87CEEB', '#228B22', '#8B4513', '#FA8072', '#4B0082', '#ADD8E6',
          '#DAA520', '#1E90FF', '#FF69B4', '#9370DB', '#CD853F', '#FFB6C1', '#FAFAD2', '#90EE90', '#808000', '#BA55D3',
          '#B8860B', '#FA8072', '#87CEFA', '#BA55D3', '#AFEEEE', '#FFDAB9', '#DA70D6', '#FF7F50', '#00FA9A', '#D8BFD8',
          '#4169E1', '#C71585', '#FF4500', '#FF69B4', '#ADFF2F', '#CD5C5C', '#EEE8AA', '#8A2BE2', '#556B2F', '#FFA07A',
          '#F5DEB3', '#00FFFF', '#00CED1', '#6B8E23', '#DB7093', '#191970', '#FAEBD7', '#FFB6C1', '#00FA9A', '#8B0000',
          '#008B8B', '#F08080', '#FF6347', '#008000', '#000080', '#FF8C00', '#9932CC', '#FF69B4', '#8B008B', '#FFA500',
          '#0000CD', '#800080', '#FFD700', '#98FB98', '#9400D3', '#20B2AA', '#FFE4E1', '#2E8B57', '#FF00FF', '#FF1493',
          '#FFC0CB', '#4169E1', '#B22222', '#FF4500', '#87CEEB', '#228B22', '#8B4513', '#FA8072', '#4B0082', '#ADD8E6',
          '#DAA520', '#1E90FF', '#FF69B4', '#9370DB', '#CD853F', '#FFB6C1', '#FAFAD2', '#90EE90', '#808000', '#BA55D3',
          '#8B0000', '#008080', '#F5DEB3', '#EEE8AA', '#FF7F50', '#00BFFF', '#483D8B', '#FF7F50', '#00BFFF', '#483D8B',
          '#4169E1', '#C71585', '#FF4500', '#FF69B4', '#ADFF2F', '#CD5C5C', '#EEE8AA', '#8A2BE2', '#556B2F', '#FFA07A',
          '#F5DEB3', '#00FFFF', '#00CED1', '#6B8E23', '#DB7093', '#191970', '#FAEBD7', '#FFB6C1', '#00FA9A', '#8B0000',
          '#008B8B', '#F08080', '#FF6347', '#008000', '#000080', '#FF8C00', '#9932CC', '#FF69B4', '#8B008B', '#FFA500',
          '#0000CD', '#800080', '#FFD700', '#98FB98', '#9400D3', '#20B2AA', '#FFE4E1', '#2E8B57', '#FF00FF', '#FF1493',
          '#FFC0CB', '#4169E1', '#B22222', '#FF4500', '#87CEEB', '#228B22', '#8B4513', '#FA8072', '#4B0082', '#ADD8E6',
          '#DAA520', '#1E90FF', '#FF69B4', '#9370DB', '#CD853F', '#FFB6C1', '#FAFAD2', '#90EE90', '#808000', '#BA55D3',
          '#8B0000', '#008080', '#F5DEB3', '#EEE8AA']
rgb_colors = [plt_colors.to_rgb(hex_color) for hex_color in custom_cmap]

class PointCloud():
    def __init__(self, rgbd, mission, id):
        self.id = id
        self.rgbd = rgbd
        self.cloud = o3d.geometry.PointCloud()
        self.clusters = None
        self.plants = None
        self.ground = None
        self.exportPath = None
        self.depthmap = None
        self.mission = mission
        self.camera_params = None
        self.colour = None

    def setDepthmap(self, depthmap: Depthmap):
        self.depthmap = depthmap

    def generateDepthMap(self, imagePath, exportPath, params, left, right):
        self.camera_params = Depthmap(self.id, imagePath, exportPath, self.mission, params, left, right)

    def makeRGBD(self, useWLS, depthmap):
        # Convert ndarray color image and depth images to O3D rgbd
        if depthmap is False:
            colour = o3d.geometry.Image((self.depthmap.rect_Left).astype(np.uint8))
            if useWLS:
                depth = o3d.geometry.Image((self.depthmap.wls).astype(np.uint8))
            else:
                depth = o3d.geometry.Image((self.depthmap.disaprity).astype(np.uint8))
        else:
            colour = o3d.geometry.Image((self.colour).astype(np.uint8))
            depth = o3d.geometry.Image((depthmap).astype(np.uint8)) 
        self.rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(colour, depth)

    def loadFile(self, file_path):
        self.cloud = o3d.io.read_point_cloud(file_path)
    
    def saveFile(self, file_path):
        o3d.io.write_point_cloud(file_path, self.points)

    def generateCamaraParams(self):
        self.camera_params = CameraExtIntrinsics(self.id)
    
    # Make the pointcloud
    def generate(self):
        self.cloud = o3d.geometry.PointCloud.create_from_rgbd_image(self.rgbd, intrinsic=self.camera_params.intMat, extrinsic=self.camera_params.extMat)

    # Configure
    def reverseScale(self):
        self.cloud.scale(-1.0, center = self.cloud.get_center()) # inverting the scale so points are not wrong way round

    def conditionPointcloud(self, useDepthShift):
        self.cloud.scale(1 / np.max(self.cloud.get_max_bound() - self.cloud.get_min_bound()), center=self.cloud.get_center())
        if useDepthShift:
            self.cloud.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(2000, 3))) # enable for blue-red depth colour shift

    def observeCloud(self):
        o3d.visualization.draw_geometries([self.cloud])

    def makeVoxel(self):
        self.voxels = o3d.geometry.VoxelGrid.create_from_point_cloud(self.cloud, voxel_size=0.005)

    def downsample(self):
        self.cloud = self.cloud.uniform_down_sample(every_k_points=K_POINTS)


class CameraExtIntrinsics():
    def __init__(self, id):
        self.id = id
        self.extFile = cv.FileStorage.open(CAMERA_INTRINSICS, cv.FileStorage_READ)
        self.intFile = cv.FileStorage.open(CAMERA_EXTRINSICS, cv.FileStorage_READ)
        self.extMat = np.asarray(self.extFile.getNode('mx').mat())
        self.intMat = np.asarray(self.intFile.getNode('mx').mat())

    def updateValues(self, extFile, intFile):
        self.extFile = extFile
        self.intFile - intFile

class PointcloudSet():
    def __init__(self, id, mission):
        self.id = id
        self.mission = mission
        self.set = []
        self.map = None
        self.crops = []
        self.ground = []

    def addPointclouds(self, set: List[PointCloud]):
        for pcd in set:
            self.set.append(pcd)

    def alignMembersICP(self, threshold=DEFAULT_ICP_THRESHOLD, trans_init=np.asarray([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])):
        target = self.set[0]
        for pcd in self.set[1:]:
            reg = o3d.pipelines.registration.registration_icp(pcd, target, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())
            # observe_pointcloud(pcd)
            # draw_registration_result(pcd, target, reg.transformation)
        self.map = target

    def observeCrops(self):
        o3d.visualization.draw_geometries(self.crops)

    def findCropsDBSCAN(self):
        with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(
                self.map.cluster_dbscan(eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS, print_progress=True))
       # Create a list of segmented point clouds
        pcd_list = []
        for i in np.unique(labels):
            points = np.asarray(self.map.points)[labels == i]
            pcd_cluster = o3d.geometry.PointCloud()
            pcd_cluster.points = o3d.utility.Vector3dVector(points)
            pcd_list.append(pcd_cluster)
        self.ground = pcd_list[0]
        self.crops = pcd_list[1:]

class PointcloudAnalysis():
    def __init__(self, id, mission):
        self.id = id
        self.mission = mission
        self.pointcloud_set = None
        self.crops = [] # clusters
        self.density = None # how close together the clusters are
        self.average_height = None

    def addSet(self, set):
        self.pointcloud_set = PointcloudSet(self.id, self.mission)
        self.pointcloud_set.addPointclouds(set)
        self.pointcloud_set.findCropsDBSCAN()
    
    def addCrop(self, crop):
        # example = Crop(self.id, self.mission)
        # self.crop = example
        self.crops.append(crop)

    def calculateAverageHeight(self):
        avgs = []
        for crop in self.crops:
            avgs.append(crop.height)
        self.average_height = np.mean(avgs)

    def calculateDensity(self):
        # make list of center values for each crop point cloud
        center_values = []
        for crop in self.crops:
            center_values.append(crop.position)
            print(crop.position)
        centers = np.array(center_values)

        # Calculate pairwise Euclidean distances between center points
        distances = cdist(centers, centers, 'euclidean')

        # Exclude self-distances (diagonal elements)
        distances = distances[np.nonzero(~np.eye(distances.shape[0], dtype=bool))]

        # Calculate average gap
        average_gap = np.mean(distances)
        print("Average gap between center points:", average_gap)
        self.density = average_gap
        



class Crop():
    def __init__(self, id, mission):
        self.id = id
        self.mission = mission
        self.pointcloud = None
        self.bounds_vector  = None
        self.bounds = None
        self.volume = None
        self.height = None
        self.position = None

    def setPointcloud(self, pcd):
        self.pointcloud = pcd

    def measureBounds(self):
        self.bounds_vector = self.pointcloud.get_axis_aligned_bounding_box().get_box_points() # measures bounds from the ground up
        self.bounds = np.asarray(self.pointcloud.get_axis_aligned_bounding_box().get_box_points()) # measures bounds from the ground up
    
    def measureVolume(self):
        self.volume = self.bounds.volume() # measures volume inside bounds

    def measureHeight(self):
        # Calculate the height along the z-axis
        min_z = np.min(self.bounds[:, 2])
        max_z = np.max(self.bounds[:, 2])
        self.height = max_z - min_z

    def getPosition(self):
        self.position = self.pointcloud.get_center()
        

# construct pointcloud from depthmap and image path
def make_rgbd(image_path, depth_path):
    """Make rgb depth image from colour image and depth image path in directory. Returns rgbd object"""
    # Load in color and depth image to create the point cloud
    color_raw = o3d.io.read_image(image_path)
    # color_raw = cv.bitwise_not(color_raw) # depth and image data currently inverse to eachother, need to correct on one source. (depth)
    depth_raw = o3d.io.read_image(depth_path)
    plot_images(color_raw, depth_raw)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw)
    print("Made RGBD:")
    print(rgbd_image)
    return rgbd_image

# construct pointcloud from depthmap and image
def make_rgbd_from_np(colourImage, depthImage):
    """Make rgb depth image from colour image and depth image (np array). Returns rgbd object"""
    # Convert ndarray color image and depth images to O3D rgbd
    o3d_colour = o3d.geometry.Image((colourImage).astype(np.uint8))
    o3d_depth = o3d.geometry.Image((depthImage).astype(np.uint8))

    # plot_images(o3d_colour, o3d_depth)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_colour, o3d_depth)

    return rgbd_image

def plot_rgbd_image_images(rgbd_image):
    """Plots the colour and depth image of an rgbd image."""
    # Plot the images
    plt.subplot(1, 2, 1)
    plt.title('Grayscale image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('Depth image')
    plt.imshow(rgbd_image.depth)
    plt.show()

def plot_images(image1, image2):
    plt.subplot(1, 2, 1)
    plt.title('A')
    plt.imshow(image1)
    plt.subplot(1, 2, 2)
    plt.title('B')
    plt.imshow(image2)
    plt.show()

def convert_to_ply(rgbd_image):
    "Constructs a ply pointcloud from rgbd image, returns pcd"
    # load in intrinsic camera parameters from file generated by calibration script
    cv_file = cv.FileStorage()
    cv_file.open(CAMERA_INTRINSICS, cv.FileStorage_READ)
    # locate intrinsic params
    camera_intrinsic = cv_file.getNode('mx').mat()
    print(camera_intrinsic)

    # Obtain intrinsic camera parameters from file and convert it something o3d understands (pinholeCameraIntrinsic object)
    camera_intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(width=1920, height=1080, fx=camera_intrinsic[0][0],fy=camera_intrinsic[1][1], cx=camera_intrinsic[0][2], cy=camera_intrinsic[1][2])
    # print(camera_intrinsic_o3d.intrinsic_matrix)

    # load in intrinsic camera parameters from file generated by calibration script
    cv_file.open(CAMERA_EXTRINSICS, cv.FileStorage_READ)
    # locate extrinsic params
    camera_extrinsic = np.asarray(cv_file.getNode('mx').mat())
    print(camera_extrinsic)

    # Create the point cloud from images and camera intrisic parameters
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic=camera_intrinsic_o3d, extrinsic=camera_extrinsic)
    pcd.scale(-1.0, center = pcd.get_center()) # inverting the scale so points are not wrong way round

    return pcd

def condition_pointcloud(pcd, use_depth_colourshift = True):
    """Returns given pointcloud with scale and colour fixed"""
    pcd.scale(-1.0, center = pcd.get_center())
    pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()), center=pcd.get_center())
    if use_depth_colourshift:
        pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(2000, 3))) # enable for blue-red depth colour shift
    return pcd

def observe_pointcloud(pcd):
    """View a pointcloud in open3d visualiser"""
    # vis = o3d.visualization.Visualizer()
    # vis.create_window("Pointcloud")
    pcd.scale(-1.0, center = pcd.get_center())
    print('Opening point cloud in window')
    o3d.visualization.draw_geometries([pcd])

def convert_ply_to_voxel(pcd):
    # vis = o3d.visualization.Visualizer()
    # vis.create_window("Pointcloud - Square_dataset")
    # pointcloud to voxel grid
    print('Displaying voxel grid ...')
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.005)
    o3d.visualization.draw_geometries([voxel_grid])
    return voxel_grid

def export_pointcloud(pcd, filename=str(round(time.time()*1000))+"_exported_pcd.ply"):
    o3d.io.write_point_cloud('%s%s'%(PLY_DESTINATION_PATH, filename), pcd, False, False, False)
    print("Pointcloud exported to " + PLY_DESTINATION_PATH)

def downsample_pointcloud(pcd):
    """Downsamples the provided pointcloud. Returns a downsampled pointcloud"""
    # downsample our pcd
    ds_voxel = pcd.uniform_down_sample(every_k_points=K_POINTS)
    return ds_voxel

def radial_outlier_inlier(ds_pcd, remove=False):
    "Performs outlier removal within a radius using a downsampled pointcloud. Returns cloud and index values"
    cl, ind = ds_pcd.remove_radius_outlier(nb_points=16, radius=0.005)

    return (cl, ind) if remove==False else (ds_pcd.select_by_index(ind)) # return either cloud and index or cloud with removed outlier values

def cluster_statistical_outlier_removal(pcd, remove=False, show_process=False):
    "Performs outlier removal within a radius using a downsampled pointcloud. Returns cloud and index values"
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=DBSCAN_MIN_POINTS-1, std_ratio=2.0)
    if show_process: display_inlier_outlier(pcd, ind)
    return (cl, ind) if remove==False else (pcd.select_by_index(ind)) # return either cloud and index or cloud with removed outlier values

def display_inlier_outlier(cloud, ind):
    """Displays inlier - outlier using pointcloud and index values"""
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

def draw_registration_result(source, target, transformation):
    """Shows how two pointclouds would be combined, provided ICP has been performed and transformation matrix supplied"""
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])

def evaluate_ICP(source, target, threshold=DEFAULT_ICP_THRESHOLD, trans_init=np.asarray([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])):
    """
        Returns the quality of the ICP, Fitness describes how well they fit (higher is better), or the overlaping areas. 
        More technically, num of inlier correspondences or points in target. RMSD (Root-mean-square deviation)
        describes a measure of error or differnece in value (between points in the two clouds in this case).
    """
    print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
    print(evaluation)

def point_to_point_ICP(source, target, threshold=DEFAULT_ICP_THRESHOLD, trans_init=np.asarray([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])):
    print("Apply point-to-point ICP")
    # reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())
    reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    draw_registration_result(source, target, reg_p2p.transformation)
    return reg_p2p

def pointcloud_clustering_basic(input_pcd):
    """Implements the DBSCAN clustering method to get segmentation of a pointcloud. Displays these clusters in a multcolour visual"""
    pcd = input_pcd

    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            pcd.cluster_dbscan(eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS, print_progress=True))
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors = plt_colors.ListedColormap(custom_cmap)(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])

def visualise_segmented_pointcloud_list(pcd_list):
    """Observe a segmented pointcloud in a clustered colours"""

    # generate 200 colours
    # num_colors = 200
    # cmap = plt.get_cmap('tab20')
    # colors_rgb = cmap(np.linspace(0, 1, num_colors))
    # colors_rgb = colors_rgb[:, :3] # convert from rgba to rgb

    num_clusters = len(pcd_list)
    print(num_clusters)
    print(len(rgb_colors))
    # colors = plt_colors.ListedColormap(custom_cmap)#(labels / (max_label if max_label > 0 else 1))
    for i in range(num_clusters):
        # print(rgb_colors[i])
        pcd_list[i].paint_uniform_color(rgb_colors[i])

    # o3d.visualization.draw_geometries(pcd_list)
    custom_draw_geometry_with_rotation(pcd=pcd_list)
    
    # pcd_l = pcd
    # new_list = []
    # for pcd in pcd_l:
    #     points = np.asarray(pcd.points)
    #     # Define the rotation matrix for 90 degrees around the Z-axis
    #     r_mat = np.array([[0, -1, 0],
    #                     [1, 0, 0],
    #                     [0, 0, 1]])

    #     # Rotate the points by multiplying with the rotation matrix
    #     r_points = np.dot(points, r_mat)

    #     # Convert the rotated points back to an Open3D point cloud
    #     r_pcd = o3d.geometry.PointCloud()
    #     r_pcd.points = o3d.utility.Vector3dVector(r_points)
    #     new_list.append(r_pcd)
    
def custom_draw_geometry_with_rotation(pcd):

    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(4.0, 0.0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback(pcd,
                                                              rotate_view)

def pointcloud_clustering_segmentation(input_pcd):
    """Implements the DBSCAN clustering method to get segmentation of a pointcloud and split the pointcloud up into those clusters. Returns pcd list"""
    pcd = input_pcd

    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            pcd.cluster_dbscan(eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS, print_progress=True))

    # Create a list of segmented point clouds
    pcd_list = []
    for i in np.unique(labels):
        points = np.asarray(pcd.points)[labels == i]
        pcd_cluster = o3d.geometry.PointCloud()
        pcd_cluster.points = o3d.utility.Vector3dVector(points)
        pcd_list.append(pcd_cluster)
    
    return pcd_list


    # with o3d.utility.VerbosityContextManager(
    #         o3d.utility.VerbosityLevel.Debug) as cm:
    #     labels = np.array(
    #         pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

    # max_label = labels.max()
    # print(f"point cloud has {max_label + 1} clusters")
    # # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # colors = plt_colors.ListedColormap(custom_cmap)(labels / (max_label if max_label > 0 else 1))
    # colors[labels < 0] = 0
    # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([pcd])

# def combine_via_icp(source, target):
#     convert_to_ply
#     combined = point_to_point_ICP(source=source, target=target)

def generate_rgbd_pack_from_mission(missionCode, path=IMAGE_DESTINATION, image_list = None, depth_list = None):
    """Generate a list of rgbd objects from given mission/directory in the format [[name, rgbd], [name, rgbd]...]"""

    rgbd_list = []
    # if list of image and depth not given,run procedure from folder instead.
    if image_list == None and depth_list == None:
        # find mission code folder then get image and depth folders
        images = glob.glob("%smissions/%s/image/%s"%(path, missionCode, '*.jpeg'), recursive = False)
        depthmaps = glob.glob("%smissions/%s/depth/%s"%(path, missionCode, '*.png'), recursive = False)

        # go through all images in folder and make depthmaps and images for them, save to respective folders
        for image,depth in zip(images, depthmaps):
            # save left frame as image
            imageName_ext = os.path.basename(image)
            depthName_ext = os.path.basename(depth)
            # print(filename_ext)
            currentImage = cv.imread(image)
            currentDepth = cv.imread(depth)
        
        # make rgbd from current image and depth files, then add to list with name attached
            print("Generating rgbd of %s"%(depthName_ext))
            rgbd = make_rgbd_from_np(colourImage=currentImage, depthImage=currentDepth)
            rgbd_list.append([depthName_ext.replace('.png', ''),rgbd])

    else:
        
        for image, depth in zip(image_list, depth_list):
            rgbd = make_rgbd_from_np(colourImage=image, depthImage=depth)
            rgbd_list.append(['', rgbd])


    return rgbd_list

def generate_pointclouds_from_rgbd_list(rgbd_list, use_voxel=True):
    """Generates a pack of pointclouds from a rgbd_list, saves them to path and returns this pack as list of pointclouds (already downsampled!)."""
    pcd_list = []
    for rgbd in rgbd_list:
        print('Converting rgbd %s to pointcloud'%(rgbd[0]))
        pcd = convert_to_ply(rgbd_image=rgbd[1])
        pcd = condition_pointcloud(pcd)
        ds_pcd = downsample_pointcloud(pcd)
        if use_voxel:
            pcd = convert_ply_to_voxel(pcd=ds_pcd)
        else:
            pcd = ds_pcd
        pcd_list.append(pcd)

    return pcd_list

# disparity, filteredDisparity = stereo_depthmap_compute(leftImage=leftRGB, rightImage=rightRGB)
# rgbd = make_rgbd_from_np(leftRGB, filteredDisparity)

# pcd = convert_to_ply(rgbd)
# pcd = condition_pointcloud(pcd)
# refined_pcd = downsample_pointcloud(pcd)
# refined_pcd = radial_outlier_inlier(refined_pcd, remove=True)

# observe_pointcloud(refined_pcd)
# export_pointcloud(refined_pcd)


def analyse_cluster(pcd):
    # volume measure
    axis_bounds = pcd.get_axis_aligned_bounding_box() # measures volume from the ground up, like it's footprint.
    rot_bounds = pcd.get_oriented_bounding_box() # measures volume of the plant
    axis_bounds.color = (0, 1, 0)
    rot_bounds.color = (0, 0, 1)
    print("Volume = ")
    print(axis_bounds.volume())
    print(rot_bounds.volume())
    # o3d.visualization.draw_geometries([pcd, axis_bounds, rot_bounds])

    # posiitonal measure
    center = pcd.get_center()
    dims = rot_bounds.get_box_points()
    # dim_points = rot_bounds.get_box_points()
    # dim_points = o3d.geometry.PointCloud(
    # points=o3d.utility.Vector3dVector(np.asarray(dim_points)))

    return axis_bounds.volume(), rot_bounds.volume(), center, dims

    # """Processes the pointcloud performing conditioning, downsampling, clustering and segmentation"""
    # # raw pointcloud is conditioned and downsampled
    # refined_pcd = condition_pointcloud(pcd)
    # refined_pcd = downsample_pointcloud(refined_pcd)
    # observe_pointcloud(refined_pcd)

    # # pointcloud clusteres are calculated by perfroming DBSCAN algorithm 
    # pcd_clusters = pointcloud_clustering_segmentation(refined_pcd)
    
    # # remove outliers from 
    # refined_pcd_clusters = []
    # for cluster in pcd_clusters:
    #     refined_cluster = cluster#cluster_statistical_outlier_removal(cluster, remove=True, show_process=False)
    #     rand_rgb = random.choice(rgb_colors)
    #     refined_cluster.paint_uniform_color(rand_rgb)
    #     refined_pcd_clusters.append(refined_cluster)
    
    # print('Opening refined pointcloud clusters in window')
    # o3d.visualization.draw_geometries(refined_pcd_clusters)



def process_pointcloud(raw_pcd):
    """Processes the pointcloud performing conditioning, downsampling, clustering and segmentation"""
    # raw pointcloud is conditioned and downsampled
    refined_pcd = condition_pointcloud(raw_pcd)
    refined_pcd = downsample_pointcloud(refined_pcd)
    observe_pointcloud(refined_pcd)

    # pointcloud clusteres are calculated by perfroming DBSCAN algorithm 
    pcd_clusters = pointcloud_clustering_segmentation(refined_pcd)
    clusters_ax_vol = []
    clusters_rot_vol = []
    clusters_center = []
    calculated_dims = []
    # remove outliers from 
    refined_pcd_clusters = []
    for cluster in pcd_clusters:
        refined_cluster = cluster#cluster_statistical_outlier_removal(cluster, remove=True, show_process=False)
        rand_rgb = random.choice(rgb_colors)
        refined_cluster.paint_uniform_color(rand_rgb)
        refined_pcd_clusters.append(refined_cluster)
        ax_vol, rot_vol, center, dims = analyse_cluster(refined_cluster)
        clusters_ax_vol.append(ax_vol)
        clusters_rot_vol.append(rot_vol)
        clusters_center.append(center)
        print(np.asarray(dims))
        calculated_dims.append(dims)
        # calculated_dims

    clusters_ax_vol = np.asarray(clusters_ax_vol)
    clusters_rot_vol = np.asarray(clusters_rot_vol)
    clusters_center = np.asarray(clusters_center)
    ax_vol_mean = np.mean(clusters_ax_vol)
    rot_vol_mean = np.mean(clusters_rot_vol)
    center_mean = np.mean(clusters_center)
    
    print("This map has an average axial volume of %f, rotated volume of %f and the avarage relative position is %f"%(ax_vol_mean, rot_vol_mean, center_mean))

    


        
    
    print('Opening refined pointcloud clusters in window')
    o3d.visualization.draw_geometries(refined_pcd_clusters)

def open_ply(path):
    """Opens the pointcloud at the file path"""
    pcd = o3d.io.read_point_cloud(path)
    return pcd

def process_mission(missionCode):
    "Main loop for reconstruction of 3d scenes from rgbd data"

    # Check folder structure
    # should have been made using camera module's make_new_capture_folder function
    missionPath, imagePath, depthPath = check_mission_folder(missionCode)

    # Making fragments
    # do rgbd making here from imagePath and depthPath 
    rgbd_list = generate_rgbd_pack_from_mission(missionCode=missionCode)

    print(rgbd_list[0][0])

    # Registering fragments


    # Refine registered fragments


    # integrate scene

def get_file_paths(directory):
    file_paths = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and filename.lower().endswith('.ply'):
            file_paths.append(file_path)
    return file_paths

def show_pcd():
    pcdPath = 'blender/blender-models/alt/disected/'
    srcL = cv.imread("./blender/images/left.png")
    srcR = cv.imread("./blender/images/right.png")
    
    make_rgbd("./blender/images/right.png", "./disparity_image_blender-overhead.png")
    blend = PointCloud(None,"blender-og", 2)
   
    depthmap = cv.imread("./disparity_image_blender-overhead.png")
    srcL = cv.imread("./blender/images/left.png")

    blend.depthmap = depthmap
    blend.colour  = srcL
    blend.makeRGBD(False, depthmap=depthmap)

    blend.cloud = convert_to_ply(blend.rgbd)

    blend.conditionPointcloud(useDepthShift=True)

    blend.downsample()
    blend.observeCloud()
    blend.cloud = cluster_statistical_outlier_removal(blend.cloud, True)
    
    clusters = pointcloud_clustering_segmentation(blend.cloud)
    visualise_segmented_pointcloud_list(clusters)

    return clusters

def show_blender_pcd(pcd):
    # pcdPath = 'blender/blender-models/alt/disected/'
    # srcL = cv.imread("./blender/images/left.png")
    # srcR = cv.imread("./blender/images/right.png")
    
    # make_rgbd("./blender/images/right.png", "./disparity_image_blender-overhead.png")
    blend = PointCloud(None,"blender-og", 2)
   
    # depthmap = cv.imread("./disparity_image_blender-overhead.png")
    # srcL = cv.imread("./blender/images/left.png")

    # blend.depthmap = depthmap
    # blend.colour  = srcL
    # blend.makeRGBD(False, depthmap=depthmap)

    blend.cloud = pcd

    blend.conditionPointcloud(useDepthShift=True)

    blend.downsample()
    blend.observeCloud()
    blend.cloud = cluster_statistical_outlier_removal(blend.cloud, True)
    
    clusters = pointcloud_clustering_segmentation(blend.cloud)
    visualise_segmented_pointcloud_list(clusters)

    return clusters
    

if __name__ == "__main__":

    clusters = show_pcd()
    
    analysis = PointcloudAnalysis(112358, "blender_normal")
    this_pcd = PointCloud("rgbd", "blender_", 112358)
    
    for cluster in clusters:
        crop = Crop(112358, "blender_normal")

        crop.setPointcloud(cluster)
        crop.measureBounds()
        print(crop.bounds)
        crop.measureHeight()
        print(crop.height)
     
        crop.getPosition()
        print(crop.position)

        analysis.addCrop(crop)
    
    analysis.calculateDensity()
    analysis.calculateAverageHeight()
    print(analysis.density)
    print(analysis.average_height)
    o3d.visualization.draw_geometries(clusters)
    # pointcloud_clustering_basic(blend.cloud)

    crop = Crop(112358, "blender_norm")
    pcd = PointCloud("rgbd", "blender_norm", 112358)
    pcd.loadFile('blender/blender-models/normal/blendercrops_normal.ply')
    # pcd.loadFile('blender/blender-models/blender_newcrops_alt_new.ply')
    crop.setPointcloud(pcd.cloud)
    custom_draw_geometry_with_rotation(pcd=[pcd.cloud])
    DBSCAN_EPS = 0.0157
    show_blender_pcd(pcd.cloud)


    # crop2 = Crop(112358, "blender_alt")
    # pcd2 = PointCloud("rgbd", "blender_alt", 112358)
    # pcd2.loadFile('blender/blender-models/alt/disected/blendercrops_alt_003.ply')
    # crop2.setPointcloud(pcd2.cloud)


    # crop2.measureBounds()
    # print(crop2.bounds)
    # crop2.measureHeight()
    # print(crop2.height)
    # crop2.getPosition()
    # print(crop2.position)

    
    






