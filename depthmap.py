import glob
import cv2 as cv
import torch
import matplotlib.pyplot as plt
from config import *
from PIL import Image
import numpy as np
import time
import tensorflow as tf
from keras import layers
import pandas as pd
import os
from keras_pipe import *
from sous import *

global debug_show
debug_show = False
tf.random.set_seed(123)

pathToImage ='./img_sets/square_dataset/images/'
exportPath = './outputs/depthmaps/'
savePrependTag = 'depth_'
# IMAGE_NAME = '1677066767959_L'
# IMAGE_TYPE = '.jpeg'
model_type = "MiDaS"
# select a version of the MiDaS model
# model_type = "DPT_Large"
# model_type = "DPT_Hybrid"

class StereoMatrix:
    def __init__(self, filePath):
        self.filePath = filePath
        self.file = None
        self.stereoMapL_x = None
        self.stereoMapL_y = None
        self.stereoMapR_x = None
        self.stereoMapR_y = None

    def getFile(self):
        self.file = cv.FileStorage(self.filePath, cv.FILE_STORAGE_READ)

    def makeMatrices(self):
        self.stereoMapL_x = self.file.getNode("stereoMapL_x").mat()
        self.stereoMapL_y = self.file.getNode("stereoMapL_y").mat()
        self.stereoMapR_x = self.file.getNode('stereoMapR_x').mat()
        self.stereoMapR_y = self.file.getNode('stereoMapR_y').mat()



class Depthmap:
    def __init__(self, id, imagePath, exportPath, mission, params, left, right):
        # self.imagePath = imagePath
        self.exportPath = exportPath
        self.mission = mission
        self.id = id
        self.params = params
        self.left = left
        self.right = right
        self.matrices = None
        self.rect_Left = None
        self.rect_Right = None
        self.preset = None
        self.disaprity = None
        self.wls = None

    
    def getStereoParams(self):
        self.matrices = StereoMatrix("./STEREO_PARAMS.xml") 
        return self.matrices
    def generate_depthmap(self):

        # Applying stereo image rectification on the left image
        self.rect_Left= cv.remap(self.left,
                self.matrices.stereoMapL_x,
                self.matrices.stereoMapL_y,
                cv.INTER_LANCZOS4,
                cv.BORDER_CONSTANT,
                0)
        
        # Applying stereo image rectification on the right image
        self.rect_Right =  cv.remap(self.right,
                self.matrices.stereoMapR_x,
                self.matrices.stereoMapR_y,
                cv.INTER_LANCZOS4,
                cv.BORDER_CONSTANT,
                0)
        
        (minDisp, numDisp, blockSize, 
         p1, p2, dispMaxDiff, prefilCap, 
         uniqueRatio, speckWinSize, speckRange) = depthmap_presets(preset=self.params)
        
        stereo = cv.StereoSGBM_create(minDisparity=minDisp, numDisparities=numDisp, blockSize=blockSize, 
                                  P1=p1, P2=p2, disp12MaxDiff=dispMaxDiff, preFilterCap=prefilCap, 
                                  uniquenessRatio=uniqueRatio, speckleWindowSize=speckWinSize, speckleRange=speckRange)
        disparity = stereo.compute(self.rect_Left, self.rect_Right)

        leftMatcher = cv.StereoSGBM_create(minDisparity=minDisp, numDisparities=numDisp, blockSize=blockSize, 
                                  P1=p1, P2=p2, disp12MaxDiff=dispMaxDiff, preFilterCap=prefilCap, 
                                  uniquenessRatio=uniqueRatio, speckleWindowSize=speckWinSize, speckleRange=speckRange, mode=cv.STEREO_SGBM_MODE_SGBM_3WAY)
        
        rightMatcher = cv.ximgproc.createRightMatcher(leftMatcher)

        leftDisp = leftMatcher.compute(self.rect_Left, self.rect_Right).astype(np.float32)/16
        rightDisp = rightMatcher.compute(self.rect_Left, self.rect_Right).astype(np.float32)/16

        sigma = 1.5
        lmbda = 8000.0

        # make WLS filter 
        wlsFilter = cv.ximgproc.createDisparityWLSFilter(leftMatcher)
        # wlsFilter = cv.ximgproc.createDisparityWLSFilterGeneric(False)
        wlsFilter.setLambda(lmbda)
        wlsFilter.setSigmaColor(sigma)
        filteredDisparity = wlsFilter.filter(leftDisp, self.left, disparity_map_right=rightDisp)
        # print("disparity image types (disparity, filtered)")
        # print(type(disparity))
        print(type(filteredDisparity))
        print(disparity)

        testName = ('_%s_%i_'%(self.mission, self.id))

        # save depth image
        cv.imwrite(("%s%s_%s.png"%(self.exportPath, "filtered_disparity_image", testName)), filteredDisparity)


        # display images
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(8, 3),
                                            sharex=True, sharey=True)
        for aa in (ax1, ax2, ax3, ax4):
            aa.set_axis_off()
        global debug_show
        if  debug_show:
            ax1.imshow(self.rect_Left, cmap='gray')
            ax1.set_title('Left  image')
            ax2.imshow(self.rect_Right, cmap='gray')
            ax2.set_title('Right image')
            ax3.imshow(disparity, cmap='gray')
            ax3.set_title('Depth map')
            ax4.imshow(filteredDisparity, cmap='jet')
            ax4.set_title('Filtered depthmap (WLS)')
            plt.savefig("%s%s_%s.jpeg"%(exportPath, "comparison", testName))
            plt.show()

        self.disaprity = disparity
        self.wls = filteredDisparity


def load_midas_model(model_type):
    """Load the chosen MiDaS model. Returns loaded transforms, model and device"""
    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    # select gpu if it is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    # load transforms to resize/normalise image depending on the model
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Hybrid" or model_type == "DPT_Hybrid" or model_type == "MiDaS":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    return [transform, midas, device]


def compute_depth_map(transform, midas, device, source_image, file_destination):
    """Compute the depth map of a given image using MiDaS model. Returns depth map as numpy array (numpy.ndarray)"""
    # load in the image and convert colour and apply transforms
    input_image = source_image
    img = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    # predict depth and then resize to the original resolution
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    output = prediction.cpu().numpy()
    output = cv.normalize(output, None, 0, 255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
    cv.imwrite(file_destination, output)

    return output

def blender_specific_depthmap(imgL, imgR, output='./', name=""):
    # imgL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    # imgR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
    minDisp = 10
    numDisp = 96
    blockSize = 11
    dispMaxDiff = 12
    prefilCap = 15
    uniqueRatio = 0
    speckWinSize = 19
    speckRange = 33
    stereo = cv.StereoSGBM_create(
    minDisparity=minDisp, numDisparities=numDisp, blockSize=blockSize, 
                                  P1=100, P2=1000, disp12MaxDiff=dispMaxDiff, preFilterCap=prefilCap, 
                                  uniquenessRatio=uniqueRatio, speckleWindowSize=speckWinSize, speckleRange=speckRange)
    disparity = stereo.compute(imgL,imgR)
    # plt.imshow(disparity)
    cv.imwrite(("%s%s_%s.png"%(output, "disparity_image", name)), disparity)
    return disparity

def show_depth_map(depth_map, imageName):
    """Displays a given depth map in matplotlib"""

    # Display depth as grayscale image
    fig, axs = plt.subplots(1, 1)
    axs.imshow(depth_map, cmap="gray")
    axs.set_title('Depth map')
    plt.savefig(("%s_%s%s"%(exportPath+"depth", imageName, IMAGE_TYPE)), dpi=DPI_COUNT)
    plt.show()

def compare_depth_map(depth_map, source_image, imageName):
    """Displays, in matplotlib, a given depth map and the source it was made from."""

    # Present comparison
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 2),
                                        sharex=True, sharey=True)
    for aa in (ax1, ax2):
        aa.set_axis_off()

    ax1.imshow(source_image)
    ax1.set_title('Source image')
    ax2.imshow(depth_map)
    ax2.set_title('Depth image')

    plt.savefig(("%s_%s%s"%(exportPath+"comp_depth", imageName, IMAGE_TYPE)), dpi=1200)
    plt.show()

def intrinsic_stereo_depthmap_compute(leftImage, rightImage, outputPath = './', name= "test"):
    """Blender intrinsics attempt at making stereo camera work in a simulated environment"""
    # Reading the mapping values for stereo image rectification
    # In blender, the directoy the intrinsic parameters are from is C:\Program Files (x86)\Steam\steamapps\common\Blender\farming-drone
    # cv_file = cv.FileStorage("./active_data/parameters/blender_intrinsics.xml", cv.FILE_STORAGE_READ)
    # Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
    # Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
    # Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
    # Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
    # cv_file.release()

    imgL = cv.imread(leftImage)
    imgR = cv.imread(rightImage)

    imgR_gray = cv.cvtColor(imgR,cv.COLOR_BGR2GRAY)
    imgL_gray = cv.cvtColor(imgL,cv.COLOR_BGR2GRAY)

    # Using SGBM and presets
    # minDisp, numDisp, blockSize, p1, p2, dispMaxDiff, prefilCap, uniqueRatio, speckWinSize, speckRange = depthmap_presets(preset=0)

    # stereo = cv.StereoSGBM_create(minDisparity=minDisp, numDisparities=numDisp, blockSize=blockSize, 
    #                               P1=p1, P2=p2, disp12MaxDiff=dispMaxDiff, preFilterCap=prefilCap, 
    #                               uniquenessRatio=uniqueRatio, speckleWindowSize=speckWinSize, speckleRange=speckRange)
    # disparity = stereo.compute(imgL_gray, imgL_gray)

    # Applying stereo image rectification on the left image
    # Left_nice= cv.remap(imgL_gray,
    #                     Left_Stereo_Map_x,
    #                     Left_Stereo_Map_y,
    #                     cv.INTER_LANCZOS4,
    #                     cv.BORDER_CONSTANT,
    #                     0)
    
    # # Applying stereo image rectification on the right image
    # Right_nice= cv.remap(imgR_gray,
    #                     Right_Stereo_Map_x,
    #                     Right_Stereo_Map_y,
    #                     cv.INTER_LANCZOS4,
    #                     cv.BORDER_CONSTANT,
    #                     0)

    output_destination = outputPath

    depthmap_presets()
    stereo = cv.StereoBM_create(48, 15)
    disparity = stereo.compute(imgL_gray,imgR_gray)

    cv.imwrite(("%s%s_%s.png"%(output_destination, "disparity_image", name)), disparity)


    # display images
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                        sharex=True, sharey=True)
    for aa in (ax1, ax2, ax3):
        aa.set_axis_off()

    ax1.imshow(imgL, cmap='gray')
    ax1.set_title('Left  image')
    ax2.imshow(imgR, cmap='gray')
    ax2.set_title('Right image')
    ax3.imshow(disparity, cmap='gray')
    ax3.set_title('Depth map')
    plt.savefig(("%s%s_%s.jpeg"%(output_destination, "comparison", name)))
    plt.show()

    return disparity
    
def stereo_depthmap_compute_from_path(leftPath="", rightPath="", outputPath="./", name="test"):

    imgL = cv.imread(leftPath, cv.IMREAD_GRAYSCALE)
    imgR = cv.imread(rightPath, cv.IMREAD_GRAYSCALE) 
    # imgL = cv.imread('./active_data/calibration_image_set/stereo_set/left/1679348001595_L.jpeg', cv.IMREAD_GRAYSCALE)
    # imgR = cv.imread('./active_data/calibration_image_set/stereo_set/right/1679348001680_R.jpeg', cv.IMREAD_GRAYSCALE)
    
    # Reading the mapping values for stereo image rectification
    cv_file = cv.FileStorage("./STEREO_PARAMS.xml", cv.FILE_STORAGE_READ)
    stereoMapL_x = cv_file.getNode("stereoMapL_x").mat()
    stereoMapL_y = cv_file.getNode("stereoMapL_y").mat()
    stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
    stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
    cv_file.release()

    # Applying stereo image rectification on the left image
    Left_nice= cv.remap(imgL,
            stereoMapL_x,
            stereoMapL_y,
            cv.INTER_LANCZOS4,
            cv.BORDER_CONSTANT,
            0)
    
    # Applying stereo image rectification on the right image
    Right_nice= cv.remap(imgR,
            stereoMapR_x,
            stereoMapR_y,
            cv.INTER_LANCZOS4,
            cv.BORDER_CONSTANT,
            0)

    # numDisp = 48
    # blockSize = 25
    # prefilCap = 63
    # uniqueRatio = 0
    # speckleRange = 49
    # speckleWinSize = 10
    # dispMaxDiff = 20
    # minDisp = 0


    # stereo = cv.StereoSGBM_create(numDisparities=numDisp, blockSize=blockSize, preFilterCap=prefilCap, speckleRange=speckleRange, speckleWindowSize=speckleWinSize, uniquenessRatio=uniqueRatio, disp12MaxDiff=dispMaxDiff, minDisparity=minDisp)
    # disparity = stereo.compute(Left_nice, Right_nice)

    # leftMatcher = cv.StereoSGBM_create(numDisparities=numDisp, blockSize=blockSize, preFilterCap=prefilCap, speckleRange=speckleRange, speckleWindowSize=speckleWinSize,uniquenessRatio=uniqueRatio, disp12MaxDiff=dispMaxDiff, minDisparity=minDisp)
    # rightMatcher = cv.ximgproc.createRightMatcher(leftMatcher)

    minDisp = 10
    numDisp = 96
    blockSize = 11
    p1 = 100
    p2 = 1000
    dispMaxDiff = 12
    prefilCap = 15
    uniqueRatio = 0
    speckWinSize = 19
    speckRange = 33

    stereo = cv.StereoSGBM_create(minDisparity=minDisp, numDisparities=numDisp, blockSize=blockSize, 
                                  P1=p1, P2=p2, disp12MaxDiff=dispMaxDiff, preFilterCap=prefilCap, 
                                  uniquenessRatio=uniqueRatio, speckleWindowSize=speckWinSize, speckleRange=speckRange)
    disparity = stereo.compute(Left_nice, Right_nice)

    leftMatcher = cv.StereoSGBM_create(minDisparity=minDisp, numDisparities=numDisp, blockSize=blockSize, 
                                  P1=p1, P2=p2, disp12MaxDiff=dispMaxDiff, preFilterCap=prefilCap, 
                                  uniquenessRatio=uniqueRatio, speckleWindowSize=speckWinSize, speckleRange=speckRange)

    # stereo = cv.StereoSGBM_create(minDisparity=10, numDisparities=85, blockSize=11)
    # In this case, disparity will be multiplied by 16 internally! Divide by 16 to get real value.
    # disparity = stereo.compute(Left_nice, Right_nice).astype(np.float32)/16

    # stereo = cv.StereoBM_create(numDisparities=96, blockSize=15)
    # disparity = stereo.compute(Left_nice, Right_nice)

    # leftMatcher = cv.StereoBM_create(numDisparities=96, blockSize=15)
    rightMatcher = cv.ximgproc.createRightMatcher(leftMatcher)

    leftDisp = leftMatcher.compute(Left_nice, Right_nice).astype(np.float32)/16
    rightDisp = rightMatcher.compute(Right_nice, Left_nice).astype(np.float32)/16

    sigma = 1.5
    lmbda = 8000.0

    # make WLS filter 
    wlsFilter = cv.ximgproc.createDisparityWLSFilter(leftMatcher)
    # wlsFilter = cv.ximgproc.createDisparityWLSFilterGeneric(False)
    wlsFilter.setLambda(lmbda)
    wlsFilter.setSigmaColor(sigma)
    filteredDisparity = wlsFilter.filter(leftDisp,imgL, disparity_map_right=rightDisp)
    print("disparity image types (disparity, filtered)")
    print(type(disparity))
    print(type(filteredDisparity))

    testName = name
    # save depth image
    cv.imwrite(("%s%s_%s.png"%(outputPath, "filtered_disparity_image", testName)), filteredDisparity)


    # display images
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(8, 3),
                                        sharex=True, sharey=True)
    for aa in (ax1, ax2, ax3, ax4):
        aa.set_axis_off()


    ax1.imshow(Left_nice, cmap='gray')
    ax1.set_title('Left  image')
    ax2.imshow(Right_nice, cmap='gray')
    ax2.set_title('Right image')
    ax3.imshow(disparity, cmap='gray')
    ax3.set_title('Depth map')
    ax4.imshow(filteredDisparity, cmap='jet')
    ax4.set_title('Filtered depthmap (WLS)')
    plt.savefig("%s%s_%s.jpeg"%(outputPath, "comparison", testName))
    plt.show()

    return disparity, filteredDisparity

def stereo_depthmap_compute(leftImage, rightImage, outputPath="./", name="test", debug_show=True, preset=0):

    grayL = leftImage#cv.cvtColor(leftImage, cv.COLOR_BGR2GRAY)
    grayR = rightImage#cv.cvtColor(rightImage, cv.COLOR_BGR2GRAY)
    # imgL = cv.imread('./active_data/calibration_image_set/stereo_set/left/1679348001595_L.jpeg', cv.IMREAD_GRAYSCALE)
    # imgR = cv.imread('./active_data/calibration_image_set/stereo_set/right/1679348001680_R.jpeg', cv.IMREAD_GRAYSCALE)
    
    # Reading the mapping values for stereo image rectification
    cv_file = cv.FileStorage("./STEREO_PARAMS.xml", cv.FILE_STORAGE_READ)
    stereoMapL_x = cv_file.getNode("stereoMapL_x").mat()
    stereoMapL_y = cv_file.getNode("stereoMapL_y").mat()
    stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
    stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
    cv_file.release()

    # Applying stereo image rectification on the left image
    Left_nice= cv.remap(grayL,
            stereoMapL_x,
            stereoMapL_y,
            cv.INTER_LANCZOS4,
            cv.BORDER_CONSTANT,
            0)
    
    # Applying stereo image rectification on the right image
    Right_nice= cv.remap(grayR,
            stereoMapR_x,
            stereoMapR_y,
            cv.INTER_LANCZOS4,
            cv.BORDER_CONSTANT,
            0)

    # numDisp = 48
    # blockSize = 25
    # prefilCap = 63
    # uniqueRatio = 0
    # speckleRange = 49
    # speckleWinSize = 10
    # dispMaxDiff = 20
    # minDisp = 0


    # stereo = cv.StereoSGBM_create(numDisparities=numDisp, blockSize=blockSize, preFilterCap=prefilCap, speckleRange=speckleRange, speckleWindowSize=speckleWinSize, uniquenessRatio=uniqueRatio, disp12MaxDiff=dispMaxDiff, minDisparity=minDisp)
    # disparity = stereo.compute(Left_nice, Right_nice)

    # leftMatcher = cv.StereoSGBM_create(numDisparities=numDisp, blockSize=blockSize, preFilterCap=prefilCap, speckleRange=speckleRange, speckleWindowSize=speckleWinSize,uniquenessRatio=uniqueRatio, disp12MaxDiff=dispMaxDiff, minDisparity=minDisp)
    # rightMatcher = cv.ximgproc.createRightMatcher(leftMatcher)


    minDisp, numDisp, blockSize, p1, p2, dispMaxDiff, prefilCap, uniqueRatio, speckWinSize, speckRange = depthmap_presets(preset=preset)
    # minDisp = 10
    # numDisp = 96
    # blockSize = 11
    # p1 = 100
    # p2 = 1000
    # dispMaxDiff = 12
    # prefilCap = 15
    # uniqueRatio = 0
    # speckWinSize = 19
    # speckRange = 33

    stereo = cv.StereoSGBM_create(minDisparity=minDisp, numDisparities=numDisp, blockSize=blockSize, 
                                  P1=p1, P2=p2, disp12MaxDiff=dispMaxDiff, preFilterCap=prefilCap, 
                                  uniquenessRatio=uniqueRatio, speckleWindowSize=speckWinSize, speckleRange=speckRange)
    disparity = stereo.compute(Left_nice, Right_nice)

    leftMatcher = cv.StereoSGBM_create(minDisparity=minDisp, numDisparities=numDisp, blockSize=blockSize, 
                                  P1=p1, P2=p2, disp12MaxDiff=dispMaxDiff, preFilterCap=prefilCap, 
                                  uniquenessRatio=uniqueRatio, speckleWindowSize=speckWinSize, speckleRange=speckRange, mode=cv.STEREO_SGBM_MODE_SGBM_3WAY)

    # stereo = cv.StereoSGBM_create(minDisparity=10, numDisparities=85, blockSize=11)
    # In this case, disparity will be multiplied by 16 internally! Divide by 16 to get real value.
    # disparity = stereo.compute(Left_nice, Right_nice).astype(np.float32)/16

    # stereo = cv.StereoBM_create(numDisparities=96, blockSize=15)
    # disparity = stereo.compute(Left_nice, Right_nice)

    # leftMatcher = cv.StereoBM_create(numDisparities=96, blockSize=15)
    rightMatcher = cv.ximgproc.createRightMatcher(leftMatcher)

    leftDisp = leftMatcher.compute(Left_nice, Right_nice).astype(np.float32)/16
    rightDisp = rightMatcher.compute(Right_nice, Left_nice).astype(np.float32)/16

    sigma = 1.5
    lmbda = 8000.0

    # make WLS filter 
    wlsFilter = cv.ximgproc.createDisparityWLSFilter(leftMatcher)
    # wlsFilter = cv.ximgproc.createDisparityWLSFilterGeneric(False)
    wlsFilter.setLambda(lmbda)
    wlsFilter.setSigmaColor(sigma)
    filteredDisparity = wlsFilter.filter(leftDisp,leftImage, disparity_map_right=rightDisp)
    print("disparity image types (disparity, filtered)")
    print(type(disparity))
    print(type(filteredDisparity))

    testName = name
    # save depth image
    cv.imwrite(("%s%s_%s.png"%(outputPath, "filtered_disparity_image", testName)), filteredDisparity)


    # display images
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(8, 3),
                                        sharex=True, sharey=True)
    for aa in (ax1, ax2, ax3, ax4):
        aa.set_axis_off()

    if debug_show:
        ax1.imshow(Left_nice, cmap='gray')
        ax1.set_title('Left  image')
        ax2.imshow(Right_nice, cmap='gray')
        ax2.set_title('Right image')
        ax3.imshow(disparity, cmap='gray')
        ax3.set_title('Depth map')
        ax4.imshow(filteredDisparity, cmap='jet')
        ax4.set_title('Filtered depthmap (WLS)')
        plt.savefig("%s%s_%s.jpeg"%(outputPath, "comparison", testName))
        plt.show()

    return disparity, filteredDisparity

def depthmap_presets(preset=0):
    match preset:
        case 0: # base
             minDisp = 10
             numDisp = 96
             blockSize = 11
             p1 = 100
             p2 = 1000
             dispMaxDiff = 12
             prefilCap = 15
             uniqueRatio = 0
             speckWinSize = 19
             speckRange = 33

        case 1: # garden
             minDisp = 17
             numDisp = 256
             blockSize = 12
             p1 = 100
             p2 = 1000
             dispMaxDiff = 20
             prefilCap = 20
             uniqueRatio = 3
             speckWinSize = 29
             speckRange = 44
        
        case 2:
             minDisp = 10
             numDisp = 256
             blockSize = 9
             p1 = 100
             p2 = 800
             dispMaxDiff = 9
             prefilCap = 20
             uniqueRatio = 1
             speckWinSize = 25
             speckRange = 40

        case 3: 
             minDisp=0
             numDisp=16
             blockSize=5
             p1=8 * 3 * 15 ** 2
             p2=32 * 3 * 15 ** 2
             dispMaxDiff = 12
             prefilCap = 15
             uniqueRatio = 0
             speckWinSize = 19
             speckRange = 33

        case 4:
            numDisp = 48
            blockSize = 7
            prefilCap = 6
            uniqueRatio = 1
            speckRange = 0
            speckWinSize = 6
            dispMaxDiff = 9
            minDisp = 5
            p1 = 100
            p2 = 1000

        case _: # base case
             minDisp = 10
             numDisp = 96
             blockSize = 11
             p1 = 100
             p2 = 1000
             dispMaxDiff = 12
             prefilCap = 15
             uniqueRatio = 0
             speckWinSize = 19
             speckRange = 33

    return minDisp, numDisp, blockSize, p1, p2, dispMaxDiff, prefilCap, uniqueRatio, speckWinSize, speckRange


        


def compute_stereo_mission_folder_to_image_and_depth(missionCode, path=IMAGE_DESTINATION, wls=True, sgbm_preset=0):

    # find mission code folder

    # split stereo image sinto left and right list
    imagesL = glob.glob("%smissions/%s/stereo/%s"%(path, missionCode, '*L.jpeg'), recursive = False)
    imagesR = glob.glob("%smissions/%s/stereo/%s"%(path, missionCode, '*R.jpeg'), recursive = False)

    # output paths
    imageFolder = "%smissions/%s/image/"%(path, missionCode)
    depthFolder = "%smissions/%s/depth/"%(path, missionCode)

    image_list = []
    depth_list = []

    # go through all images in folder and make depthmaps and images for them, save to respective folders
    for imageL,imageR in zip(imagesL, imagesR):
        # save left frame as image
        filenameL_ext = os.path.basename(imageL)
        # print(filename_ext)
        currentImageL = cv.imread(imageL)
        # cv.imshow(filename_ext, currentImageL)
        cv.imwrite(("%s%s"%(imageFolder, filenameL_ext)), currentImageL)
        image_list.append(currentImageL)

        # compute depthmap of the current left and right images
        # show current left image
        filenameR_ext = os.path.basename(imageR)
        currentImageR = cv.imread(imageR)
        # cv.imshow(filename_ext, currentImageR)
        # make depth map
        disparity, filteredDisparity = stereo_depthmap_compute(leftImage=currentImageL, rightImage=currentImageR, debug_show=False, preset=sgbm_preset)
        print("Generating depth map of %s"%(filenameL_ext.replace('.jpeg', '')))
        if wls:
            cv.imwrite(("%s%s"%(depthFolder, filenameL_ext.replace('_L.jpeg', '.png'))), filteredDisparity)
            depth_list.append(filteredDisparity)
        else:
            cv.imwrite(("%s%s"%(depthFolder, filenameL_ext.replace('_L.jpeg', '.png'))), disparity)
            depth_list.append(disparity)
        
    return image_list, depth_list
        


    

        

if __name__ == "__main__":
    # stereo_depthmap_compute(leftPath='./active_data/images/woodland_bikepark_L.jpeg', rightPath='./active_data/images/woodland_bikepark_R.jpeg', outputPath="./active_data/", name="test4")
    # stereo_depthmap_compute(leftPath='./active_data/images/guitar_L.jpeg', rightPath='./active_data/images/guitar_R.jpeg', outputPath="./active_data/", name="test5")
    # compute_stereo_mission_folder_to_image_and_depth(missionCode="fins room", path="./active_data/", sgbm_preset=0)
    # intrinsic_stereo_depthmap_compute()
    srcL = ("./blender/older_images/left2.png")
    srcR = ("./blender/older_images/right2.png")
    # srcL = cv.imread("./blender/older_images/left2.png")
    # srcR = cv.imread("./blender/older_images/right2.png")

    # blender_specific_depthmap(srcL, srcR, "./outputs/depthmaps/", name="blender-og")
    intrinsic_stereo_depthmap_compute(srcL, srcR, "./outputs/depthmaps/", "blender-og")
    d = Depthmap(1,"", "./outputs/depthmaps", "blender-og", 2, srcL, srcR)
    d.getStereoParams()
    d.matrices.getFile()
    d.matrices.makeMatrices()
    print(d.matrices.stereoMapL_x)

    d.generate_depthmap()
    

    # stereo_depthmap_compute(leftPath='./active_data/images/tree1_L.jpeg', rightPath='./active_data/images/tree1_R.jpeg', outputPath="./active_data/", name="test6")
    # stereo_depthmap_compute(leftPath='./active_data/images/tree2_L.jpeg', rightPath='./active_data/images/tree2_R.jpeg', outputPath="./active_data/", name="test7")
    # stereo_depthmap_compute(leftPath='./active_data/images/guitar_L.jpeg', rightPath='./active_data/images/guitar_R.jpeg', outputPath="./active_data/", name="test8")

#   load_model =  load_midas_model(model_type)
#   transform = load_model[0]
#   midas = load_model[1]
#   device = load_model[2]

#   depth_map = compute_depth_map(transform, midas, device, src1)

#   show_depth_map(depth_map, "right")

#   depth_map2 = compute_depth_map(transform, midas, device, src2)
#   show_depth_map(depth_map2, "left")

#   compare_depth_map(depth_map, depth_map2, "depths")
