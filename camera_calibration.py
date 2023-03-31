import time
import cv2 as cv
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
# import distortion_removal
import sys
# sys.path.insert(1, './stereo_camera/')
# from stereo_camera import getCurrentMS
from config import *


def stereoComputeCalibration(left_calibration_set="", right_calibration_set=""):
    '''
    Using supplied images, returns a list containing the camera calibration values (matrix, distance, rotation vectors, translation vectors)
    '''
    imgL = 0 
    grayL = 0
    imgR = 0 
    grayR = 0
    
    # Defining the dimensions of checkerboard (defined by the inner corners of the outer board!)
    # CHECKERBOARD = (7,9)
    frameSize = FRAME_SIZE
    # stopping criterion
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpointsL = []
    imgpointsR = [] 
    
    
    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    objp = objp*20 # 20mm squares on the chessboard

    # get all images in left and right image set folders
    imagesLeft = glob.glob("./active_data/calibration_image_set/stereo_set/left/*.jpeg")
    imagesRight = glob.glob("./active_data/calibration_image_set/stereo_set/right/*.jpeg")

    #  go through both folders and perform chessboard calibration on each image
    for imgLeft,imgRight in zip(imagesLeft, imagesRight):

        imgL = cv.imread(imgLeft)
        imgR = cv.imread(imgRight)
        grayL = cv.cvtColor(imgL,cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(imgR,cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        retL, cornersL = cv.findChessboardCorners(grayL, CHECKERBOARD, None)
        retR, cornersR = cv.findChessboardCorners(grayR, CHECKERBOARD, None)
        
        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if retL and retR == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            cornersL = cv.cornerSubPix(grayL, cornersL, (11,11),(-1,-1), criteria)
            imgpointsL.append(cornersL)

            cornersR = cv.cornerSubPix(grayR, cornersR, (11,11),(-1,-1), criteria)
            imgpointsR.append(cornersR)
    
            # Draw and display the corners
            
            cv.drawChessboardCorners(imgL, CHECKERBOARD, cornersL, retL)
            cv.imshow('left image',imgL)
            cv.drawChessboardCorners(imgR, CHECKERBOARD, cornersR, retR)
            cv.imshow('right image',imgR)
            cv.waitKey(1000)
    
    cv.destroyAllWindows()
    return objpoints, imgpointsL, imgL, grayL, imgpointsR, imgR, grayR

def useMatlabCalibration():
    newCameraMatrixL=np.asarray([[1.7743,0,0.9508],
                                 [0,1.7651,0.5343],
                                 [0,0,0.0010]])
    distL = np.asarray([[-0.1436,0.8213,0,0,0]])

    newCameraMatrixR=np.asarray([[1.7716,0,0.9076],
                                 [0,1.7651,0.5847],
                                 [0,0,0.0010]])
    distR = np.asarray([[-0.0907,0.4179,0,0,0]])

    rot = np.asarray([[1.0000,-0.0086,0.0047],
                      [0.0088,0.9994,-0.0350],
                      [-0.0044,0.0350,0.9994]])
    trans = np.asarray([[-55.2852],
                        [-1.1765],
                        [5.2840]])

    print(newCameraMatrixL)
    print(distL)
    print(newCameraMatrixR)
    print(distR)
    print(rot)
    print(trans)

    return newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans


def calibrateStereo(objpoints, imgpointsL, imgL, grayL, imgpointsR, imgR, grayR):

    retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, (2592,1944), None, None)
    heightL, widthL, channelsL = imgL.shape
    newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

    retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, (2592,1944), None, None)
    heightR, widthR, channelsR = imgR.shape
    newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))

    flags = 0
    flags |= cv.CALIB_USE_INTRINSIC_GUESS#cv.CALIB_FIX_INTRINSIC # Fix intrinsics so only rot, trans emat and fmat calculated

    criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans = useMatlabCalibration()
    retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], criteria= criteria_stereo, flags=flags)

    # newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans = useMatlabCalibration()

    # intrinsicMatrix1,distortionCoefficients1,intrinsicMatrix2,distortionCoefficients2,rotationOfCamera2,translationOfCamera2,imageSize
    print("--Extrinsics:--")
    print("Left matrix")
    print(newCameraMatrixL)
    print(distL)
    print("Left matrix")
    print(newCameraMatrixR)
    print(distR)

    print("\n--Intrinsics:--")
    print("Rotation:")
    print(type(rot))
    print("Translation")
    print(type(trans))

    
    
    # print("MATLAB PARAMS")
    # newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans = useMatlabCalibration()

    # rectify stereo images

    rectifyScale = 1
    # get the rectification for left and right cameras, aligning the y and c axis so we can find the true disparity
    rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, (0,0), alpha=0.9)

    stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
    stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)

    print("Saving stereo parameters")
    cv_file = cv.FileStorage("./STEREO_PARAMS.xml", cv.FILE_STORAGE_WRITE)
    cv_file.write('stereoMapL_x',stereoMapL[0])
    cv_file.write('stereoMapL_y',stereoMapL[1])
    cv_file.write('stereoMapR_x',stereoMapR[0])
    cv_file.write('stereoMapR_y',stereoMapR[1])  
    cv_file.release()

    print("Saving camera instrinsics")
    print(newCameraMatrixL)
    cv_file = cv.FileStorage("./INTRINSICS.xml", cv.FILE_STORAGE_WRITE)
    cv_file.write('mx', newCameraMatrixL) # mx = [[fx, 0 , cx], [0, fy, cy], [0, 0 ,1]]
    cv_file.release()

    # We have to convert R and T to the Extrinsic Matrix format written as 4x4 array to be used later in pointcloud calculation 
    # T values 1-3 are inserted at the end of each row of R
    # like this [R11,R12,R13,T1], [R21,R22,R23,T2], [R31,R32,R33,T3]
    extrinsics = rot
    extrinsics = np.concatenate((extrinsics, trans), 1) # axis set to column ways so we can just put each T value on the end
    extrinsics = np.concatenate((extrinsics, [[0.0, 0.0, 0.0, 1.0]]), 0) # add the 

    print("Saving camera extrinsics")
    print(extrinsics)
    cv_file = cv.FileStorage("./EXTRINSICS.xml", cv.FILE_STORAGE_WRITE)
    cv_file.write('mx', extrinsics)
    cv_file.release()

def calibrateCamera(images):
    
    '''
    Using supplied images, returns a list containing the camera calibration values (matrix, distance, rotation vectors, translation vectors)
    '''
    # Defining the dimensions of checkerboard (defined by the inner corners of the outer board!)
    CHECKERBOARD = (7,9)
    frameSize = (1920,1080)
    # stopping criterion
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = [] 
    
    
    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    objp = objp*20 # 20mm squares on the chessboard

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
        
        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            
            imgpoints.append(corners2)
    
            # Draw and display the corners
            img = cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
     
        cv.imshow('img',img)
        cv.waitKey(0)
    cv.destroyAllWindows()
    
    h,w = img.shape[:2]
    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    new_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1,(w,h))

    # print("Camera matrix : \n")
    # print(mtx)
    # print("dist : \n")
    # print(dist)
    # print("rvecs : \n")
    # print(rvecs)
    # print("tvecs : \n")
    # print(tvecs)

    # values = [h, w, mtx, dist, rvecs, tvecs]
    values = [objpoints, imgpoints, new_mtx, roi, dist, gray]
    return values


def saveCalibrationValues(name, image_directory, destination):
    chessboard_images = glob.glob(image_directory)
    values = calibrateCamera(chessboard_images)
    file = open('%s%s.txt'%(destination,name), 'w')
    for value in values:
        file.write(str(value)+"\n")
    file.close()

def plot_images(image1, image2):
    plt.subplot(1, 2, 1)
    plt.title('A')
    plt.imshow(image1)
    plt.subplot(1, 2, 2)
    plt.title('B')
    plt.imshow(image2)
    plt.show()

def stereo_depthmap_compute(left_image, right_image, outputPath="./", name="test"):

    imgL = left_image # cv.imread('./active_data/images/stereo_L.jpeg', cv.IMREAD_GRAYSCALE)
    imgR = right_image # cv.imread('./active_data/images/stereo_R.jpeg', cv.IMREAD_GRAYSCALE)

    stereo = cv.StereoBM_create(numDisparities=48, blockSize=25)
    disparity = stereo.compute(imgL,imgR)

    testName = "irl_stereo"
    # save depth image
    cv.imwrite(("%s%s_%s.png"%(outputPath, "disparity_image", testName)), disparity)


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
    plt.savefig("%s%s_%s.jpeg"%(outputPath, "comparison", testName))
    plt.show()

if __name__ == "__main__":

    objpoints, imgpointsL, imgL, grayL, imgpointsR, imgR, grayR = stereoComputeCalibration()

    calibrateStereo(objpoints, imgpointsL, imgL, grayL, imgpointsR, imgR, grayR)

    cv_file = cv.FileStorage()
    cv_file.open('./STEREO_PARAMS.xml', cv.FileStorage_READ)
    stereoMapL_x = cv_file.getNode("stereoMapL_x").mat()
    stereoMapL_y = cv_file.getNode("stereoMapL_y").mat()
    stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
    stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

    current_L = cv.imread('./active_data/images/guitar_L.jpeg')
    current_R = cv.imread('./active_data/images/guitar_R.jpeg')

    current_L_corrected = cv.remap(src=current_L, map1=stereoMapL_x, map2=stereoMapL_y, interpolation=cv.INTER_LANCZOS4, borderMode=cv.BORDER_CONSTANT, borderValue=0)
    current_R_corrected = cv.remap(src=current_R, map1=stereoMapR_x, map2=stereoMapR_y, interpolation=cv.INTER_LANCZOS4, borderMode=cv.BORDER_CONSTANT, borderValue=0)

    plot_images(current_L, current_L_corrected)
    plot_images(current_R, current_R_corrected)
    plot_images(current_L_corrected, current_R_corrected)
    
    cv.imwrite(("%s.png"%("./stereo_rectified_L")), current_L_corrected)
    cv.imwrite(("%s.png"%("./stereo_rectified_R")), current_R_corrected)

    l = cv.imread('./stereo_rectified_L.png', cv.IMREAD_GRAYSCALE)
    r = cv.imread('./stereo_rectified_R.png', cv.IMREAD_GRAYSCALE)

    stereo_depthmap_compute(l, r, "./", "stereo_rect_test01")
    # cv.imshow("Left corrected", current_L_corrected)
    # cv.imshow("Right corrected", current_R~_corrected)
    cv.destroyAllWindows()




