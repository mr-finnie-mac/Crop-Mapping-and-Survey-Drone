import numpy as np
import cv2 as cv
import glob
from config import *



# Chessboard calibration technique for finding camera (int/ext)rinsic parameters
def make_calibration_params():
    "Will update active intrinsic values in active_data directory"
    # define chessboard
    chessboardSize = (7,9)
    frameSize = (1920,1080)

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

    size_of_chessboard_squares_mm = 25
    objp = objp * size_of_chessboard_squares_mm


    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.


    images = glob.glob(CALIBRATION_IMAGES + '*.jpeg')

    for image in images:

        img = cv.imread(image)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

        # If found, add object points, image points (after refining them)
        if ret == True:

            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(500)


    cv.destroyAllWindows()


    # Make calibration matrix (intrinsic)

    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
    h,  w = img.shape[:2]
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

    print("Updating intrinsic parameters at " + CURRENT_INTRINSICS)
    cv_file = cv.FileStorage(CURRENT_INTRINSICS, cv.FILE_STORAGE_WRITE)
    cv_file.write('intrinsic', newCameraMatrix)
    print(type(newCameraMatrix))
    cv_file.release()

# def stereo_calibration():


# Chessboard calibration technique for finding camera (int/ext)rinsic parameters
def make_new_calibration_params(source_folder, savefile_name, destination):
    "Make new intrinsic values in provided directory"
    # define chessboard
    chessboardSize = (7,9)
    frameSize = (1920,1080)

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

    size_of_chessboard_squares_mm = 25
    objp = objp * size_of_chessboard_squares_mm


    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.


    images = glob.glob(source_folder + '*.jpeg')

    for image in images:

        img = cv.imread(image)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

        # If found, add object points, image points (after refining them)
        if ret == True:

            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(500)


    cv.destroyAllWindows()


    # Make calibration matrix (intrinsic)

    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
    h,  w = img.shape[:2]
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

    print("Saving new intrinsic parameters at " + destination)
    cv_file = cv.FileStorage(destination + savefile_name, cv.FILE_STORAGE_WRITE)
    cv_file.write('intrinsic', newCameraMatrix)
    print(type(newCameraMatrix))
    cv_file.release()


# Remove distortion

# print(cameraMatrix)
# img = cv.imread('./img_sets/field_dataset/images/1677068524771_L.jpeg')
# h,  w = img.shape[:2]


# print(newCameraMatrix)
# with open('intrinsicNew.xml', 'wb') as f:
#     np.save(f, newCameraMatrix)

# # Undistort
# dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
# # crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv.imwrite('caliResult1.png', dst)

# # Undistort with Remapping
# mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
# dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# # crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv.imwrite('caliResult2.png', dst)




# # Reprojection Error
# mean_error = 0

# for i in range(len(objpoints)):
#     imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
#     error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
#     mean_error += error

# print( "total error: {}".format(mean_error/len(objpoints)) )

# if __name__ == "__main__":
#     make_calibration_params()