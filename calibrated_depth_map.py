import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import sys

# Reading the mapping values for stereo image rectification
cv_file = cv.FileStorage("./STEREO_PARAMS.xml", cv.FILE_STORAGE_READ)
stereoMapL_x = cv_file.getNode("stereoMapL_x").mat()
stereoMapL_y = cv_file.getNode("stereoMapL_y").mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
cv_file.release()

output_destination = "./depth_map/output/"

# images we want to get depth from
imgL = cv.imread('./active_data/missions/garden/stereo/1679849953746_L.jpeg', cv.IMREAD_COLOR)
imgR = cv.imread('./active_data/missions/garden/stereo/1679849953835_R.jpeg', cv.IMREAD_COLOR)

def nothing(x):
    pass

# Depth map adjustment window
cv.namedWindow('Disparity options',cv.WINDOW_FREERATIO)
cv.resizeWindow('Disparity options',640,480)
 
cv.createTrackbar('numDisparities','Disparity options',1,17,nothing)
cv.createTrackbar('blockSize','Disparity options',5,50,nothing)
# cv.createTrackbar('preFilterType','Disparity options',1,1,nothing)
# cv.createTrackbar('preFilterSize','Disparity options',2,25,nothing)
cv.createTrackbar('preFilterCap','Disparity options',5,62,nothing)
# cv.createTrackbar('textureThreshold','Disparity options',10,100,nothing)
cv.createTrackbar('uniquenessRatio','Disparity options',15,100,nothing)
cv.createTrackbar('speckleRange','Disparity options',0,100,nothing)
cv.createTrackbar('speckleWindowSize','Disparity options',3,25,nothing)
cv.createTrackbar('disp12MaxDiff','Disparity options',5,25,nothing)
cv.createTrackbar('minDisparity','Disparity options',5,25,nothing)

cv.namedWindow('Disparity map',cv.WINDOW_AUTOSIZE)
cv.resizeWindow('Disparity map',1920,1080)

stereo = cv.StereoSGBM_create(numDisparities=48, blockSize=15, P1=8 * 3 * 5**2, P2=32 * 3 * 5**2)

plt.imshow(imgL)
plt.show()
while True:
    # if retL and retR:
    if True:

        imgR_gray = imgL #cv.cvtColor(imgR,cv.COLOR_BGR2GRAY)
        imgL_gray = imgR #cv.cvtColor(imgL,cv.COLOR_BGR2GRAY)
    
        # Applying stereo image rectification on the left image
        Left_nice= cv.remap(imgL_gray,
                stereoMapL_x,
                stereoMapL_y,
                cv.INTER_LANCZOS4,
                cv.BORDER_CONSTANT,
                0)
        
        # Applying stereo image rectification on the right image
        Right_nice= cv.remap(imgR_gray,
                stereoMapR_x,
                stereoMapR_y,
                cv.INTER_LANCZOS4,
                cv.BORDER_CONSTANT,
                0)
    
        # Updating the parameters based on the trackbar positions
        numDisparities = cv.getTrackbarPos('numDisparities','Disparity options')*16
        blockSize = cv.getTrackbarPos('blockSize','Disparity options')*2 + 5
        # preFilterType = cv.getTrackbarPos('preFilterType','Disparity options')
        # preFilterSize = cv.getTrackbarPos('preFilterSize','Disparity options')*2 + 5
        preFilterCap = cv.getTrackbarPos('preFilterCap','Disparity options')
        # textureThreshold = cv.getTrackbarPos('textureThreshold','Disparity options')
        uniquenessRatio = cv.getTrackbarPos('uniquenessRatio','Disparity options')
        speckleRange = cv.getTrackbarPos('speckleRange','Disparity options')
        speckleWindowSize = cv.getTrackbarPos('speckleWindowSize','Disparity options')*2
        disp12MaxDiff = cv.getTrackbarPos('disp12MaxDiff','Disparity options')
        minDisparity = cv.getTrackbarPos('minDisparity','Disparity options')
        
        # Setting the updated parameters before computing disparity map
        stereo.setNumDisparities(numDisparities)
        stereo.setBlockSize(blockSize)
        # stereo.setPreFilterType(preFilterType)
        # stereo.setPreFilterSize(preFilterSize)
        stereo.setPreFilterCap(preFilterCap)
        # stereo.setTextureThreshold(textureThreshold)
        stereo.setUniquenessRatio(uniquenessRatio)
        stereo.setSpeckleRange(speckleRange)
        stereo.setSpeckleWindowSize(speckleWindowSize)
        stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setMinDisparity(minDisparity)

    
        # Calculating disparity using the StereoBM algorithm
        disparity = stereo.compute(imgL_gray, imgR_gray)
        # NOTE: Code returns a 16bit signed single channel image,
        # CV_16S containing a disparity map scaled by 16. Hence it 
        # is essential to convert it to CV_32F and scale it down 16 times.
    
        # Converting to float32 
        disparity = disparity.astype(np.float32)
    
        # Scaling down the disparity values and normalizing them 
        disparity = (disparity/16.0 - minDisparity)/numDisparities
        # Displaying the disparity map
        cv.imshow('Disparity map',disparity)

        if cv.waitKey(1) == 99:
            print("numDisp = %s\nblockSize = %s\nprefilCap = %s\nuniqueRatio = %s\nspeckRange = %s\nspeckWinSize = %s\ndispMaxDiff = %s\nminDisp = %s"%(
                numDisparities,
                blockSize,
                preFilterCap,
                uniquenessRatio,
                speckleRange,
                speckleWindowSize,
                disp12MaxDiff,
                minDisparity)
            )
        if cv.waitKey(1) == 115:
            cv.imwrite("houseplant_depthmap.png", disparity)
        # Close window using esc key
        if cv.waitKey(1) == 27:
            break
    else:
        print("Something went wrong!")











# disparity = stereo.compute(imgL,imgR)

# # save depth image
# cv.imwrite(("%s%s_%s.jpeg"%(output_destination, "disparity_image", getCurrentMS())), disparity)


# # display images
# fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
#                                     sharex=True, sharey=True)
# for aa in (ax1, ax2, ax3):
#     aa.set_axis_off()

# ax1.imshow(imgL, cmap='gray')
# ax1.set_title('Left  image')
# ax2.imshow(imgR, cmap='gray')
# ax2.set_title('Right image')
# ax3.imshow(disparity, cmap='gray')
# ax3.set_title('Depth map')
# plt.savefig(("%s%s_%s.jpeg"%(output_destination, "comparison", getCurrentMS())))
# plt.show()
