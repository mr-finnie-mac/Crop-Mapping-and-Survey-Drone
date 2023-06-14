from camera import *
from depthmap_navqplus import stereo_depthmap_compute, multithread_depthmap

if __name__ == '__main__':
    # make camera objects for left and right then print their configs
    left_camera = Camera(LEFT_ID, LEFT_DEVICE_ADDRESS, LEFT_WIDTH, LEFT_HEIGHT, LEFT_FRAMERATE, TARGET_DIR)
    right_camera = Camera(RIGHT_ID, RIGHT_DEVICE_ADDRESS, RIGHT_WIDTH, RIGHT_HEIGHT, RIGHT_FRAMERATE, TARGET_DIR)
    # print the current config of left and right cameras
    left_camera.cameraInfo()
    right_camera.cameraInfo()
    while True:
        # Take a single image with stereo camera
        runStereoCamera(left_camera,right_camera)
        left, right = captureStereo(left_camera, right_camera, imageDestination="./stereo_camera/demo_version/v1/static/")

        # stereo_depthmap_compute(leftImage=left, rightImage=right, preset=1, wls=True, outputPath="./stereo_camera/demo_version/v1/static/depth/")
        
        # multithread_depthmap(image_left=left, image_right=right, preset=5, wls=True, outputPath="./stereo_camera/demo_version/v1/static/depth/")


        