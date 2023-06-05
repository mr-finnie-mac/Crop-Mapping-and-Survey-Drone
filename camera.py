import cv2 as cv
import threading
import numpy as np
import time
from config import *
import os

# This function is depreciated, use captureMono or captureStereo instead
def capture_image(imageNumber, device_addr, position):
    # imageNumber : given number for image
    # device addr : device address on system
    # position : is camera left, right or monocular
    period = 4
    capture_image = True

    start = time.time()
    time.process_time()
    elapsed = 0
    while elapsed < period:
        elapsed = time.time() - start
        if(capture_image):
            try:
                destination = "/home/user/transfer_folder/" + str(position) + "_camera_"+str(imageNumber)+".jpeg"
                capture_params = "v4l2src device=" + str(device_addr) + " ! video/x-raw, framerate=30/1,width=720, height=576 ! appsink"
                camera = cv.VideoCapture(capture_params, cv.CAP_GSTREAMER)
                ret, frame = camera.read()
                cv.imwrite(destination, frame)

                print("-- Image captured -- ")
                print(time.time())
                break
            except:
                print("Camera at %s failed to open!"%(device_addr))


# Camera class for taking pictures
class Camera:
    def __init__(self, id, device_address, res_width, res_height, fr, save_path):
        self.video_capture = None
        self.frame = None
        self.grabbed = False
        self.read_thread = None
        self.read_lock = threading.Lock()
        self.running = False
        self.LR = "Mono"

        # configs
        self.id = id
        self.device_address = device_address
        self.res_width = res_width
        self.res_height = res_height
        self.fr = fr
        self.save_path = save_path

    def begin(self, capture_params):
        try:
            self.video_capture = cv.VideoCapture(capture_params, cv.CAP_GSTREAMER)
            self.grabbed, self.frame = self.video_capture.read()

        except RuntimeError:
            self.video_capture = None
            print(capture_params + " : Camera did not open correctly")
        
    def start(self):
        if self.running:
            print("Video capture already running")
            return None
        if self.video_capture != None:
            self.running = True
            self.read_thread = threading.Thread(target=self.updateCamera)
            self.read_thread.start()
        return self

    def stop(self):
        self.running = False
        self.read_thread.join()
        self.read_thread = None
        
    def updateCamera(self):
        while self.running:
            try:
                grabbed, frame = self.video_capture.read()
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            except RuntimeError:
                print("Could not read image from camera")

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def release(self):
        if self.video_capture != None:
            self.video_capture.release()
            self.video_capture = None
        # Now kill the thread
        if self.read_thread != None:
            self.read_thread.join()

    def cameraInfo(self):
        print(self.id,
              self.device_address,
              self.res_width,
              self.res_height,
              self.fr,
              self.save_path)

# Build a gstreamer pipeline using a given set of parameters
def gstreamer_pipeline(
    device_addr = DEF_DEVICE_ADDRESS,
    width = DEF_WIDTH,
    height = DEF_HEIGHT,
    framerate = DEF_FRAMERATE,
    id = DEF_ID
):
    string =  (
        "v4l2src device=/dev/%s ! video/x-raw,framerate=%d/1, width=%d, height=%d ! appsink" 
        %(str(device_addr), framerate, width, height)
    )
    print(id + " PIPELINE MADE USING : " + string)
    return(string)


def runMonoCamera(mono_camera):
    print(" -- Starting Monocular CSI Camera now -- ")
    mono_camera.begin(
        gstreamer_pipeline(
            device_addr = mono_camera.device_address,
            width = mono_camera.res_width,
            height = mono_camera.res_height,
            framerate = mono_camera.fr,
            id = mono_camera.id
        )
    )
    mono_camera.start()

def runStereoCamera(left_camera, right_camera):
    print("-- Starting Stereo CSI Cameras now -- ")
    left_camera.begin(
        gstreamer_pipeline(
            device_addr = left_camera.device_address,
            width = left_camera.res_width,
            height = left_camera.res_height,
            framerate = left_camera.fr,
            id = left_camera.id
        )
    )
    left_camera.start()

    
    right_camera.begin(
        gstreamer_pipeline(
            device_addr = right_camera.device_address,
            width = right_camera.res_width,
            height = right_camera.res_height,
            framerate = right_camera.fr,
            id = right_camera.id
        )
    )
    right_camera.start()


def recordStereoVideo(duration, left_camera, right_camera):
    left_result = cv.VideoWriter(VIDEO_DESTINATION + left_camera.id + ".avi", cv.VideoWriter_fourcc(*'MJPG'),FPS, (V_WIDTH, V_HEIGHT))
    right_result = cv.VideoWriter(VIDEO_DESTINATION + right_camera.id + ".avi", cv.VideoWriter_fourcc(*'MJPG'),FPS, (V_WIDTH, V_HEIGHT))
    timer = 0
    while True:
        left_ret, left_image = left_camera.read()
        right_ret, right_image = right_camera.read()

        if left_ret == True and right_ret == True:
            left_result.write(left_image)
            right_result.write(right_image)
        # time.sleep(0.1)
        timer +=1
        if timer > duration:
            break
    
    left_camera.stop()
    right_camera.stop()
    left_camera.release()
    right_camera.release()
    left_result.release()
    right_result.release()
    print("Recordings done!")

def recordMonoVideo(duration, mono_camera):
    mono_result = cv.VideoWriter(VIDEO_DESTINATION + mono_camera.id + ".avi", cv.VideoWriter_fourcc(*'MJPG'),FPS, (V_WIDTH, V_HEIGHT))
    timer = 0
    while True:
        mono_ret, mono_image = mono_camera.read()

        if mono_ret == True:
            mono_result.write(mono_image)

        # time.sleep(0.1)
        timer +=1
        if timer > duration:
            break
    
    mono_camera.stop()
    mono_camera.release()
    mono_result.release()
    print("Monocular recording complete at %s"%(VIDEO_DESTINATION))

def getCurrentMS():
    return str(round(time.time()*1000))

def captureMono(mono_camera, imageDestination=IMAGE_DESTINATION):
    if mono_camera.video_capture.isOpened():

        _, mono_image = mono_camera.read()
        cv.imwrite(("%s%s_%s.jpeg"%(imageDestination, getCurrentMS(), mono_camera.id)), mono_image) # save image with epoch
    
        mono_camera.stop()
        mono_camera.release()

    else:
        print("Error: Unable to open camera device")
        mono_camera.stop()
        mono_camera.release()

def captureStereo(left_camera, right_camera, imageDestination=IMAGE_DESTINATION):
    if left_camera.video_capture.isOpened() and right_camera.video_capture.isOpened():

        _, left_image = left_camera.read()
        cv.imwrite(("%s%s_%s%s"%(imageDestination, getCurrentMS(), left_camera.id, IMAGE_TYPE)), left_image) # save left image with epoch
        _, right_image = right_camera.read()
        cv.imwrite(("%s%s_%s%s"%(imageDestination, getCurrentMS(), right_camera.id, IMAGE_TYPE)), right_image) # save right image with epoch
        print("Saved to" + imageDestination)
        print(os.getcwd())
        
        
        left_camera.stop()
        left_camera.release()
        right_camera.stop()
        right_camera.release()

        return left_image, right_image

    else:
        print("Error: Unable to open both cameras")
        left_camera.stop()
        left_camera.release()
        right_camera.stop()
        right_camera.release()

def monoCalibrationSequence(mono_camera):
    for cap in range(SEQ_COUNT):
        print("CAPTURING MONO IMAGE %d of %d"% (cap+1, SEQ_COUNT))
        runMonoCamera(mono_camera)
        captureMono(mono_camera)
        time.sleep(SEQ_DELAY)

def stereoCalibrationSequence(left_camera, right_camera):
    # Start cameras up and capture images 
    for cap in range(SEQ_COUNT):
        print("CAPTURING STEREO IMAGE %d of %d"% (cap+1, SEQ_COUNT))
        runStereoCamera(left_camera, right_camera)
        captureStereo(left_camera, right_camera)
        time.sleep(SEQ_DELAY)

def capture_stereo_frame(left_camera, right_camera):
    """Captures a frame from each camera, left and right, returns RGB left and right"""
    runStereoCamera(left_camera, right_camera)
    leftRGB, rightRGB = captureStereo(left_camera, right_camera)
    return leftRGB, rightRGB

def inflight_monocular_procedure(mono_camera, missionCode=getCurrentMS(), duration=0):
    """Make a sequence of capture, inflight. Interupt this capture sequence by setting the END_CAPTURE to 1."""
    captureCounter = 0
    newPath = IMAGE_DESTINATION + "/missions/" + missionCode + '/'
    os.makedirs(newPath, exist_ok=True)
    start_time = time.perf_counter()
    end_time = start_time + duration
    print("Starting inflight monocular procedure")
    while END_CAPTURE == 0:
            captureCounter = captureCounter + 1
            print("CAPTURING MONO IMAGE  No.  %d"% (captureCounter))
            runMonoCamera(mono_camera)
            captureMono(mono_camera,imageDestination=newPath)
            time.sleep(CAPTURE_FREQ) # wait x seconds before next image is captured
            if time.perf_counter() >= end_time and duration != 0:
                print("Stopping inflight monocular procedure, timer has expired (%ds)"%(duration))
                break

def inflight_stereo_procedure(left_camera, right_camera, missionCode=getCurrentMS(), duration=0):
    """Make a sequence of stereo captures, inflight. Interupt this capture sequence by setting the END_CAPTURE to 1."""
    captureCounter = 0
    newPath = IMAGE_DESTINATION + "/missions/" + missionCode + STEREO_OUT
    os.makedirs(newPath, exist_ok=True)
    start_time = time.perf_counter()
    end_time = start_time + duration
    print("Starting inflight stereo procedure")
    while END_CAPTURE == 0:
            captureCounter = captureCounter + 1
            print("CAPTURING STEREO IMAGE  No.  %d"% (captureCounter))
            runStereoCamera(left_camera, right_camera)
            captureStereo(left_camera, right_camera, imageDestination=newPath)
            time.sleep(CAPTURE_FREQ) # wait x seconds before next stereo image is captured
            if time.perf_counter() >= end_time and duration != 0:
                print("Stopping inflight stereo procedure, timer has expired (%ds)"%(duration))
                break

def make_new_capture_folder(missionCode):
    "Build a folder tree for storing the image data for a new mission"
    os.makedirs("%smissions/%s/stereo/"%(IMAGE_DESTINATION, missionCode), exist_ok=True)
    os.makedirs("%smissions/%s/image/"%(IMAGE_DESTINATION, missionCode), exist_ok=True)
    os.makedirs("%smissions/%s/depth/"%(IMAGE_DESTINATION, missionCode), exist_ok=True)
    print("New mission folder at %smissions/%s/ with subfolders [stereo, depth, image]"%(IMAGE_DESTINATION, missionCode))
    return ("%smissions/%s/"%(IMAGE_DESTINATION, missionCode)), missionCode

def testAllFeatures():

    # Monocular camera
    
    # mono setup
    # make camera_object with monocular attributes then print its config
    mono_camera = Camera(DEF_ID, DEF_DEVICE_ADDRESS, DEF_WIDTH, DEF_HEIGHT, DEF_FRAMERATE, IMAGE_DESTINATION)
    mono_camera.cameraInfo()

    # mono controls
    # Take a sequence of images using monocular camera
    monoCalibrationSequence(mono_camera)
    # Take a single image with monocular camera
    runMonoCamera(mono_camera)
    captureMono(mono_camera)
    # Make a 10s stereo video
    runMonoCamera(mono_camera)
    recordMonoVideo(10, mono_camera)
    # Stereo camera(s) setup 

    # stereo setup
    # make camera objects for left and right then print their configs
    left_camera = Camera(LEFT_ID, LEFT_DEVICE_ADDRESS, LEFT_WIDTH, LEFT_HEIGHT, LEFT_FRAMERATE, IMAGE_DESTINATION)
    right_camera = Camera(RIGHT_ID, RIGHT_DEVICE_ADDRESS, RIGHT_WIDTH, RIGHT_HEIGHT, RIGHT_FRAMERATE, IMAGE_DESTINATION)
    # print the current config of left and right cameras
    left_camera.cameraInfo()
    right_camera.cameraInfo()

    # stereo controls
    # Take a sequence of images using stereo camera
    stereoCalibrationSequence(left_camera, right_camera)
    # Take a single image with stereo camera
    runStereoCamera(left_camera,right_camera)
    captureStereo(left_camera, right_camera)
    # Make a 10s stereo video
    runStereoCamera(left_camera,right_camera)
    recordStereoVideo(10, left_camera, right_camera)


if __name__ == "__main__":
    # testAllFeatures()
    # mono_camera = Camera(DEF_ID, DEF_DEVICE_ADDRESS, DEF_WIDTH, DEF_HEIGHT, DEF_FRAMERATE, IMAGE_DESTINATION)
    # mono_camera.cameraInfo()

    # # inflight_monocular_procedure(mono_camera, "sandlewood_and_amber", duration=10)
    # inflight_monocular_procedure(mono_camera, "ardu")

    # make camera objects for left and right then print their configs
    left_camera = Camera(LEFT_ID, LEFT_DEVICE_ADDRESS, LEFT_WIDTH, LEFT_HEIGHT, LEFT_FRAMERATE, IMAGE_DESTINATION)
    right_camera = Camera(RIGHT_ID, RIGHT_DEVICE_ADDRESS, RIGHT_WIDTH, RIGHT_HEIGHT, RIGHT_FRAMERATE, IMAGE_DESTINATION)
    # print the current config of left and right cameras
    left_camera.cameraInfo()
    right_camera.cameraInfo()
    
    path, missionCode = make_new_capture_folder(getCurrentMS())
    inflight_stereo_procedure(left_camera, right_camera, missionCode=missionCode, duration=150) # make 2.5 min long capture procedure with folder name being current ms


