from flask import Flask, render_template, jsonify, request, url_for
import os
import glob
import sys
import logging

from camera import *
from depthmap_navqplus import stereo_depthmap_compute, multithread_depthmap

folder_path = './static' 




app = Flask(__name__)

app.config['TIMEOUT'] = 60  # Set timeout to 60 seconds
# Configure the Flask logger
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.DEBUG)  # Set the desired logging level

@app.route('/')
def hello_world():
    return 'Hello edge computing world!'

@app.route('/image')
def show_image():
    # left_image_filename = 'None'
    # right_image_filename = 'None'
    # Get a list of all files in the folder
    file_list = glob.glob(os.path.join(folder_path, '*'))
    file_list = [file_name for file_name in file_list if "jpeg" in file_name]

    # Sort the file list by modification time (newest first)
    file_list.sort(key=os.path.getmtime, reverse=True)

    if file_list:
        try:
            left_image_filename = os.path.basename(file_list[0])
            right_image_filename = os.path.basename(file_list[1])
        except:
            left_image_filename = os.path.basename(file_list[0])
            right_image_filename = os.path.basename(file_list[0])

    
    return render_template('./page.html', left_image_filename=left_image_filename, right_image_filename=right_image_filename)

@app.route('/depth')
def show_depth():
    left_image_filename = None
    depth_image_filename = None
    # Get a list of all files in the folder
    image_list = glob.glob(os.path.join(folder_path, '*'))
    image_list = [file_name for file_name in image_list if "jpeg" in file_name]
    depth_list = glob.glob(os.path.join(folder_path+"/depth", '*'))

    # Sort the file list by modification time (newest first)
    image_list.sort(key=os.path.getmtime, reverse=True)
    depth_list.sort(key=os.path.getmtime, reverse=True)
 
    if image_list and depth_list:
        try:
            left_image_filename = os.path.basename(image_list[0])
            depth_image_filename = os.path.basename(depth_list[0])

        except:
            left_image_filename = os.path.basename(image_list[0])
            depth_image_filename = os.path.basename(image_list[0])
            

    
    return render_template('./page.html', left_image_filename=left_image_filename, right_image_filename="/depth/"+depth_image_filename)

@app.route('/index')
def show_index():
    left_image_filename = None
    right_image_filename = None
    depth_image_filename = None
    # Get a list of all files in the folder
    image_list = glob.glob(os.path.join(folder_path, '*'))
    image_list = [file_name for file_name in image_list if "jpeg" in file_name]
    depth_list = glob.glob(os.path.join(folder_path+"/depth", '*'))

    # Sort the file list by modification time (newest first)
    image_list.sort(key=os.path.getmtime, reverse=True)
    depth_list.sort(key=os.path.getmtime, reverse=True)
 
    if image_list and depth_list:
        try:
            left_image_filename = os.path.basename(image_list[1])
            right_image_filename = os.path.basename(image_list[0])
            depth_image_filename = os.path.basename(depth_list[0])

        except:
            left_image_filename = os.path.basename(image_list[0])
            right_image_filename = os.path.basename(image_list[0])
            depth_image_filename = os.path.basename(depth_list[0])
            
    return render_template('./index.html', left_image_filename=left_image_filename, right_image_filename=right_image_filename, depth_image_filename="/depth/"+depth_image_filename)

def getLatestImages():
    # Find images and return page
    left = None
    right = None
    # Get a list of all files in the folder
    image_list = glob.glob(os.path.join(folder_path, '*'))
    image_list = [file_name for file_name in image_list if "jpeg" in file_name]
    depth_list = glob.glob(os.path.join(folder_path+"/depth", '*'))

    # Sort the file list by modification time (newest first)
    image_list.sort(key=os.path.getmtime, reverse=True)
    depth_list.sort(key=os.path.getmtime, reverse=True)
    cwd = url_for('static', filename = os.path.basename(image_list[1]))
    app.logger.debug(f"Current LEFT FILE: {cwd}")
    if image_list:
        try:
            leftFile = url_for('static', filename = os.path.basename(image_list[1]))
            rightFile = url_for('static', filename = os.path.basename(image_list[0]))

            imgL = cv.imread(leftFile)
            imgR = cv.imread(rightFile)

            imgR_gray = cv.cvtColor(imgR,cv.COLOR_BGR2GRAY)
            imgL_gray = cv.cvtColor(imgL,cv.COLOR_BGR2GRAY)
            left = imgL_gray
            right = imgR_gray
            
            # left =  cv.imread(url_for('static', filename = os.path.basename(image_list[1])), cv.IMREAD_COLOR)
            # right =  cv.imread(url_for('static', filename = os.path.basename(image_list[0])), cv.IMREAD_COLOR)
           
        except:
            app.logger.debug("Latest images could not be found")

    
    
    # left = cv.imread(os.path.basename(image_list[1]))
    # right = cv.imread(os.path.basename(image_list[0]))

    return left, right

    

@app.route('/capture-depth', methods=['GET'])
def capture_depth():
    # Capture process
    req_preset = request.args.get('preset')
    req_wls = request.args.get('wls')
    req_sigma = request.args.get('sigma')
    req_lambda = request.args.get('lambda')
    req_post_filter = request.args.get('filters')
    req_newPics = request.args.get('newPics')


    if req_newPics == 'true':
        # make camera objects for left and right then print their configs
        left_camera = Camera(LEFT_ID, LEFT_DEVICE_ADDRESS, 640, 480, LEFT_FRAMERATE, "/")
        right_camera = Camera(RIGHT_ID, RIGHT_DEVICE_ADDRESS, 640, 480, RIGHT_FRAMERATE, "/")
        # print the current config of left and right cameras
        left_camera.cameraInfo()
        right_camera.cameraInfo()
    
        # Take a single image with stereo camera
        runStereoCamera(left_camera,right_camera)
        left, right = captureStereo(left_camera, right_camera, imageDestination="./static/")
        # cwd = os.getcwd()
        # app.logger.debug(f"Current Working Directory: {cwd}")

    else:
        left, right = getLatestImages()

    stereo_depthmap_compute(leftImage=left, rightImage=right, preset=int(req_preset), wls=req_wls=="true", outputPath="./static/depth/", sigma=float(req_sigma), lmbda=float(req_lambda), post_filter=int(req_post_filter))
    # Handle the logic for endpoint 1


    # Find images and return page
    print("reached capture depth")
    left_image_filename = None
    right_image_filename = None
    depth_image_filename = None
    # Get a list of all files in the folder
    image_list = glob.glob(os.path.join(folder_path, '*'))
    image_list = [file_name for file_name in image_list if "jpeg" in file_name]
    depth_list = glob.glob(os.path.join(folder_path+"/depth", '*'))

    # Sort the file list by modification time (newest first)
    image_list.sort(key=os.path.getmtime, reverse=True)
    depth_list.sort(key=os.path.getmtime, reverse=True)
 
    if image_list and depth_list:
        try:
            left_image_filename = os.path.basename(image_list[1])
            right_image_filename = os.path.basename(image_list[0])
            depth_image_filename = os.path.basename(depth_list[0])

        except:
            left_image_filename = os.path.basename(image_list[0])
            right_image_filename = os.path.basename(image_list[0])
            depth_image_filename = os.path.basename(depth_list[0])

    response = {'left': left_image_filename, 'right': right_image_filename, 'depth': "/depth/"+depth_image_filename}
    print("reached capture depth")

    return jsonify(response)

@app.route('/capture-image', methods=['POST'])
def endpoint2():
    # left_image_filename = 'None'
    # right_image_filename = 'None'
    # Get a list of all files in the folder
    file_list = glob.glob(os.path.join(folder_path, '*'))
    file_list = [file_name for file_name in file_list if "jpeg" in file_name]

    # Sort the file list by modification time (newest first)
    file_list.sort(key=os.path.getmtime, reverse=True)

    if file_list:
        try:
            left_image_filename = os.path.basename(file_list[0])
            right_image_filename = os.path.basename(file_list[1])
        except:
            left_image_filename = os.path.basename(file_list[0])
            right_image_filename = os.path.basename(file_list[0])

    return left_image_filename, right_image_filename


if __name__ == '__main__':
    app.run(host='0.0.0.0')