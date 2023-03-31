# Configuration points
TEST_NAME = "blender"

# Output paths
STEREO_OUT = "/stereo/"
COLOUR_OUT = "/image/"
DEPTH_OUT = "/depth/"


# CAMERA MODULE
# Stereo camera presets
LEFT_ID = "L"
LEFT_DEVICE_ADDRESS = "video3"
LEFT_WIDTH = 1920
LEFT_HEIGHT = 1080
LEFT_FRAMERATE = 30
# LEFT_IMAGE_DESTINATION = "/home/user/transfer_folder/"
RIGHT_ID = "R"
RIGHT_DEVICE_ADDRESS = "video4"
RIGHT_WIDTH = 1920
RIGHT_HEIGHT = 1080
RIGHT_FRAMERATE = 30
# RIGHT_IMAGE_DESTINATION = "/home/user/transfer_folder/"
IMAGE_TYPE = ".png"

IMAGE_DESTINATION = "/home/user/transfer_folder/"

# defualts (for monocular capture)
DEF_ID = "M"
DEF_DEVICE_ADDRESS = "video3"
DEF_WIDTH = 2592
DEF_HEIGHT = 1944
DEF_FRAMERATE = 8

# calibration sequence
SEQ_COUNT = 8
SEQ_DELAY = 2

# video
FPS = 10
V_WIDTH = 1920
V_HEIGHT = 1080

MOCK_DATA_SRC = "./blender/blender-models/"

VIDEO_DESTINATION = "/home/user/transfer_folder/"
CAPTURE_FREQ = 1 # capture an image every x seconds
END_CAPTURE = 0

# DEPTHMAP MODULE
IMAGE_TYPE = ".jpeg"
MAP_EXT = ".png"
SRC_IMAGE_DATA_PATH = "./active_data/images/"
MIDAS_MODEL_TYPE = "MiDaS" # "DPT_Large" "DPT_Hybrid"
DPI_COUNT = "1000" # DPI

# CALIBRATION MODULE
CALIBRATION_IMAGES = "./active_data/calibration_image_set/"
CURRENT_INTRINSICS = "./STEREO_PARAMS.xml"
CAMERA_INTRINSICS = "./INTRINSICS.xml"
CAMERA_EXTRINSICS = "./EXTRINSICS.xml"
CHECKERBOARD = (9,7)
FRAME_SIZE = (1920,1080)

# POINTCLOUD_MODULE
PLY_DESTINATION_PATH = "./active_data/pointclouds/active/"
K_POINTS = 3

DEFAULT_ICP_TRANSFORM_ESTIMATE = [1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]
DEFAULT_ICP_THRESHOLD = 0.02

DBSCAN_EPS = 0.02
DBSCAN_MIN_POINTS = 17