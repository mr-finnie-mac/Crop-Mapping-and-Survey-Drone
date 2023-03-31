import os
from config import *



def check_mission_folder(missionCode):
    """Checks if a mission folder tree exists, returns the path (/missions, /image, /depth) or none"""

    # Check folder structure
    # should have been made using camera module's make_new_capture_folder function
    if ((os.path.exists("%smissions/%s/stereo"%(IMAGE_DESTINATION, missionCode))) and
        (os.path.exists("%smissions/%s/depth"%(IMAGE_DESTINATION, missionCode))) and
        (os.path.exists("%smissions/%s/image"%(IMAGE_DESTINATION, missionCode)))):

        missionPath = "%smissions/%s/"%(IMAGE_DESTINATION, missionCode)
        imagePath = "%smissions/%s/image/"%(IMAGE_DESTINATION, missionCode)
        depthPath = "%smissions/%s/depth/"%(IMAGE_DESTINATION, missionCode)
        return missionPath, imagePath, depthPath
    
    else:
        missionPath = None
        print("Mission code and/or vali directory was not found")
        return "mission path error"