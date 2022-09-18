#%% Import fundamental dependencies (which are installed by default or by cloning taxibot repo)
import os
import sys
import time
from pprint import pprint
import argparse

import userSettings
import setup

#%% System setup

# Load user settings
userSettings = userSettings.main()

# Check whether to run program in first installation mode or default
parser = argparse.ArgumentParser()
parser.add_argument("--initialInstallation", type=bool, default=False)
parser.add_argument("--debugging", type=bool, default=False)
args = parser.parse_args()

print('\n\n\n\n\n________________Starting project "teleoperatedTaxibot"________________\n\n')

print('\n________________1. System setup________________\n')

print('Home path set to: '+userSettings['homePath'])

# Setup folder structure and install extented packages
if args.initialInstallation == True:
    files, paths = setup.installPackages(userSettings['homePath'], userSettings['labels'], firstInstallation=True)
elif args.initialInstallation == False:
    files, paths = setup.installPackages(userSettings['homePath'], userSettings['labels'], firstInstallation=False)


#%% Camera setup
print('\n\n________________2. Camera setup________________\n')

# Check whether debugging mode shall be started
if args.debugging == True:
    debuggingMode = True
else:
    debuggingMode = False

# Load settings for camera setup
import cv2 as cv
import numpy as np
import cameraSetup as cameraSetup

# Open camera capture
try:
    capL = cv.VideoCapture(0)
    capR = cv.VideoCapture(2)
except:
    print('Cannot find correct (internal) camera ports. Please visit main.py on line 50 and exchange values 0 to 2 for other values, usually 1, 2 or 3 depending on the number of cameras that are connected')
    sys.exit()

# Loop control for camera setup
cameraSetupCompleted = False
while cameraSetupCompleted == False:

    print("\n\033[4m" + 'Main menu' + "\033[0m")

    selectionMain = input('-Enter "Q" to exit program\n-Enter "0" to check if cameras are connected properly\n-Enter "1" to set up left camera\n-Enter "2" to set up right camera\n-Enter "3" to setup stereo system\n-Enter "4" to start program with settings from file\n')

    # Quit program
    if selectionMain == 'Q':
        print('Program terminated by user...')
        sys.exit()

    # Check if cameras are connected properly
    elif selectionMain == '0':

        print('Checking camera configuration..')
        print('Click on one pop up window and press the ESC key to quit or SPACE key to confirm correct camera arrangement')
        
        while True:
            key = cv.waitKey(25)

            if key == 27:
                print('Terminating program. Please switch the ports that you connected your cameras to (A->B, B->A)')
                cv.destroyAllWindows()
                break
                
            elif key == ord(' '):
                print('Correct alignment of cameras confirmed...')
                cv.destroyAllWindows()
                break

            # Display livestream of camera
            isTrueL, frameL = capL.read()
            isTrueR, frameR = capR.read()

            cv.imshow('Camera L', frameL)
            cv.imshow('Camera R', frameR)

        cv.destroyAllWindows()
    
    # Left camera setup
    elif selectionMain == '1':
        
        leftCameraCompleted = False
        while leftCameraCompleted == False:
            print("\n\033[4m" + 'Main menu -> Left camera menu' + "\033[0m")
            selectionLeftCam = input('-Enter "Q" to exit program\n-Enter "B" to get back to main menu\n-Enter "1" to generate new images\n-Enter "2" to find best image combination\n-Enter "3" to calibrate camera\n')
         
            # Quit program
            if selectionLeftCam == 'Q':
                print('Program terminated by user...')
                sys.exit()

            # Go back to main menu
            elif selectionLeftCam == 'B':
                leftCameraCompleted = True

            # Generate new images
            elif selectionLeftCam == '1':
                cameraSetup.getIntrinsicsLeftCamera(capL, paths, singleLGenImages=True, singleLTestCombinations=False, singleLCalibration=False, debuggingMode=debuggingMode)

            # Find best combination of images
            elif selectionLeftCam == '2':
                cameraSetup.getIntrinsicsLeftCamera(capL, paths, singleLGenImages=False, singleLTestCombinations=True, singleLCalibration=False, debuggingMode=debuggingMode)

            # Find best combination of images
            elif selectionLeftCam == '3':
                cameraSetup.getIntrinsicsLeftCamera(capL, paths, singleLGenImages=False, singleLTestCombinations=False, singleLCalibration=True, debuggingMode=debuggingMode)
                leftCameraCompleted = True
            else:
                print('Please make a selection')
   
    # Right camera setup
    elif selectionMain == '2':
        
        rightCameraCompleted = False
        while rightCameraCompleted == False:
            print("\n\033[4m" + 'Main menu -> Right camera menu' + "\033[0m")
            selectionRightCam = input('-Enter "Q" to exit program\n-Enter "B" to get back to main menu\n-Enter "1" to generate new images\n-Enter "2" to find best image combination\n-Enter "3" to calibrate camera\n')

            # Quit program
            if selectionRightCam == 'Q':
                print('Program terminated by user...')
                sys.exit()

            # Go back to main menu
            elif selectionRightCam == 'B':
                rightCameraCompleted = True

            # Generate new images
            elif selectionRightCam == '1':
                cameraSetup.getIntrinsicsRightCamera(capR, paths, singleRGenImages=True, singleRTestCombinations=False, singleRCalibration=False, debuggingMode=debuggingMode)

            # Find best combination of images
            elif selectionRightCam == '2':
                cameraSetup.getIntrinsicsRightCamera(capR, paths, singleRGenImages=False, singleRTestCombinations=True, singleRCalibration=False, debuggingMode=debuggingMode)

            # Find best combination of images
            elif selectionRightCam == '3':
                cameraSetup.getIntrinsicsRightCamera(capR, paths, singleRGenImages=False, singleRTestCombinations=False, singleRCalibration=True, debuggingMode=debuggingMode)
                rightCameraCompleted = True
            else:
                print('Please make a selection')

    # Stereo setup
    elif selectionMain == '3':

        
        stereoCompleted = False
        while stereoCompleted == False:
            print('_____ Computing stereo parameters _____\n')
            print('\nMain menu -> Stereo menu')
            selectionStereo = input('__Enter "B" to get back to main menu\n__Enter "1" to generate new images\n__Enter "2" to find best image combination\n__Enter "3" to calibrate stereo\n')

            # Go back to main menu
            if selectionStereo == 'B':
                stereoCompleted = True

            # Generate new images
            elif selectionStereo == '1':
                cameraSetup.calibrateStereoSetup(capL, capR, paths, stereoGenImages=True, stereoTestCombinations=False, stereoCalibration=False, debuggingMode=debuggingMode)

            # Find best combination of images
            elif selectionStereo == '2':
                cameraSetup.calibrateStereoSetup(capL, capR, paths, stereoGenImages=False, stereoTestCombinations=True, stereoCalibration=False, debuggingMode=debuggingMode)

            # Calibrate stereo and compute rectification maps
            elif selectionStereo == '3':
                Trns = cameraSetup.calibrateStereoSetup(capL, capR, paths, stereoGenImages=False, stereoTestCombinations=False, stereoCalibration=True, debuggingMode=debuggingMode)

                leftMapX, leftMapY, rightMapX, rightMapY, Q, rectifiedCameraMatrixL = cameraSetup.getRectificationMap(capL, capR, paths, newRectificationMapping=True, debuggingMode=debuggingMode)
                stereoCompleted = True
            
            print('.\n_____ Finished computing stereo parameters _____\n.')

    # Load parameters from file
    elif selectionMain == '4':

        print('\n_____ Loading all parameters from file _____\n')

        # Camera intrinsics
        cameraSetup.getIntrinsicsLeftCamera(capL, paths, singleLGenImages=False, singleLTestCombinations=False, singleLCalibration=False, debuggingMode=debuggingMode)
        cameraSetup.getIntrinsicsRightCamera(capR, paths, singleRGenImages=False, singleRTestCombinations=False, singleRCalibration=False, debuggingMode=debuggingMode)

        # Stereo params
        Trns = cameraSetup.calibrateStereoSetup(capL, capR, paths, stereoGenImages=False, stereoTestCombinations=False, stereoCalibration=False, debuggingMode=debuggingMode)
        leftMapX, leftMapY, rightMapX, rightMapY, Q, rectifiedCameraMatrixL = cameraSetup.getRectificationMap(capL, capR, paths, newRectificationMapping=False, debuggingMode=debuggingMode)
        stereoCompleted = True
        cameraSetupCompleted = True

    else:
        print('No option was selected. Please try again...')


#%% Object detection setup
print('\n________________3. Object detection initialisation________________\n')

print('Importing modules for object detection')
import tensorflow as tf
import object_detection
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
print('Finished importing modules')
os.chdir(paths['home'])

# Determine latest checkpoint
path_model = paths['trainedModels']+'/'+userSettings['pretrainedModelName']
files = os.listdir(path_model)
last_checkpoint = 0
for file in files:
    ckpt = file.split('.')[0]
    try:
        ckpt = ckpt.split('-')[1]
    except:
        break
    else:
        #print('Current checkpoint: '+ckpt)
        if (int(ckpt) > last_checkpoint):
            last_checkpoint = int(ckpt)
last_checkpoint = 'ckpt-'+str(last_checkpoint)
print('Found latest checkpoint: '+last_checkpoint)

# Load pipeline config and build a detection model
path_pipeline = path_model+'/pipeline.config'
configs = config_util.get_configs_from_pipeline_file(path_pipeline)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(path_model, str(last_checkpoint))).expect_partial()

def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


path_labelmap = path_model+'/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(path_labelmap)


#%% Main routine
print('\n________________4. Starting Taxibot livestream________________\n')

# Initiate correspondence algorithms
left_matcher, right_matcher, wls_filter = cameraSetup.initialiseCorrespondence() 

new_frame_time = 0
prev_frame_time = 0
while True:
    key = cv.waitKey(25)

    isTrueL, frameL = capL.read()
    isTrueR, frameR = capR.read()

    Left_rectified = cv.remap(frameL,leftMapX,leftMapY, cv.INTER_LINEAR, cv.BORDER_CONSTANT)
    Right_rectified = cv.remap(frameR,rightMapX,rightMapY, cv.INTER_LINEAR, cv.BORDER_CONSTANT)

    height, width, channels = Left_rectified.shape

    if key == 27:
        print('Program was terminated by user..')
        sys.exit()

    points_3D, disparityMap = cameraSetup.getDepth(Left_rectified, Right_rectified, Q, debuggingMode, left_matcher, right_matcher, wls_filter)

    # Detect object on left rectified camera stream
    image_np = np.array(Left_rectified)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    image_np_with_detections, aryFound = viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=2,
            min_score_thresh=.8,
            agnostic_mode=False)

    # If an object is found, get its/their image coordinates and compute depth
    if aryFound != 0:

        keys = aryFound.keys()

        for key in keys:

            object = key.split(':')[0]
            accuracy = key.split(':')[1]

            xmin = aryFound[key][0]*width
            xmax = aryFound[key][1]*width
            ymin = aryFound[key][2]*height
            ymax = aryFound[key][3]*height

            # Location (centre of object) as seen from left rectified camera
            x_centre = xmin+0.5*(xmax-xmin)
            x_centre = int(x_centre)
            y_centre = ymin+0.5*(ymax-ymin)
            y_centre = int(y_centre)
            

            # Mark point
            cv.putText(image_np_with_detections, 'x', (x_centre, y_centre), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2, cv.LINE_AA)
            depthPCL = points_3D[y_centre][x_centre][2]

            # Depth via PCL
            depth_text_PCL = ('Engine at: Z = {:.2f}cm'.format(depthPCL))
            cv.putText(image_np_with_detections, depth_text_PCL, (x_centre-15, y_centre+20), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2, cv.LINE_AA)



    # Add frame rate to display
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = ('FPS: {:.2f}'.format(fps))
    cv.putText(image_np_with_detections, fps, (7, 70), cv.FONT_HERSHEY_PLAIN, 3, (100, 255, 0), 1, cv.LINE_AA)
    
    cv.imshow('Camera Left - Detection Mode', image_np_with_detections)
    cv.imshow('Disparity map', disparityMap)

cv.destroyAllWindows()






    