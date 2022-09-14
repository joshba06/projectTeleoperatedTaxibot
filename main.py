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

print('{:^75s}'.format('\n_____Starting project "teleoperatedTaxibot"_____\n'))

print('{:75s}'.format('Home path set to: '+userSettings['homePath']))

# Setup folder structure and install extented packages
if args.initialInstallation == True:
    files, paths = setup.installPackages(userSettings['homePath'], userSettings['labels'], firstInstallation=True)
elif args.initialInstallation == False:
    files, paths = setup.installPackages(userSettings['homePath'], userSettings['labels'], firstInstallation=False)


#%% Camera setup
print('{:^75s}'.format('\n_____Starting camera setup_____\n'))

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
capL = cv.VideoCapture(1)
# capR = cv.VideoCapture(0)

# Loop control for camera setup
cameraSetupCompleted = False
while cameraSetupCompleted == False:

    print('__Main menu__')

    selectionMain = input('Enter "Q" to exit program\nEnter "0" to check if cameras are connected properly\nEnter "1" to set up left camera\nEnter "2" to set up right camera\nEnter "3" to setup stereo system\nEnter "4" to start program with settings from file\n')

    # Quit program
    if selectionMain == 'Q':
        print('Program terminated by user...')
        sys.exit()

    # Check if cameras are connected properly
    elif selectionMain == '0':

        print('Checking camera configuration..')
        print('Press the ESC key to quit or press SPACE key to confirm correct connection')
        
        while True:
            key = cv.waitKey(25)

            if key == 27:
                print('Going back to main screen. Cameras have been switched')
                capL = cv.VideoCapture(0)
                # capR = cv.VideoCapture(1)
                
            elif key == ord(' '):
                print('Correct alignment of cameras confirmed...')
                break

            # Display livestream of camera
            isTrueL, frameL = capL.read()
            #isTrueR, frameR = capR.read()

            cv.imshow('Camera L', frameL)
            #cv.imshow('Camera R', frameR)

        cv.destroyAllWindows()
    
    # Left camera setup
    elif selectionMain == '1':
        
        leftCameraCompleted = False
        while leftCameraCompleted == False:
            print('\n____Left camera menu____')
            selectionLeftCam = input('__Enter "B" to get back to main menue\n__Enter "1" to generate new images\n__Enter "2" to find best image combination\n__Enter "3" to calibrate camera\n')

            # Go back to main menue
            if selectionLeftCam == 'B':
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
            print('\n____Right camera menu____')
            selectionRightCam = input('__Enter "B" to get back to main menue\n__Enter "1" to generate new images\n__Enter "2" to find best image combination\n__Enter "3" to calibrate camera\n')

            # Go back to main menue
            if selectionRightCam == 'B':
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
            print('\n____Stereo menu____')
            selectionStereo = input('__Enter "B" to get back to main menue\n__Enter "1" to generate new images\n__Enter "2" to find best image combination\n__Enter "3" to calibrate stereo\n')

            # Go back to main menue
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

    # Load parameters from file
    elif selectionMain == '4':

        # Camera intrinsics
        cameraSetup.getIntrinsicsLeftCamera(capL, paths, singleLGenImages=False, singleLTestCombinations=False, singleLCalibration=False, debuggingMode=debuggingMode)
        cameraSetup.getIntrinsicsRightCamera(capR, paths, singleRGenImages=False, singleRTestCombinations=False, singleRCalibration=False, debuggingMode=debuggingMode)

        # Stereo params
        Trns = cameraSetup.calibrateStereoSetup(capL, capR, paths, stereoGenImages=False, stereoTestCombinations=False, stereoCalibration=False, debuggingMode=debuggingMode)
        leftMapX, leftMapY, rightMapX, rightMapY, Q, rectifiedCameraMatrixL = cameraSetup.getRectificationMap(capL, capR, paths, newRectificationMapping=False, debuggingMode=debuggingMode)
        stereoCompleted = True

    else:
        print('No option was selected. Please try again...')

print('{:^75s}'.format('\n_____Finished camera setup_____\n'))


#%% Object detection setup
print('{:^75s}'.format('\n_____Initiating object detection_____\n'))

print('Importing required modules')
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

print('{:^75s}'.format('\n_____Initiation complete_____\n'))


#%% Main routine
print('{:^75s}'.format('\n_____Starting taxibot livestream_____\n'))

# Initiate correspondence algorithms
left_matcher, right_matcher, wls_filter = cameraSetup.initialiseCorrespondence()

new_frame_time = 0
prev_frame_time = 0
while True:
    key = cv.waitKey(1)

    if key == 27:
        print('Program was terminated by user..')
        sys.exit()

    isTrueL, frameL = capL.read()
    #isTrueR, frameR = capR.read()

    Left_rectified = cv.remap(frameL,leftMapX,leftMapY, cv.INTER_LINEAR, cv.BORDER_CONSTANT)
    #Right_rectified = cv.remap(frameR,rightMapX,rightMapY, cv.INTER_LINEAR, cv.BORDER_CONSTANT)

    heightL, widthL, channelsL = Left_rectified.shape

    #points_3D, colored_disp = cameraSetup.getDepth(Left_rectified, Right_rectified, Q, debuggingMode, left_matcher, right_matcher, wls_filter)

    # Detect object on LEFT rectified camera stream
    image_np = np.array(Left_rectified)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    label_id_offset = 1
    image_np_with_detectionsL = image_np.copy()

    image_np_with_detectionsL, aryfoundL = viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detectionsL,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=2,
                min_score_thresh=.8,
                agnostic_mode=False)

    if aryfoundL != 0:
        xmin = aryfoundL['xmin']*widthL
        xmax = aryfoundL['xmax']*widthL
        ymin = aryfoundL['ymin']*heightL
        ymax = aryfoundL['ymax']*heightL
        #print('Found {} at location x={}pixels, y={}pixels'.format(aryfound['class'], xmin, ymin))

        # Location (centre of object) as seen from left rectified camera
        x_centreL = xmin+0.5*(xmax-xmin)
        x_centreL = int(x_centreL)
        y_centreL = ymin+0.5*(ymax-ymin)
        y_centreL = int(y_centreL)

        ## Print information

        # Mark point
        cv.putText(image_np_with_detectionsL, 'x', (x_centreL, y_centreL), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2, cv.LINE_AA)

        # Depth via PCL
        #depthPCL = points_3D[y_centreL][x_centreL][2]
        #depth_text_PCL = ('Depth from PCL: z = {:.2f}cm'.format(depthPCL))


        #cv.putText(image_np_with_detectionsL, depth_text_PCL, (x_centreL, y_centreL), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2, cv.LINE_AA)
    
    cv.imshow('Camera Left - Detection Mode', image_np_with_detectionsL)
    #cv.imshow('Disparity', colored_disp)


    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = ('FPS: {:.2f}'.format(fps))
    cv.putText(image_np_with_detectionsL, fps, (7, 70), cv.FONT_HERSHEY_PLAIN, 3, (100, 255, 0), 1, cv.LINE_AA)


 
cv.destroyAllWindows()






    