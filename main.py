#%% User settings___________________________________________________________________________

labels = ['Engine']
custom_model_name = 'my_ssd_mobilenet_v2_fpnlite'

#%% Import fundamental dependencies_________________________________________________________
import os
import sys
import time
from pprint import pprint
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("-ii",
#                     "--initalInstallation",
#                     type=bool, default=False)
# parser.add_argument("-h",
#                     "--homePath",
#                     type=str, default=False)
# args = parser.parse_args()



#%% Installation ___________________________________________________________________
print('{:^75s}'.format('_____Starting project "taxibotAutonomous"_____'))

# if args.homePath is None:
#     home_path = '/Users/niklas/Virtual_Environment/Version_5/projectAutonomous'
# elif args.homePath is not None:
#     home_path = args.homePath
home_path = '/Users/niklas/Virtual_Environment/Version_6/projectTeleoperatedTaxibot'
print('{:75s}'.format('Home path set to: '+home_path))


# Setup folder structure and install extented packages
import setup
# if args.initialInstallation == True:
#     files, paths = setup.installPackages(home_path, labels, firstInstallation=True)
# elif args.initialInstallation == False:
#     files, paths = setup.installPackages(home_path, labels, firstInstallation=False)

#%% Loop control
# print('{:75s}'.format('.'))
# print('{:75s}'.format('.'))
# print('{:75s}'.format('Enter "Q" to exit program\nEnter "0" to start object detection\nEnter "1" to calibrate left camera\nEnter "2" to calibrate right camera\nEnter "3" to calibrate stereo setup\n'))
# print('{:75s}'.format('.'))
# print('{:75s}'.format('.'))
# selection = input('Selection: ')

# Quit program

# Start program

# Left camera calibration
# print('{:75s}'.format('Selected 1: Calibrate left camera'))
# print('{:75s}'.format('Enter "1" to generate new images\nEnter "2" to find best image combination\nEnter "3" to calibrate camera'))

# # Right camera calibration
# print('{:75s}'.format('Selected 2: Calibrate right camera'))
# print('{:75s}'.format('Enter "1" to generate new images\nEnter "2" to find best image combination\nEnter "3" to calibrate camera'))

# # Stereo calibration
# print('{:75s}'.format('Selected 2: Calibrate right camera'))
# print('{:75s}'.format('Enter "1" to generate new images\nEnter "2" to find best image combination\nEnter "3" to calibrate camera'))


debuggingMode = False

stereoGenImages = False
singleLGenImages = False
singleRGenImages = False

singleLTestCombinations = False
singleRTestCombinations = False
stereoTestCombinations = False

singleLCalibration = False
singleRCalibration = False
stereoCalibration = False

newRectificationMapping = False

files, paths = setup.installPackages(home_path, labels, firstInstallation=False)
# Import newly installed dependencies
print('Importing further packages...')
import cv2 as cv
import numpy as np

import cameraSetup as cameraSetup

#%% 0. Check if cameras are displayed correctly_______________________________________
# capL = cv.VideoCapture(1)
#capR = cv.VideoCapture(0)

if debuggingMode is True:
    print('Checking camera configuration..')

    while True:
        key = cv.waitKey(25)

        if key == 27:
            print('Program was terminated by user..')
            sys.exit()
        elif key == ord(' '):
            print('User confirmed correct arrangement of cameras...')
            break

        # Display livestream of camera
        isTrueL, frameL = capL.read()
        #isTrueR, frameR = capR.read()

        cv.imshow('Camera L', frameL)
        #cv.imshow('Camera R', frameR)

    cv.destroyAllWindows()

#%% 1. Initiate object detection
print('__________ 1. Loading object dection model __________')
print('.')
print('.')

import tensorflow as tf
import object_detection
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
os.chdir(paths['home'])

# Determine latest checkpoint
path_model = paths['trainedModels']+'/'+custom_model_name
files = os.listdir(path_model)
last_checkpoint = 0
for file in files:
    ckpt = file.split('.')[0]
    try:
        ckpt = ckpt.split('-')[1]
    except:
        # do nothing
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

# Initiate detection
path_labelmap = path_model+'/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(path_labelmap)

print('.')
print('.')
print('__________ 1. Loading object dection model completed __________')


## 2. Calibration and rectification
print('__________ 2. Calibration and rectification __________')
print('.')
print('.')
capL = 0
capR = 0
newCameraMatrixL = cameraSetup.getIntrinsicsLeftCamera(capL, paths, singleLGenImages, singleLTestCombinations, singleLCalibration, debuggingMode)

cameraSetup.getIntrinsicsRightCamera(capR, paths, singleRGenImages, singleRTestCombinations, singleRCalibration, debuggingMode)

Trns = cameraSetup.calibrateStereoSetup(capL, capR, paths, stereoGenImages, stereoTestCombinations, stereoCalibration, debuggingMode)

leftMapX, leftMapY, rightMapX, rightMapY, Q, rectifiedCameraMatrixL = cameraSetup.getRectificationMap(capL, capR, paths, newRectificationMapping, debuggingMode)


print('.')
print('.')
print('__________ 2. Calibration and rectification completed __________')
print('.')

## 3. Correspondence (disparity map)

print('.')
print('__________ 3. Starting correspondence calculation __________')


#left_matcher, right_matcher, wls_filter = cameraSetup.initialiseCorrespondence()
# frameL = cv.imread(paths['stereoTesting']+'/camLTesting.png')
# frameR = cv.imread(paths['stereoTesting']+'/camRTesting.png')

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
                max_boxes_to_draw=1,
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






    