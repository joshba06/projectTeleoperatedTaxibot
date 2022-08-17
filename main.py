## User settings___________________________________________________________________
home_path = '/Users/niklas/Virtual_Environment/Version_5/projectAutonomous'
custom_model_name = 'my_ssd_mobilenet_v2_fpnlite'

labels = ['Engine']

capL = cv.VideoCapture(0)
capR = cv.VideoCapture(2)

debuggingMode = True

stereoGenImages = False
singleLGenImages = False
singleRGenImages = False

stereoTestCombinations = False
singleLTestCombinations = False
singleRTestCombinations = False

singleLCalibration = True
singleRCalibration = False
stereoCalibration = False

newRectificationMapping = False

showAllSteps = False
showCams = False

## Installation ___________________________________________________________________

# Import fundamental dependencies
#pip install GitPython
import os
import sys
import time
import pathlib
import shutil
import math
import uuid
from pprint import pprint


# Setup folder structure and install extented packages
import setup
files, paths = setup.installPackages(home_path, labels, firstInstallation=True)

import cameraCalibration as camcal

# Import newly installed dependencies
import cv2 as cv
import numpy as np
import tensorflow as tf
os.chdir(paths['research'])
import object_detection
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
os.chdir(paths['home'])

import cameraCalibrationNew as cameraSetup

## 0. Check if cameras are displayed correctly_______________________________________
if debuggingMode is True:

    while True:
        key = cv.waitKey(25)

        if key == 27:
            print('Program was terminated by user..')
            sys.exit()
        elif key == ' ':
            print('User confirmed correct arrangement of cameras...')

        # Display livestream of camera
        isTrueL, frameL = capL.read()
        isTrueR, frameR = capR.read()
        cv.imshow('Camera L', frameL)
        cv.imshow('Camera R', frameR)
    cv.destroyAllWindows()

## 1. Calibration and rectification
print('__________ 1. Calibration and rectification __________')
print('.')
print('.')

cameraSetup.getIntrinsicsLeftCamera(capL, paths, singleLGenImages, singleLTestCombinations, singleLCalibration, debuggingMode)
cameraSetup.getIntrinsicsRightCamera(capR, paths, singleRGenImages, singleRTestCombinations, singleRCalibration, debuggingMode)
cameraSetup.calibrateStereoSetup(capL, capR, paths, stereoGenImages, stereoTestCombinations, stereoCalibration, debuggingMode)
leftMapX, leftMapY, rightMapX, rightMapY, Q = cameraSetup.getRectificationMap(capL, capR, paths, newRectificationMapping, debuggingMode)


print('.')
print('.')
print('__________ 1. Calibration and rectification completed __________')
print('.')

## 2. Correspondence (disparity map)

print('.')
print('__________ Starting correspondence calculation __________')

disparityMap, pointCloud = cameraSetup.initiateDepthDetection(Left_rectified, Right_rectified, Q, paths, debuggingMode)



## 2. Load object detection model

# Determine latest checkpoint
path_checkpoints = paths['3_Output']+'/'+custom_model_name
files = os.listdir(path_checkpoints)
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
path_pipeline = paths['3_Output']+'/'+custom_model_name+'/pipeline.config'
configs = config_util.get_configs_from_pipeline_file(path_pipeline)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(path_checkpoints, str(last_checkpoint))).expect_partial()

#@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

# Initiate detection
path_labelmap = paths['3_Output']+'/'+custom_model_name+'/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(path_labelmap)

#%% 2. Prepare stereo parameters

# Load camera intrinsics
cameraMatrixL, newCameraMatrixL, distortionCoefficientsL = cameraSetup.getIntrinsicsLeft(paths)
cameraMatrixR, newCameraMatrixR, distortionCoefficientsR = cameraSetup.getIntrinsicsRight(paths)
print('Imported camera intrinsics matrices...')


# Load geometric stereo properties
Rot, Trns, Emat, Fmat = cameraSetup.getStereoProperties(paths, cameraMatrixL, newCameraMatrixL, distortionCoefficientsL, cameraMatrixR, newCameraMatrixR, distortionCoefficientsR)
print('Imported geometric stereo parameters...')

# Load rectification map
Left_Stereo_Map, Right_Stereo_Map, Q = cameraSetup.getRectificationMap(paths, cameraMatrixL, newCameraMatrixL, distortionCoefficientsL, cameraMatrixR, newCameraMatrixR, distortionCoefficientsR, Rot, Trns, Emat, Fmat)
print('Imported rectification maps...')


#%% 3 Create instance of stereo
minDisparity = 80 #144 
maxDisparity = 272 #544
numDisparities = maxDisparity-minDisparity
blockSize = 3
disp12MaxDiff = 1
uniquenessRatio = 10

left_matcher = cv.StereoSGBM_create(minDisparity = minDisparity,
        numDisparities = numDisparities,
        blockSize = blockSize,
        disp12MaxDiff = disp12MaxDiff,
        uniquenessRatio = uniquenessRatio,

)
# left_matcher = cv.StereoBM_create(numDisparities = numDisparities, blockSize=blockSize)

sigma = 1.5
lmbda = 8000.0

right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
wls_filter = cv.ximgproc.createDisparityWLSFilter(left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)


while True:

    ## Display live camera feed
    isTrueL, frameL = capL.read()
    isTrueR, frameR = capR.read()
    width = int(capL.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(capL.get(cv.CAP_PROP_FRAME_HEIGHT))

    key = cv.waitKey(25)

    if key == 27:
        print('Program was terminated by user..')
        break

    Left_rectified= cv.remap(frameL,Left_Stereo_Map[0],Left_Stereo_Map[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    Right_rectified= cv.remap(frameR,Right_Stereo_Map[0],Right_Stereo_Map[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    grayL = cv.cvtColor(Left_rectified,cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(Right_rectified,cv.COLOR_BGR2GRAY)

    ## Display disparity map
    left_disp = left_matcher.compute(grayL, grayR)
    right_disp = right_matcher.compute(grayR,grayL)
    cv.imshow('Left disparity', left_disp)
    cv.imshow('Right disparity', right_disp)
    filtered_disp = wls_filter.filter(left_disp, grayL, disparity_map_right=right_disp)
    temp = filtered_disp.copy()

    x1 = temp.shape[1]//2
    y1 = 0
    x2 = temp.shape[1]//2
    y2 = temp.shape[0]

    x3 = 0
    y3 = temp.shape[0]//2
    x4 = temp.shape[1]
    y4 = temp.shape[0]//2

    yellow = [0, 255, 255]

    cv.line(temp, (x1,y1), (x2,y2), yellow, thickness=2)
    cv.line(temp, (x3,y3), (x4,y4), yellow, thickness=2)

    cv.imshow('Filtered disparity with crosshairs', temp) 

    ## Compute 3D
    filtered_disp_3D = filtered_disp.astype(np.float32) / 16
    points_3D = cv.reprojectImageTo3D(filtered_disp_3D, Q)

    X_bi = points_3D[temp.shape[0]//2][temp.shape[1]//2][0]
    Y_bi = points_3D[temp.shape[0]//2][temp.shape[1]//2][1]
    Z_bi = points_3D[temp.shape[0]//2][temp.shape[1]//2][2]

    print('Coordinates from left camera: x = {:.2f}mm, y = {:.2f}mm, z = {:.2f}mm.'.format(X_bi, Y_bi, Z_bi))

    
    ## Display colored disparity map
    # left_disp = left_matcher.compute(grayL, grayR).astype(np.float32) / 16
    # right_disp = right_matcher.compute(grayR,grayL).astype(np.float32) / 16
    # filtered_disp_col = wls_filter.filter(left_disp, grayL, disparity_map_right=right_disp)
    # filtered_disp_col = cv.normalize(filtered_disp_col, filtered_disp_col, alpha=255, beta=0, norm_type=cv.NORM_MINMAX)
    # filtered_disp_col = np.uint8(filtered_disp_col)
    # colored = cv.applyColorMap(filtered_disp_col, cv.COLORMAP_JET)
    # cv.imshow('Colored disparity', colored)   


    # left_disp = left_matcher.compute (grayL, grayR).astype(np.float32) / 16
    # right_disp = right_matcher.compute(grayR,grayL).astype(np.float32) / 16
    # filtered_disp = wls_filter.filter(left_disp, grayL, disparity_map_right=right_disp)#.astype(np.float32) / 16
    # #disp8forPCL = np.uint8(filtered_disp)
    # points_3D = cv.reprojectImageTo3D (filtered_disp, Q)

    ## Detect object on left rectified camera stream
    # image_np = np.array(Left_rectified)
    # input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    # detections = detect_fn(input_tensor)
    
    # num_detections = int(detections.pop('num_detections'))
    # detections = {key: value[0, :num_detections].numpy()
    #               for key, value in detections.items()}
    # detections['num_detections'] = num_detections
    # detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    # label_id_offset = 1
    # image_np_with_detections = image_np.copy()

    # image_np_with_detections, aryfound = viz_utils.visualize_boxes_and_labels_on_image_array(
    #             image_np_with_detections,
    #             detections['detection_boxes'],
    #             detections['detection_classes']+label_id_offset,
    #             detections['detection_scores'],
    #             category_index,
    #             use_normalized_coordinates=True,
    #             max_boxes_to_draw=5,
    #             min_score_thresh=.8,
    #             agnostic_mode=False)
   
    # cv.imshow('Camera Left - Detection Mode', image_np_with_detections)
    # if aryfound != 0:
        # xmin = aryfound['xmin']*width
        # xmax = aryfound['xmax']*width
        # ymin = aryfound['ymin']*height
        # ymax = aryfound['ymax']*height
        # #print('Found {} at location x={}pixels, y={}pixels'.format(aryfound['class'], xmin, ymin))

        # # Location (centre of object) as seen from left camera
        # x_centre = xmin+0.5*(xmax-xmin)
        # x_centre = int(x_centre)
        # y_centre = ymin+0.5*(ymax-ymin)
        # y_centre = int(y_centre)

        # # Manual disparity and depth calculation (for center of bbox)
        # disp = filtered_disp[y_centre][x_centre]

        # Z = ((-Trns[0][0])*cameraMatrixL[0][0])/(disp)
        # # from u = fx*(X/Z)+cx -> X = (Z/fx)*(u-cx)
        # X = (Z/cameraMatrixL[0][0])*(x_centre-cameraMatrixL[0][2])
        # Y = (Z/cameraMatrixL[1][1])*(y_centre-cameraMatrixL[1][2])
    
        # X_bi = points_3D[y_centre][x_centre][0]
        # Y_bi = points_3D[y_centre][x_centre][1]
        # Z_bi = points_3D[y_centre][x_centre][2]

        # # print('Coordinates f.l.c: x_m = {:.2f}mm, y_m = {:.2f}mm, z_m = {:.2f}mm. Disparity: {} pixels'.format(X, Y, Z, disp))
        # print('Coordinates f.l.c: x_b = {:.2f}mm, y_b = {:.2f}mm, z_b = {:.2f}mm.'.format(X_bi, Y_bi, Z_bi))

        # Z = int(Z)
        # Z = str(Z)+' mm'
        # font = cv.FONT_HERSHEY_PLAIN
        # fontScale = 3

        # cv.putText(image_np_with_detections, Z, (x_centre, y_centre), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
        # cv.imshow('Camera Left - Detection Mode', image_np_with_detections)

        # Disparity for colored map
        # left_disp_col = left_disp.astype(np.float32) / 16
        # right_disp_col = right_disp.astype(np.float32) / 16
        # filtered_disp_col = wls_filter.filter(left_disp_col, grayL, disparity_map_right=right_disp_col)

        # # Use 8bit unsigned integer disparity display (0...255) for color mapping
        # filtered_disp_col = cv.normalize(filtered_disp_col, filtered_disp_col, alpha=255, beta=0, norm_type=cv.NORM_MINMAX)
        # disp8forColor = np.uint8(filtered_disp_col)
        # colored = cv.applyColorMap(disp8forColor, cv.COLORMAP_JET)
        # cv.imshow('Disparity Coloured', colored) 



cv.destroyAllWindows()






    