from xml.dom.expatbuilder import FragmentBuilder
import cv2 as cv
import numpy as np
import os
import cameraCalibration as camcal
import sys
import depthDetectionasModules as cameraSetup
from pprint import pprint

home_path = '/Users/niklas/Virtual_Environment/Version_5/projectAutonomous'
custom_model_name = 'my_ssd_mobilenet_v2_fpnlite'

labels = ['Engine']

## Installation________________________________________________________________________________________________________

# Import fundamental dependencies
#pip install GitPython
import os
import sys
import time
import pathlib
import shutil
import math
import uuid

# Setup folder structure
import setup
files, paths = setup.installPackages(home_path, labels, firstInstallation=True)
sys.exit()


squareSize = 2.2 #cm

## Step 0: Initiate camera capture

capL = cv.VideoCapture(0)
capR = cv.VideoCapture(2)
# while True:
#         key = cv.waitKey(25)

#         if key == 27:
#             print('Program was terminated by user..')
#             sys.exit()
#         elif key == ' ':
#             break

#         # Display livestream of camera
#         isTrueL, frameL = capL.read()
#         isTrueR, frameR = capR.read()
#         cv.imshow('Camera L', frameL)
#         cv.imshow('Camera R', frameR)

# cv.destroyAllWindows()

## Step 1: Rectification and mapping

# Load intrinsics and extrinics from calibration process
cameraMatrixL = np.load(paths['individual']+'/cameraMatrixL.npy')
newCameraMatrixL = np.load(paths['individual']+'/newCameraMatrixL.npy')
distortionCoefficientsL = np.load(paths['individual']+'/distortionCoefficientsL.npy')
cameraMatrixR = np.load(paths['individual']+'/cameraMatrixR.npy')
newCameraMatrixR = np.load(paths['individual']+'/newCameraMatrixR.npy')
distortionCoefficientsR = np.load(paths['individual']+'/distortionCoefficientsR.npy')


Rot = np.load(paths['stereo']+'/rotationVector.npy')
Trns = np.load(paths['stereo']+'/translationVector.npy')
Emat = np.load(paths['stereo']+'/essentialMatrixE.npy')
Fmat = np.load(paths['stereo']+'/fundamentalMatrixF.npy')
print('Loaded intrinsics and extriniscs matrices')

# Create rectification maps using images from stereo calibration
path_img_L = '/Users/niklas/Virtual_Environment/Version_5/projectAutonomous/1_Calibration/stereo/camL/7_L.png'
path_img_R = '/Users/niklas/Virtual_Environment/Version_5/projectAutonomous/1_Calibration/stereo/camR/7_R.png'
imgL = cv.imread(path_img_L)
imgR = cv.imread(path_img_R)

gray = cv.cvtColor(imgL,cv.COLOR_BGR2GRAY)
imgSize = gray.shape[::-1]


R_L, R_R, proj_mat_l, proj_mat_r, Q, roiL, roiR= cv.stereoRectify(newCameraMatrixL, distortionCoefficientsL, newCameraMatrixR, distortionCoefficientsR, imgSize, Rot, Trns, flags=cv.CALIB_ZERO_DISPARITY , alpha=1)


leftMapX, leftMapY = cv.initUndistortRectifyMap(newCameraMatrixL, distortionCoefficientsL, R_L, proj_mat_l, imgSize, cv.CV_32FC1)
rightMapX, rightMapY = cv.initUndistortRectifyMap(newCameraMatrixR, distortionCoefficientsR, R_R, proj_mat_r, imgSize, cv.CV_32FC1)

## Debugging rectification _________________________________________________________________________________________________________________________________________________________

# Take an image of a horizontal chessboard for debugging purposes
path_depth_testing = '/Users/niklas/Virtual_Environment/Version_5/projectAutonomous/Depth_measurement'

takeTestPhoto = False
while True and (takeTestPhoto is True) :

    key = cv.waitKey(5)

    if key == 27:
        print('Program was terminated by user..')
        sys.exit()

    isTrueL, frameL = capL.read()
    isTrueR, frameR = capR.read()
    cv.imshow('Cam R ', frameR)
    temp = frameL.copy()

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
    
    cv.imshow('Cam L ', temp)

    if key == ord(' '):
        imgPathL = path_depth_testing+'/camL100.png'
        imgPathR = path_depth_testing+'/camR100.png'

        cv.imwrite(imgPathL,frameL)
        cv.imwrite(imgPathR,frameR)
        break

cv.destroyAllWindows()

# Load and rectify the image pair
imgL = cv.imread(path_depth_testing+'/camL99.png')
imgR = cv.imread(path_depth_testing+'/camR99.png')

# path_img_L = '/Users/niklas/Virtual_Environment/Version_5/projectAutonomous/1_Calibration/stereo/camL/7_L.png'
# path_img_R = '/Users/niklas/Virtual_Environment/Version_5/projectAutonomous/1_Calibration/stereo/camR/7_R.png'

Left_rectified = cv.remap(imgL,leftMapX,leftMapY, cv.INTER_LINEAR, cv.BORDER_CONSTANT)
Right_rectified = cv.remap(imgR,rightMapX,rightMapY, cv.INTER_LINEAR, cv.BORDER_CONSTANT)

grayL = cv.cvtColor(Left_rectified,cv.COLOR_BGR2GRAY)
grayR = cv.cvtColor(Right_rectified,cv.COLOR_BGR2GRAY)

# Settings for chessboard corners
font = cv.FONT_HERSHEY_PLAIN
fontScale = 2
boardSize = (9,6)
subpix_criteria = (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 100, 10e-06)
winSize = (11,11)
objp = np.zeros((1, boardSize[0]*boardSize[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:boardSize[0], 0:boardSize[1]].T.reshape(-1, 2)
objp = objp * squareSize

# Find chessboard corners in rectified image pair
retL, cornersL = cv.findChessboardCorners(grayL, boardSize, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)
retR, cornersR = cv.findChessboardCorners(grayR, boardSize, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)

# If chessboard is found, display rectified chessboard (best option), otherwise display any other rectified image
vis_beforeRectification = np.concatenate((imgL, imgR), axis=1)
vis_afterRectification = np.concatenate((Left_rectified, Right_rectified), axis=1)

pointsL = []
if retR is True and retL is True:
    slopes = []
    objectPoints = []
    imagePointsL = [] 
    imagePointsR = [] 
    objectPoints.append(objp)
    cv.cornerSubPix(grayR, cornersR,(3,3),(-1,-1),subpix_criteria)
    cv.cornerSubPix(grayL, cornersL,(3,3),(-1,-1),subpix_criteria)
    imagePointsR.append(cornersR)
    imagePointsL.append(cornersL)
    
    # Get points in 4th row (vertical centre) and display them
    for i in range(0,54,10):
        x_l = int(round(imagePointsL[0][i][0][0]))
        x_l_text = 'x_l= '+str(x_l)+'px'
        y_l = int(round(imagePointsL[0][i][0][1]))
        cv.circle(vis_afterRectification, (x_l, y_l), 5, (0,255,255), -1)
        cv.putText(vis_afterRectification, x_l_text, (x_l+5, y_l-5), font, fontScale, (255, 0, 0), 2, cv.LINE_AA)

        temp = [x_l, y_l]
        pointsL.append(temp.copy())
    
        x_r = int(round(imagePointsR[0][i][0][0]))
        x_r_text = 'x_r= '+str(x_r)+'px'
        y_r = int(round(imagePointsR[0][i][0][1]))
        cv.circle(vis_afterRectification, (x_r+Left_rectified.shape[1], y_r), 5, (0,255,255), -1)
        cv.putText(vis_afterRectification, x_r_text, (x_r+Left_rectified.shape[1]+5, y_r-5), font, fontScale, (255, 0, 0), 2, cv.LINE_AA)
        
        cv.line(vis_afterRectification, (x_l,y_l), (x_r+Left_rectified.shape[1],y_r), (0,255,255), 2)
        disp_x = x_l - x_r
        depthcamL  = ((-Trns[0][0])*cameraMatrixL[0][0])/(disp_x)
        disp_x_text = ('Camera L disparity: {}px, depth: {:.2f}cm'.format(disp_x, depthcamL))
        cv.putText(vis_afterRectification, disp_x_text, (x_l + int((x_r+Left_rectified.shape[1]-x_l)/2), y_r -10), font, fontScale, (255, 0, 0), 2, cv.LINE_AA)

        slope = (y_l-y_r)/(x_r+Left_rectified.shape[1]-x_l)
        slopes.append(slope)

    avg = sum(slopes)/len(slopes)
    cv.putText(vis_afterRectification, 'Average slope '+str(avg),(vis_afterRectification.shape[1]//3, (vis_afterRectification.shape[0]//5)*4), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
    
else:
    print('No chessboard found!')


# cv.imshow('Rectification check - before rectification', vis_beforeRectification)
cv.imshow('Rectification check - after rectification', vis_afterRectification)
    
## Disparity mapping ________________________________________________________________________________________________________________________________________________________          

minDisparity = 144
maxDisparity = 272*2
numDisparities = maxDisparity-minDisparity
blockSize = 3
disp12MaxDiff = 5
uniquenessRatio = 15

left_matcher = cv.StereoSGBM_create(minDisparity = minDisparity,
        numDisparities = numDisparities,
        blockSize = blockSize,
        disp12MaxDiff = disp12MaxDiff,
        uniquenessRatio = uniquenessRatio,

)

sigma = 1.5
lmbda = 8000.0

right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
wls_filter = cv.ximgproc.createDisparityWLSFilter(left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

## Disparity debugging___________________________________________________________________________________________________________________________          

win_left_disp = 'Left disparity map'
win_filtered_disp = 'Filtered disparity map'

cv.namedWindow(win_left_disp)
cv.namedWindow(win_filtered_disp)

Left_rectified = grayL.copy()
Right_rectified = grayR.copy()

# Get disparity value at centre of image, which should align with one of the chessboard corners marked in the previous step
font = cv.FONT_HERSHEY_PLAIN
fontScale = 2


print('____________Checking left disparity map & reprojected depth____________')
left_disp = left_matcher.compute(Left_rectified, Right_rectified).astype(np.float32) / 16
print('Data type: '+str(left_disp.dtype))
print('Shape: '+str(left_disp.shape))

points_3D_LM = cv.reprojectImageTo3D(left_disp, Q)


print('____________Checking filtered disparity map & reprojected depth____________')
right_disp = right_matcher.compute(Right_rectified,Left_rectified).astype(np.float32) / 16
filtered_disp = wls_filter.filter(left_disp, Left_rectified, disparity_map_right=right_disp)
print('Data type: '+str(filtered_disp.dtype))
print('Shape: '+str(filtered_disp.shape))

points_3D_FM = cv.reprojectImageTo3D(filtered_disp, Q)

def printCoordinatesdispLM(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        dispLM = left_disp[y][x]

        depthLM = points_3D_LM[y][x][2]

        text = ('Disparity: {:.2f}px, depth: {:.2f}cm'.format(dispLM, depthLM))
        cv.putText(left_disp_forVis, text, (x, y), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)


def printCoordinatesdispFM(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:

        dispFM = filtered_disp[y][x]

        depthFM = points_3D_FM[y][x][2]

        text = ('Disparity: {:.2f}px, depth: {:.2f}cm'.format(dispFM, depthFM))
        cv.putText(filtered_disp_forVis, text, (x, y), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)

cv.setMouseCallback(win_filtered_disp, printCoordinatesdispFM)
cv.setMouseCallback(win_left_disp, printCoordinatesdispLM)

temp_right = right_matcher.compute(Right_rectified,Left_rectified)
temp_left = left_matcher.compute(Left_rectified, Right_rectified)
filtered_disp_forVis = wls_filter.filter(temp_left, Left_rectified, disparity_map_right=temp_right)

dispForColor = filtered_disp.copy()

dispForColor = cv.normalize(src=dispForColor, dst=dispForColor, alpha=255, beta=0 , norm_type=cv.NORM_MINMAX)
disp8 = np.uint8(dispForColor)

colored = cv.applyColorMap(disp8, cv.COLORMAP_JET)

# mask_map = filtered_disp_forVis > filtered_disp_forVis.min()
# colors = cv.cvtColor(Left_rectified, cv.COLOR_BGR2RGB)
# output_points = points_3D_LM[mask_map]
# output_colors = colors[mask_map]

left_disp_forVis = left_matcher.compute(Left_rectified, Right_rectified)

# Proof that coordinates in left rectified and leftdisp are the same
# for n in range(len(imagePointsL[0])):
#     x=int(round(imagePointsL[0][n][0][0]))
#     y=int(round(imagePointsL[0][n][0][1]))
#     cv.circle(left_disp_forVis, (x,y), 5, (0,0,0), -1)


while True:

    key = cv.waitKey(25)

    if key == 27:
        print('Program was terminated by user..')
        break

    cv.imshow(win_left_disp, left_disp_forVis)
    #cv.imshow(win_filtered_disp, filtered_disp_forVis)
    #cv.imshow('Coloured Disparity', colored)


cv.destroyAllWindows()