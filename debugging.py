from xml.dom.expatbuilder import FragmentBuilder
import cv2 as cv
import numpy as np
import os
import cameraCalibration as camcal
import sys
import depthDetectionasModules as cameraSetup
from pprint import pprint

print(cv.__main__)

sys.exit()

home_path = '/Users/niklas/Virtual_Environment/Version_5/projectAutonomous'
paths = {
        '1_Calibration': os.path.join(home_path,'1_Calibration'),
        'stereo': os.path.join(home_path,'1_Calibration/stereo'),
        'individual': os.path.join(home_path,'1_Calibration/individual'),
        'stCamL': os.path.join(home_path,'1_Calibration/stereo/camL'),
        'stCamR': os.path.join(home_path,'1_Calibration/stereo/camR'),
        'indCamR': os.path.join(home_path,'1_Calibration/individual/camR'),
        'indCamL': os.path.join(home_path,'1_Calibration/individual/camL'),
    
    }

squareSize = 2.2 #cm

## Check installed packages
# os.chdir(home_path)
# import pkg_resources
# installed_packages = pkg_resources.working_set
# installed_packages_list = sorted(["%s==%s" % (i.key, i.version)
#    for i in installed_packages])
# #print(installed_packages_list)

# for i in range(len(installed_packages_list)):
    
#     if 'opencv' in installed_packages_list[i]:
#         package = installed_packages_list[i].split('==')[0]
#         print(package)
#         %python pip -m uninstall package -y




## Step 0: Initiate camera capture

# capL = cv.VideoCapture(2)
# capR = cv.VideoCapture(0)
# while True:
#         key = cv.waitKey(25)

#         if key == 27:
#             print('Program was terminated by user..')
#             sys.exit()

#         # Display livestream of camera
#         isTrueL, frameL = capL.read()
#         isTrueR, frameR = capR.read()
#         cv.imshow('Camera L', frameL)
#         cv.imshow('Camera R', frameR)
# cv.destroyAllWindows()

## Step 2: Rectification and mapping
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


## Take one image of the scene with both cameras

path_depth_testing = '/Users/niklas/Virtual_Environment/Version_5/projectAutonomous/Depth_measurement'
imgPathL = ''
imgPathR = ''

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
        imgPathL = path_depth_testing+'/camL99.png'
        imgPathR = path_depth_testing+'/camR99.png'

        cv.imwrite(imgPathL,frameL)
        cv.imwrite(imgPathR,frameR)
        break

    cv.destroyAllWindows()


# Load image pair used for MATLAB rectification
path_img_L = '/Users/niklas/Virtual_Environment/Version_5/projectAutonomous/1_Calibration/stereo/camL/10_L.png'
path_img_R = '/Users/niklas/Virtual_Environment/Version_5/projectAutonomous/1_Calibration/stereo/camR/10_R.png'
# path_img_L = path_depth_testing+'/camL99.png'
# path_img_R = path_depth_testing+'/camR99.png'

imgL = cv.imread(path_img_L)
imgR = cv.imread(path_img_R)

gray = cv.cvtColor(imgL,cv.COLOR_BGR2GRAY)
imgSize = gray.shape[::-1]

# Create rectification maps
R_L, R_R, proj_mat_l, proj_mat_r, Q, roiL, roiR= cv.stereoRectify(newCameraMatrixL, distortionCoefficientsL, newCameraMatrixR, distortionCoefficientsR, imgSize, Rot, Trns, flags=cv.CALIB_ZERO_DISPARITY, alpha=1)

leftMapX, leftMapY = cv.initUndistortRectifyMap(newCameraMatrixL, distortionCoefficientsL, R_L, proj_mat_l, imgSize, cv.CV_32FC1)
rightMapX, rightMapY = cv.initUndistortRectifyMap(newCameraMatrixR, distortionCoefficientsR, R_R, proj_mat_r, imgSize, cv.CV_32FC1)

## Debugging rectification _________________________________________________________________________________________________________________________________________________________

# Settings for chessboard corners
font = cv.FONT_HERSHEY_PLAIN
fontScale = 4
boardSize = (9,6)
subpix_criteria = (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 100, 10e-06)
winSize = (11,11)
objp = np.zeros((1, boardSize[0]*boardSize[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:boardSize[0], 0:boardSize[1]].T.reshape(-1, 2)
objp = objp * squareSize

# Check rectified image for epipolar lines
Left_rectified = cv.remap(imgL,leftMapX,leftMapY, cv.INTER_LINEAR, cv.BORDER_CONSTANT)
Right_rectified = cv.remap(imgR,rightMapX,rightMapY, cv.INTER_LINEAR, cv.BORDER_CONSTANT)

grayL = cv.cvtColor(Left_rectified,cv.COLOR_BGR2GRAY)
grayR = cv.cvtColor(Right_rectified,cv.COLOR_BGR2GRAY)

retL, cornersL = cv.findChessboardCorners(grayL, boardSize, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)
retR, cornersR = cv.findChessboardCorners(grayR, boardSize, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)

# If chessboard is found, display rectified chessboard (best option), otherwise display any other rectified image
visb4 = np.concatenate((imgL, imgR), axis=1)
vis = np.concatenate((Left_rectified, Right_rectified), axis=1)
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
    for i in range(0, 54, 9):
        x_l = int(round(imagePointsL[0][i][0][0]))
        y_l = int(round(imagePointsL[0][i][0][1]))
        cv.circle(vis, (x_l, y_l), 7, (0,255,255), -1)
        x_r = int(round(imagePointsR[0][i][0][0]+Left_rectified.shape[1]))
        y_r = int(round(imagePointsR[0][i][0][1]))
        cv.circle(vis, (x_r, y_r), 7, (0,255,255), -1)
        slope = (y_l-y_r)/(x_r-x_l)
        slopes.append(slope)
        #cv.line(vis, (x_l,y_l), (x_r,y_r), (0,255,255), 2)
        cv.line(vis, (0,y_l), (vis.shape[1],y_l), (255,0,0), 2)

    avg = sum(slopes)/len(slopes)
    cv.putText(vis, 'Average slope '+str(avg),(vis.shape[1]//3, (vis.shape[0]//5)*4), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
    
else:
    steps = 20
    step_height = Left_rectified.shape[1] / steps
    y = 0
    for j in range(steps):
        y += step_height
        y = int(y)
        cv.line(vis, (0,y), (vis.shape[1],y), (255,0,0), 2)
# cv.imshow('Rectification check - before rectification', visb4)
# cv.imshow('Rectification check - after rectification', vis)
# cv.waitKey(0)
cv.destroyAllWindows()

    
## Disparity mapping ________________________________________________________________________________________________________________________________________________________          

minDisparity = 144
maxDisparity = 272*2
numDisparities = maxDisparity-minDisparity
blockSize = 3
disp12MaxDiff = 1
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
win_rec_left = 'Camera L rectified'
win_rec_right = 'Camera R rectified'
win_rectified_both = 'Both frames rectified'

cv.namedWindow(win_left_disp)
cv.namedWindow(win_filtered_disp)
cv.namedWindow(win_rectified_both)
cv.namedWindow(win_rec_right)

Left_rectified = grayL.copy()
Right_rectified = grayR.copy()

# Get disparity value at left centre of disparity map
print('____________Checking left disparity map & reprojected depth____________')
left_disp = left_matcher.compute(Left_rectified, Right_rectified).astype(np.float32) / 16
print('Data type: '+str(left_disp.dtype))
print('Shape: '+str(left_disp.shape))
x_centre = left_disp.shape[1] // 2
y_centre = left_disp.shape[0] // 2
dispL = left_disp[y_centre][x_centre]

disp8forPCL_LM = np.uint8(left_disp)
points_3D_LM = cv.reprojectImageTo3D(disp8forPCL_LM, Q)
Z_LM = points_3D_LM[y_centre][x_centre][2]

print('Disparity in center of frame:  {:.2f}pixels, depth: {:.2f}cm'.format(dispL, Z_LM))


# Get value at filtered centre of disparity map
print('____________Checking filtered disparity map & reprojected depth____________')
left_disp = left_matcher.compute(Left_rectified, Right_rectified).astype(np.float32) / 16
right_disp = right_matcher.compute(Right_rectified,Left_rectified).astype(np.float32) / 16
filtered_disp = wls_filter.filter(left_disp, Left_rectified, disparity_map_right=right_disp)
print('Data type: '+str(filtered_disp.dtype))
print('Shape: '+str(filtered_disp.shape))
x_centre = filtered_disp.shape[1] // 2
y_centre = filtered_disp.shape[0] // 2
dispF = filtered_disp[y_centre][x_centre]

disp8forPCL_FM = np.uint8(filtered_disp)
points_3D_FM = cv.reprojectImageTo3D(disp8forPCL_FM, Q)
Z_FM = points_3D_FM[y_centre][x_centre][2]

print('Disparity in center of frame:  {:.2f}pixels, depth: {:.2f}cm'.format(dispF, Z_FM))

font = cv.FONT_HERSHEY_PLAIN
fontScale = 3

Left_rectified = cv.remap(imgL,leftMapX,leftMapY, cv.INTER_LINEAR, cv.BORDER_CONSTANT)
Right_rectified = cv.remap(imgR,rightMapX,rightMapY, cv.INTER_LINEAR, cv.BORDER_CONSTANT)
visRectified = np.concatenate((Left_rectified, Right_rectified), axis=1)

def printCoordinatesLRView(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(visRectified, (x, y), 5, (0, 255, 255), -1)
        if x < Left_rectified.shape[1]:
            coordinates['leftImage'] = x
            temp = 'x='+str(x)+' y='+str(y)
            cv.putText(visRectified, temp, (x, y), font, fontScale, (255, 0, 0), 2, cv.LINE_AA)
        elif x >= Left_rectified.shape[1]:
            coordinates['rightImage'] = x-Left_rectified.shape[1]
            temp = 'x='+str(x-Left_rectified.shape[1])+' y='+str(y)
            cv.putText(visRectified, temp, (x, y), font, fontScale, (255, 0, 0), 2, cv.LINE_AA)


def printCoordinatesdispL(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        dispLM = left_disp[y_centre][x_centre].astype(np.float32) / 16
        dispLM = str(dispLM)+' pixels'

        z_L = ((-Trns[0][0])*cameraMatrixL[0][0])/(dispLM)

        points_3D_FM[y][x][2]

        #string = 'MANUAL: disparity: '+str(dispLM)+'pixels , depth: '+str(z_L)+' pixels?'
        #cv.putText(left_disp, string, (x, y), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
        print('Disparity = {}, MANUAL depth = {}, BUILTIN depth = {}'.format(dispLM, z_L, ))
        print('BUILTIN: disparity = {}, depth = {}')

def printCoordinatesdispF(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        dispFM = filtered_disp[y][x].astype(np.float32) / 16
        dispFM = str(dispFM)+' pixels'

        z_L = ((-Trns[0][0])*cameraMatrixL[0][0])/(dispFM)

        string = 'disparity: '+str(dispFM)+'pixels , depth: '+str(z_F)+' pixels?'
        cv.putText(filtered_disp, string, (x, y), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)

#cv.setMouseCallback(win_filtered_disp, printCoordinatesdispF)
#cv.setMouseCallback(win_left_disp, printCoordinatesdispL)
cv.setMouseCallback(win_rectified_both, printCoordinatesLRView)

coordinates = {'leftImage': 0, 'rightImage':0} 
while True:

    key = cv.waitKey(25)

    if key == 27:
        print('Program was terminated by user..')
        break

    #cv.imshow(win_left_disp, left_disp)
   # cv.imshow(win_filtered_disp, filtered_disp)

    cv.imshow(win_rectified_both, visRectified)
    if coordinates['leftImage'] != 0 and coordinates['rightImage'] != 0:
        print('Coordinates left: x={xl}, right: x={xr}. Delta={delta} [pixels]'.format(xl=coordinates['leftImage'], xr=coordinates['rightImage'], delta=coordinates['leftImage']-coordinates['rightImage'] ))
        coordinates['leftImage'] = 0
        coordinates['rightImage'] = 0


cv.destroyAllWindows()


# left_disp = left_matcher.compute(Left_rectified, Right_rectified)#.astype(np.float32) / 16
# #cv.imshow('Left Disparity', left_disp) 
# right_disp = right_matcher.compute(Right_rectified,Left_rectified)#.astype(np.float32) / 16
# #cv.imshow('Right Disparity', right_disp) 
# filtered_disp = wls_filter.filter(left_disp, Left_rectified, disparity_map_right=right_disp)
# filtered_disp = cv.normalize(src=filtered_disp, dst=filtered_disp, beta=0, alpha=255, norm_type=cv.NORM_MINMAX);
# filtered_disp = np.uint8(filtered_disp)
# cv.imshow('Disparity', filtered_disp) 

# disp8forPCL = np.uint8(filtered_disp)
# cv.imshow('8bit Disparity', disp8forPCL)
# colored = cv.applyColorMap(disp8forPCL, cv.COLORMAP_JET)
# cv.imshow('Coloured Disparity', colored)

# filtered_disp = filtered_disp.astype(np.float32) / 16
# cv.imshow('Scaled Filtered_Disparity', filtered_disp) 

# points_3D = cv.reprojectImageTo3D(disp8forPCL, Q)

# cv.waitKey(0)
sys.exit()
temp = filtered_disp.astype(np.float32) / 16
points_3D1 = cv.reprojectImageTo3D(temp, Q)
    

# Compute left and right disparity, scale to float and divide by 16, then apply filter
left_disp_test = left_disp.astype(np.float32) / 16
right_disp_test = right_disp.astype(np.float32) / 16
filtered_disp_test = wls_filter.filter(left_disp_test, Left_rectified, disparity_map_right=right_disp_test)
disp8forPCL2 = np.uint8(filtered_disp_test)
cv.imshow('Disparity Test', disp8forPCL2) 

temp2 = filtered_disp_test.astype(np.float32) / 16
points_3D2 = cv.reprojectImageTo3D(temp2, Q)


# Coloured version of disparity map 1
##___
## Check why dividing and scaling the disparity map after filtering results in a different image than when scaling and dividing before filtering...
#filtered_disp = filtered_disp.astype(np.float32) / 16
## ___
disp8forPCL1 = np.uint8(filtered_disp)
colored1 = cv.applyColorMap(disp8forPCL1, cv.COLORMAP_JET)
cv.imshow('Disparity  Coloured', colored1) 

# # Coloured version of disparity map 2

# Use 8bit unsigned integer disparity display (0...255) for color mapping
colored2 = cv.applyColorMap(disp8forPCL2, cv.COLORMAP_JET)
cv.imshow('Disparity Test Coloured', colored2)

cv.waitKey(0)


    #filtered_disp_test = cv.normalize(src=filtered_disp_test, dst=filtered_disp_test, beta=0, alpha=255, norm_type=cv.NORM_MINMAX);
    #filtered_disp_test = np.uint8(filtered_disp_test)

    # Reproject 3D points using NOT NORMALIZED version of disparity map that is scaled to float and divided by 16
    # temp = filtered_disp.astype(np.float32) / 16
    # points_3D = cv.reprojectImageTo3D(temp, Q)

    # Normalize the values to a range from 0..255 for a grayscale image
    # #cv.normalize(filtered_disp, filtered_disp, alpha=255, beta=0, norm_type=cv.NORM_MINMAX)
    



    

    



#cv.imshow('Disparity Display', disparity_map)   

sys.exit()