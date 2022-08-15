from xml.dom.expatbuilder import FragmentBuilder
import cv2 as cv
import numpy as np
import os
import cameraCalibration as camcal
import sys
import depthDetectionasModules as cameraSetup
from pprint import pprint

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

print('__________ Step 2: Stereo rectification and mapping __________')
print('.')
print('.')


# Load camera intrinsics
distortionCoefficientsL = np.array([0.1112, -0.2270, 0.0014, 7.5801e-04, 0.0835])
cameraMatrixL = np.array([[1384.3, 0, 933.5327], [0, 1383.2, 532.1460], [0, 0, 1]])
newCameraMatrixL = cameraMatrixL

distortionCoefficientsR = np.array([0.0362, -0.1640, -2.2236e-04, 3.4982e-04, 0.1148])
cameraMatrixR = np.array([[1417.1, 0, 972.7481], [0, 1418.0, 542.9659], [0, 0, 1]])
newCameraMatrixR = cameraMatrixR

print('Imported camera intrinsics matrices...')


Rot = np.array([[0.9999, 0.0109, 0.0068],[-0.0111, 0.9998, 0.0178],[-0.0066, -0.0179, 0.9998]])
Trns = np.array([[-96.5080], [-1.0640], [-0.8036]])
Emat = np.array([[0.0015, 0.7844, -1.0782],[-0.1459, 1.7298, 96.4957],[0.0084, -96.4985, 1.7210]])
Fmat = np.array([[7.8440e-10, 4.0019e-07, -9.7456e-04],[-7.4317e-08, 8.8188e-07, 0.0677],[4.5630e-05, -0.0706, 3.0555]])

# Load rectification map

path_img_L = '/Users/niklas/Virtual_Environment/Version_5/projectAutonomous/1_Calibration/stereo/camL/10_L.png'
path_img_R = '/Users/niklas/Virtual_Environment/Version_5/projectAutonomous/1_Calibration/stereo/camR/10_R.png'
imgL = cv.imread(path_img_L)
imgR = cv.imread(path_img_R)

grayL = cv.cvtColor(imgL,cv.COLOR_BGR2GRAY)
grayR = cv.cvtColor(imgR,cv.COLOR_BGR2GRAY)

imgSize = grayL.shape[::-1]

R_L, R_R, proj_mat_l, proj_mat_r, Q, roiL, roiR= cv.stereoRectify(newCameraMatrixL, distortionCoefficientsL, newCameraMatrixR, distortionCoefficientsR, imgSize, Rot, Trns, flags=cv.CALIB_ZERO_DISPARITY, alpha=1)

leftMapX, leftMapY = cv.initUndistortRectifyMap(newCameraMatrixL, distortionCoefficientsL, R_L, proj_mat_l, imgSize, cv.CV_32FC1)
rightMapX, rightMapY = cv.initUndistortRectifyMap(newCameraMatrixR, distortionCoefficientsR, R_R, proj_mat_r, imgSize, cv.CV_32FC1)
Left_rectified = cv.remap(imgL,leftMapX,leftMapY, cv.INTER_LINEAR, cv.BORDER_CONSTANT)
Right_rectified = cv.remap(imgR,rightMapX,rightMapY, cv.INTER_LINEAR, cv.BORDER_CONSTANT)

cv.imshow('Left rectified', Left_rectified)
cv.imshow('Right rectified', Right_rectified)
cv.waitKey(0)

grayL = cv.cvtColor(Left_rectified,cv.COLOR_BGR2GRAY)
grayR = cv.cvtColor(Right_rectified,cv.COLOR_BGR2GRAY)

font = cv.FONT_HERSHEY_PLAIN
fontScale = 4

# Find all chessboard corners at subpixel accuracy
boardSize = (6,9)
subpix_criteria = (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 100, 10e-06)
winSize = (11,11)

retL, cornersL = cv.findChessboardCorners(grayL, boardSize, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)
retR, cornersR = cv.findChessboardCorners(grayR, boardSize, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)

objp = np.zeros((1, boardSize[0]*boardSize[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:boardSize[0], 0:boardSize[1]].T.reshape(-1, 2)
objectPoints = []
imagePointsL = [] 
imagePointsR = [] 

slopes = []
if retR is True and retL is True:
    objectPoints.append(objp)
    cv.cornerSubPix(grayR, cornersR,(3,3),(-1,-1),subpix_criteria)
    cv.cornerSubPix(grayL, cornersL,(3,3),(-1,-1),subpix_criteria)
    imagePointsR.append(cornersR)
    imagePointsL.append(cornersL)
    
    # Get points in 4th row (vertical centre) and display them
    vis = np.concatenate((Left_rectified, Right_rectified), axis=1)
    for i in range(24,30):
        x_l = int(round(imagePointsL[0][i][0][0]))
        y_l = int(round(imagePointsL[0][i][0][1]))
        cv.circle(vis, (x_l, y_l), 7, (0,255,255), -1)
        x_r = int(round(imagePointsR[0][i][0][0]+Left_rectified.shape[1]))
        y_r = int(round(imagePointsR[0][i][0][1]))
        cv.circle(vis, (x_r, y_r), 7, (0,255,255), -1)
        slope = (y_l-y_r)/(x_r-x_l)
        slopes.append(slope)
        cv.line(vis, (x_l,y_l), (x_r,y_r), (0,255,255), 2)

    avg = sum(slopes)/len(slopes)
    cv.putText(vis, 'Average slope '+str(avg),(vis.shape[1]//3, (vis.shape[0]//5)*4), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
    cv.imshow('Rectification check - remapped images', vis)
    
cv.waitKey(0)
cv.destroyAllWindows()


print('.')
print('.')
print('__________ Finished stereo rectification and mapping __________')



## Step 5: Create disparity and depth map from scenery
print('__________ Starting disparity and depth mapping __________')
print('.')
print('.')

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
        imgPathL = path_depth_testing+'/camL5.png'
        imgPathR = path_depth_testing+'/camR5.png'

        cv.imwrite(imgPathL,frameL)
        cv.imwrite(imgPathR,frameR)
        break
cv.destroyAllWindows()

# Rectify images
imgL = cv.imread(path_depth_testing+'/camL5.png')
imgR = cv.imread(path_depth_testing+'/camR5.png')

Left_rectified= cv.remap(imgL,leftMapX,leftMapY, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
Right_rectified= cv.remap(imgR,rightMapX, rightMapY, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)

grayL = cv.cvtColor(Left_rectified,cv.COLOR_BGR2GRAY)
grayR = cv.cvtColor(Right_rectified,cv.COLOR_BGR2GRAY)


## Debug rectification
from pprint import pprint

font = cv.FONT_HERSHEY_PLAIN
fontScale = 4

# Find all chessboard corners at subpixel accuracy
boardSize = (6,9)
subpix_criteria = (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
winSize = (11,11)

retL, cornersL = cv.findChessboardCorners(grayL, boardSize, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)
retR, cornersR = cv.findChessboardCorners(grayR, boardSize, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)

objp = np.zeros((1, boardSize[0]*boardSize[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:boardSize[0], 0:boardSize[1]].T.reshape(-1, 2)
objectPoints = []
imagePointsL = [] 
imagePointsR = [] 

slopes = []
# If found, add object points, image points (after refining them)
if retR is True and retL is True:
    objectPoints.append(objp)
    cv.cornerSubPix(grayR, cornersR,(3,3),(-1,-1),subpix_criteria)
    cv.cornerSubPix(grayL, cornersL,(3,3),(-1,-1),subpix_criteria)
    imagePointsR.append(cornersR)
    imagePointsL.append(cornersL)
    
    # Get points in 4th row (vertical centre) and display them
    vis = np.concatenate((Left_rectified, Right_rectified), axis=1)
    for i in range(24,30):
        x_l = int(round(imagePointsL[0][i][0][0]))
        y_l = int(round(imagePointsL[0][i][0][1]))
        #print('Left: x = {} pixels, y = {} pixels'.format(x_l, y_l))
        cv.circle(vis, (x_l, y_l), 7, (0,255,255), -1)

        x_r = int(round(imagePointsR[0][i][0][0]+Left_rectified.shape[1]))
        y_r = int(round(imagePointsR[0][i][0][1]))
        #print('Right: x = {} pixels, y = {} pixels'.format(x_r, y_r))
        cv.circle(vis, (x_r, y_r), 7, (0,255,255), -1)

        slope = (y_l-y_r)/(x_r-x_l)
        slopes.append(slope)

        cv.line(vis, (x_l,y_l), (x_r,y_r), (0,255,255), 2)


    avg = sum(slopes)/len(slopes)
    cv.putText(vis, 'Average slope '+str(avg),(vis.shape[1]//3, (vis.shape[0]//5)*4), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
    cv.imshow('Rectification check - remapped images', vis)
    
    
# Check how undistorted frame looks like
dstL = cv.undistort(imgL, cameraMatrixL, distortionCoefficientsL, None, newCameraMatrixL)
dstR = cv.undistort(imgR, cameraMatrixR, distortionCoefficientsR, None, newCameraMatrixR)
visDst = np.concatenate((dstL, dstR), axis=1)

grayL = cv.cvtColor(dstL,cv.COLOR_BGR2GRAY)
grayR = cv.cvtColor(dstR,cv.COLOR_BGR2GRAY)

retL, cornersL = cv.findChessboardCorners(grayL, boardSize, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)
retR, cornersR = cv.findChessboardCorners(grayR, boardSize, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)

objp = np.zeros((1, boardSize[0]*boardSize[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:boardSize[0], 0:boardSize[1]].T.reshape(-1, 2)
objectPoints = []
imagePointsL = [] 
imagePointsR = [] 

slopes2 = []
# If found, add object points, image points (after refining them)
if retR is True and retL is True:
    objectPoints.append(objp)
    cv.cornerSubPix(grayR, cornersR,(3,3),(-1,-1),subpix_criteria)
    cv.cornerSubPix(grayL, cornersL,(3,3),(-1,-1),subpix_criteria)
    imagePointsR.append(cornersR)
    imagePointsL.append(cornersL)
    
    # Get points in 4th row (vertical centre) and display them
    for i in range(24,30):
        x_l = int(round(imagePointsL[0][i][0][0]))
        y_l = int(round(imagePointsL[0][i][0][1]))
        #print('Left: x = {} pixels, y = {} pixels'.format(x_l, y_l))
        cv.circle(visDst, (x_l, y_l), 7, (0,255,255), -1)

        x_r = int(round(imagePointsR[0][i][0][0]+Left_rectified.shape[1]))
        y_r = int(round(imagePointsR[0][i][0][1]))
        #print('Right: x = {} pixels, y = {} pixels'.format(x_r, y_r))
        cv.circle(visDst, (x_r, y_r), 7, (0,255,255), -1)

        cv.line(visDst, (x_l,y_l), (x_r,y_r), (0,255,255), 2)

        slope = (y_l-y_r)/(x_r-x_l)
        slopes2.append(slope)
    

    avg2 = sum(slopes2)/len(slopes2)
    cv.putText(visDst, 'Average slope '+str(avg2),(vis.shape[1]//3, (vis.shape[0]//5)*4), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)

cv.imshow('Rectification check - undistorted images', visDst) 
cv.waitKey(0)
                    
sys.exit()

# Find keypoints and their locations in the left UN-rectified image
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(grayL, None)
kp2, des2 = sift.detectAndCompute(grayR, None)

imgSift = cv.drawKeypoints(grayL, kp1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow("SIFT Keypoints", imgSift)

# Find corresponding matches in right un-rectified image and connect the points
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=100)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

matchesMask = [[0, 0] for i in range(len(matches))]
good = []
pts1 = []
pts2 = []

for i, (m, n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        # Keep this keypoint pair
        matchesMask[i] = [1, 0]
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask,
                   flags=cv.DrawMatchesFlags_DEFAULT)

keypoint_matches = cv.drawMatchesKnn(grayL, kp1, grayR, kp2, matches, None, **draw_params)
#cv.imshow("Keypoint matches", keypoint_matches)

def drawlines(img1src, img2src, lines, pts1src, pts2src):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1src.shape
    img1color = cv.cvtColor(img1src, cv.COLOR_GRAY2BGR)
    img2color = cv.cvtColor(img2src, cv.COLOR_GRAY2BGR)
    # Edit: use the same random seed so that two images are comparable!
    np.random.seed(0)
    for r, pt1, pt2 in zip(lines, pts1src, pts2src):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1color = cv.line(img1color, (x0, y0), (x1, y1), color, 1)
        img1color = cv.circle(img1color, tuple(pt1), 5, color, -1)
        img2color = cv.circle(img2color, tuple(pt2), 5, color, -1)
    return img1color, img2color

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, Fmat)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(grayL, grayR, lines1, pts1, pts2)

lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, Fmat)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(grayR, grayL, lines2, pts2, pts1)
cv.imshow('Epilines L', img6)
cv.imshow('Epilines R', img4)
cv.waitKey(0)

sys.exit()



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

## Disparity debugging
win_left_disp = 'Left disparity map'
win_filtered_disp = 'Filtered disparity map'
win_rec_left = 'Camera L rectified'
win_rec_right = 'Camera R rectified'

cv.namedWindow(win_left_disp)
cv.namedWindow(win_filtered_disp)
cv.namedWindow(win_rec_left)
cv.namedWindow(win_rec_right)

# Get value at left centre of disparity map
print('____________Checking left disparity map____________')
left_disp = left_matcher.compute(Left_rectified, Right_rectified)
print('Data type: '+str(left_disp.dtype))
print('Shape: '+str(left_disp.shape))
x_centre = left_disp.shape[1] // 2
y_centre = left_disp.shape[0] // 2
disp = left_disp[y_centre][x_centre].astype(np.float32) / 16
print('Disparity in center of map: {:.2f} pixels'.format(disp))
#cv.imshow(win_left_disp, left_disp)

# Get value at filtered centre of disparity map
print('____________Checking filtered disparity map____________')
right_disp = right_matcher.compute(Right_rectified,Left_rectified)
filtered_disp = wls_filter.filter(left_disp, Left_rectified, disparity_map_right=right_disp)
print('Data type: '+str(filtered_disp.dtype))
print('Shape: '+str(filtered_disp.shape))
x_centre = filtered_disp.shape[1] // 2
y_centre = filtered_disp.shape[0] // 2
disp = filtered_disp[y_centre][x_centre].astype(np.float32) / 16
print('Disparity in center of map: {:.2f} pixels'.format(disp))
#cv.imshow(win_filtered_disp, filtered_disp)

# Get disparity value from left and right image manually
print('____________Computing disparity value manually____________')
font = cv.FONT_HERSHEY_PLAIN
fontScale = 3

def printCoordinatesL(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(Left_rectified, (x, y), 5, (0, 255, 255), -1)   
        print('Coordinates on left screen x={xscreen}, y={yscreen} [pixels]'.format(xscreen=x, yscreen=y))

def printCoordinatesR(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(Right_rectified, (x, y), 5, (0, 255, 255), -1)   
        print('Coordinates on right screen x={xscreen}, y={yscreen} [pixels]'.format(xscreen=x, yscreen=y))

def printCoordinatesdispL(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        Z = left_disp[y_centre][x_centre].astype(np.float32) / 16
        Z = str(Z)+' pixels'
        cv.putText(left_disp, Z, (x, y), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)

def printCoordinatesdispF(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        Z = filtered_disp[y][x].astype(np.float32) / 16
        Z = str(Z)+' pixels'
        cv.putText( filtered_disp, Z, (x, y), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)

cv.setMouseCallback(win_filtered_disp, printCoordinatesdispF)
cv.setMouseCallback(win_left_disp, printCoordinatesdispL)
cv.setMouseCallback(win_rec_left, printCoordinatesL)
cv.setMouseCallback(win_rec_right, printCoordinatesR)

while True:


    key = cv.waitKey(25)

    if key == 27:
        print('Program was terminated by user..')
        break

    cv.imshow(win_left_disp, left_disp)
    cv.imshow(win_filtered_disp, filtered_disp)

    cv.imshow(win_rec_left, Left_rectified)
    cv.imshow(win_rec_right, Right_rectified)  


cv.destroyAllWindows()

# Manual disparity and depth calculation (for center of frame)
# disp = filtered_disp[imgL.shape[0]//2][imgL.shape[1]//2]

# z = ((-Trns[0][0])*cameraMatrixL[0][0])/(disp)

# print('Disparity: {}, depth: {}'.format(disp, z))

# z = int(z)
# z = str(z)+' mm'
# font = cv.FONT_HERSHEY_PLAIN
# fontScale = 3

# x = imgL.shape[1]//2 +2
# y = imgL.shape[0]//2 -2

# cv.putText(imgL, z, (x, y), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)

# img_center = [imgL.shape[0]//2, imgL.shape[1]//2]
# x1 = img_center[1]
# y1 = img_center[0]-75
# x2 = img_center[1]
# y2 = img_center[0]+75

# x3 = img_center[1]-75
# y3 = img_center[0]
# x4 = img_center[1]+75
# y4 = img_center[0]

# yellow = [0, 255, 255]

# cv.line(imgL, (x1,y1), (x2,y2), yellow, thickness=2)
# cv.line(imgL, (x3,y3), (x4,y4), yellow, thickness=2)

# cv.imshow('Camera L', imgL)

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