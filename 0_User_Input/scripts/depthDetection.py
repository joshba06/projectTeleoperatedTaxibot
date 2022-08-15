import cv2 as cv
import numpy as np
import os
import cameraCalibration as camcal
import sys

dst_chessboard_corners = 22 #mm

boardSize = (6,9)
subpix_criteria = (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 30, 0.01)
winSize = (11,11)

numImgsGenerateSingle = 30
numImgsGenerateStereo = 20

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

calibration_flags = cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv.fisheye.CALIB_CHECK_COND+cv.fisheye.CALIB_FIX_SKEW

stereoGenImages = False
singleLGenImages = False
singleRGenImages = False

stereoTestCombinations = False
singleLTestCombinations = False
singleRTestCombinations = False

singleLCalibration = False
singleRCalibration = False
stereoCalibration = False

newRectificationMapping = True

showAllSteps = False
showCams = False


#%% Main

## Step 0: Initiate camera capture

capL = cv.VideoCapture(2)
capR = cv.VideoCapture(0)


if showCams is True:
    #Check whether cameras are connected correctly
        # Error guide:
        # Usually the cameras are at index 0 and 2. If it doesnt work check if the USBC USB3 adapter is plugged in the correct way

    while True:
        key = cv.waitKey(25)

        if key == 27:
            print('Program was terminated by user..')
            sys.exit()
        elif key == ord('o'):
            print('User confirmed correct alignment of cameras...')
            break

        # Display livestream of camera
        isTrueL, frameL = capL.read()
        isTrueR, frameR = capR.read()
        cv.imshow('Camera L', frameL)
        cv.imshow('Camera R', frameR)


## Step 1: Calibration
print('__________ Step 1: Calibration __________')
print('.')
print('.')

# Option A: Create new images
if (singleLGenImages is True) and (singleLTestCombinations is False): # Works
    
    print('Starting image generation for camera L...')
    camcal.generateImgsSingle(capL, 'L', paths['indCamL'], numImgsGenerateSingle, subpix_criteria, boardSize)
    print('Images successfully generated and saved to file...')

elif (singleRGenImages is True) and (singleRTestCombinations is False): # Works
    
    print('Starting image generation for camera R...')
    camcal.generateImgsSingle(capR, 'R', paths['indCamR'], numImgsGenerateSingle, subpix_criteria, boardSize)
    print('Images successfully generated and saved to file...')

elif (stereoGenImages is True) and (stereoTestCombinations is False): # Works
    
    print('Starting image generation for stereo setup...')
    camcal.generateImagesStereo(capL, capR, paths['stCamL'], paths['stCamR'], numImgsGenerateStereo, subpix_criteria, boardSize)
    print('Images successfully generated and saved to file...')

# Option B: Test existing images for best combination
elif (singleLGenImages is False) and (singleLTestCombinations is True): # Works
    
    print('Computing best image combination for camera L')
    # Load all image paths and their image points
    dict_pathsPoints = camcal.loadImgsandImgPoints('Single', boardSize, winSize, subpix_criteria, paths['indCamL'], None)

    # Find best combination for all images
    lsCombination = camcal.testAllImgCombinations('Single', dict_pathsPoints, 'L', paths['individual'], boardSize)
    print('Best image combination for camera L '+str(lsCombination))

elif (singleRGenImages is False) and (singleRTestCombinations is True): # Works
    
    print('Computing best image combination for camera R')
    # Load all image paths and their image points
    dict_pathsPoints = camcal.loadImgsandImgPoints('Single', boardSize, winSize, subpix_criteria, None, paths['indCamR'])

    # Find best combination for all images
    lsCombination = camcal.testAllImgCombinations('Single', dict_pathsPoints, 'R', paths['individual'], boardSize)
    print('Best image combination for camera R '+str(lsCombination))

elif (stereoGenImages is False) and (stereoTestCombinations is True): # Works
    
    print('Computing best image combination for stereo setup')
    # Load all image paths and their image points
    dict_pathsPoints = camcal.loadImgsandImgPoints('Stereo', boardSize, winSize, subpix_criteria, paths['indCamL'], paths['indCamR']) 

    # Find best combination for all images
    lsCombination = camcal.testAllImgCombinations('Stereo', dict_pathsPoints, None, paths['stereo'], boardSize)
    print('Best image combination for stereo setup '+str(lsCombination))

# Option C: Perform Calibration
else:
    if singleLCalibration == True: # Works

        print('__________ Starting cameraL calibration __________')
        print('.')
        print('.')

        # Load list with best combination from file
        # lsBestCombination = np.load(paths['individual']+'/bestCombinationCamL.npy')

        # Compute intrinsics
        # cameraMatrixL, distortionCoefficientsL, newCameraMatrixL = camcal.calibrateSingle('L', lsBestCombination, paths['individual'], subpix_criteria, calibration_flags, boardSize, winSize)
        
        # Notation: (k1, k2, p1, p2, k3), where k are radial distortion coefficients and p are tangential distortion coefficients
        distortionCoefficientsL = np.array([0.1113, -0.2276, 0.0014, 8.0021e-04, 0.0844])

        # Camera matrix from MATLAB Camera Calibrator
        cameraMatrixL = np.array([[1384.9, 0, 933.0051], [-0.4568, 1383.7, 531.6792], [0, 0, 1]])
        newCameraMatrixL = cameraMatrixL

        #Save matrices to folder
        os.chdir(paths['individual'])
        np.save('cameraMatrixL', cameraMatrixL)
        np.save('distortionCoefficientsL', distortionCoefficientsL)
        np.save('newcameraMatrixL', newCameraMatrixL)
        print('CameraMatrix, distortionCoefficients and newCameraMatrix computed and saved to file...')
        
        print('Camera L intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}'.format(fx=cameraMatrixL[0][0], fy=cameraMatrixL[1][1], cx=cameraMatrixL[0][2], cy=cameraMatrixL[1][2]))
        
        # Optional: Show result of calibration for cam
        if showAllSteps == True:
            
            #Undistort livestream to check whether calibration was successfull (this function is heavy on resources and shall not be used for constant undistortion)
            while True:
                key = cv.waitKey(25)
                if key == 27:
                    print('Program was terminated by user.')
                    break

                isTrue, frame = capL.read()

                dst = cv.undistort(frame, cameraMatrixL, distortionCoefficientsL, None, newCameraMatrixL)

                cv.imshow('Camera L - Undistored', dst)

        print('__________ Finished cameraL calibration __________')
        print('.')
        print('.')
    
    elif singleRCalibration == True: # Works
        print('__________ Starting cameraR calibration __________')
        print('.')
        print('.')

        # Load list with best combination from file
        # lsBestCombination = np.load(paths['individual']+'/bestCombinationCamR.npy')
        

        # Compute intrinsics
        # cameraMatrixR, distortionCoefficientsR, newCameraMatrixR = camcal.calibrateSingle('R', lsBestCombination, paths['individual'], subpix_criteria, calibration_flags, boardSize, winSize)
        # Notation: (k1, k2, p1, p2, k3), where k are radial distortion coefficients and p are tangential distortion coefficients
        distortionCoefficientsR = np.array([0.0362, -0.1640, -2.2219e-04, 3.4858e-04, 0.1149])

        # Camera matrix from MATLAB Camera Calibrator [pixels]
        cameraMatrixR = np.array([  [1417.2, 0, 972.8353], 
                                    [0.0543, 1418.1, 542.8851], 
                                    [0, 0, 1]])
        newCameraMatrixR = cameraMatrixR

        #Save matrices to folder
        os.chdir(paths['individual'])
        np.save('cameraMatrixR', cameraMatrixR)
        np.save('distortionCoefficientsR', distortionCoefficientsR)
        np.save('newcameraMatrixR', newCameraMatrixR)
        print('CameraMatrix, distortionCoefficients and newCameraMatrix computed and saved to file...')
        
        print('Camera R intrinsics: fx={fx}px, fy={fy}px, cx={cx}px, cy={cy}px'.format(fx=cameraMatrixR[0][0], fy=cameraMatrixR[1][1], cx=cameraMatrixR[0][2], cy=cameraMatrixR[1][2]))
    
        # Optional: Show result of calibration for cam
        if showAllSteps == True:
            
            #Undistort livestream to check whether calibration was successfull (this function is heavy on resources and shall not be used for constant undistortion)
            while True:
                key = cv.waitKey(25)
                if key == 27:
                    print('Program was terminated by user.')
                    break

                isTrue, frame = capR.read()

                dst = cv.undistort(frame, cameraMatrixR, distortionCoefficientsR, None, newCameraMatrixR)

                cv.imshow('Camera R - Undistored', dst)
        
        print('__________ Finished cameraR calibration __________')
        print('.')
        print('.')


    elif stereoCalibration == True: # Works
        
        print('__________ Starting stereo camera calibration __________')
        print('.')
        print('.')

        # Load list with best combination from file
        # lsCombination = np.load(paths['stereo']+'/bestCombinationStereo.npy')

        # Load intrinsics for left and right camera
        cameraMatrixL = np.load(paths['individual']+'/cameraMatrixL.npy')
        newCameraMatrixL = np.load(paths['individual']+'/newCameraMatrixL.npy')
        distortionCoefficientsL = np.load(paths['individual']+'/distortionCoefficientsL.npy')
        cameraMatrixR = np.load(paths['individual']+'/cameraMatrixR.npy')
        newCameraMatrixR = np.load(paths['individual']+'/newCameraMatrixR.npy')
        distortionCoefficientsR = np.load(paths['individual']+'/distortionCoefficientsR.npy')
        print('Read individual camera matrices from file...')

        # Compute stereo intrinsics
        # retS, newCameraMatrixL, distortionCoefficientsL, newCameraMatrixR, distortionCoefficientsR, Rot, Trns, Emat, Fmat = camcal.calibrateStereo(lsCombination, paths['stereo'], newCameraMatrixL, distortionCoefficientsL, newCameraMatrixR, distortionCoefficientsR, boardSize)

        
        Rot = np.array([ [0.9999, 0.0110, 0.0072],
                [-0.0112, 0.9998, 0.0181],
                [-0.0070, -0.0182, 0.9998]
        ])
        #Rot = np.divide(Rot, 22)
        
        Trns = np.array([[-96.5287], [-1.0656], [-1.0285]])
        #Trns = np.divide(Trns, 22)

        # Essential matrix in physical coordinates i.e. mm. Relates point P as seen on the left imager to how it is seen by the right imager
        Emat = np.array([   [0.0037, 1.0090, -1.0841],
                            [-0.3329, 1.7597, 96.5176],
                            [-1.8098e-04, -96.5188, 1.7483]
        ])
        #Emat = np.divide(Emat, 22)

        # Fundamental matrix in camera coordinates i.e. pixels
        Fmat = np.array([   [1.8736e-09, 5.1454e-07, -0.0010],
                            [-1.6953e-07, 8.9672e-07, 0.0677],
                            [9.0080e-05, -0.0707, 3.0696]
        ])
        #Fmat = np.divide(Fmat, 22)


        """
        -> cameraMatrix, newCameraMatrix and distortionCoefficients remain the same as flag=KeepIntrinsics is set. Hence, they wont be saved to file below!
        """

        # Get baseline after calibration process
        coord_x = Trns[0][0]#*dst_chessboard_corners
        coord_y = Trns[1][0]#*dst_chessboard_corners
        coord_z = Trns[2][0]#*dst_chessboard_corners
        print('Distance between camR and cam L: x={:.2f}mm (pos to the right), y={:.2f} mm, z={:.2f}mm (pos. along optical axis'.format(coord_x, coord_y, coord_z))
        print('Reminder: z is positive towards the front of the cameras. X is positive to the right and Y is positive downwards. Hence, x must be negative in the above result')

        # Save stereo intrinsics
        np.save(paths['stereo']+'/rotationVector', Rot)
        np.save(paths['stereo']+'/translationVector', Trns)
        np.save(paths['stereo']+'/essentialMatrixE', Emat)
        np.save(paths['stereo']+'/fundamentalMatrixF', Fmat)
        print('Saved stereo matrices to file...')

        print('.')
        print('.')
        print('__________ Finished stereo camera calibration __________')


## Step 2: Rectification and mapping

print('__________ Step 2: Stereo rectification and mapping __________')
print('.')
print('.')

if newRectificationMapping == True:

    print('Computing new rectification map...')

    # Read intrinsics matrices and stereo matrices from file
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
    print('Read camera matrices from file...')

    # Read two images for stereo Rectification (of the ones that did not cause calibration to fail)
    # lsCombination = list(np.load(paths['stereo']+'/bestCombinationStereo.npy'))
    # indices = []
    # for imgNum in lsCombination:
    #     temp = imgNum.split('_')[0]
    #     indices.append(temp)

    path_img_L = paths['stereo']+'/camL/4_L.png'
    path_img_R = paths['stereo']+'/camR/4_R.png'
    imgL = cv.imread(path_img_L)
    imgR = cv.imread(path_img_R)
    grayL = cv.cvtColor(imgL,cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR,cv.COLOR_BGR2GRAY)

    rectify_scale= 1
    rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR= cv.stereoRectify(newCameraMatrixL, distortionCoefficientsL, newCameraMatrixR, distortionCoefficientsR, grayL.shape[::-1], Rot, Trns, rectify_scale,(0,0))

    Left_Stereo_Map= cv.initUndistortRectifyMap(newCameraMatrixL, distortionCoefficientsL, rect_l, proj_mat_l, grayL.shape[::-1], cv.CV_16SC2)
    Right_Stereo_Map= cv.initUndistortRectifyMap(newCameraMatrixR, distortionCoefficientsR, rect_r, proj_mat_r, grayR.shape[::-1], cv.CV_16SC2)
    print('Computed stereo maps...')


    # Optional: Show result of rectification
    if showAllSteps is True:

        cv.imshow("Left image before rectification", imgL)
        cv.imshow("Right image before rectification", imgR)

        Left_rectified= cv.remap(imgL,Left_Stereo_Map[0],Left_Stereo_Map[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
        Right_rectified= cv.remap(imgR,Right_Stereo_Map[0],Right_Stereo_Map[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)

        cv.imshow("Left image after rectification", Left_rectified)
        cv.imshow("Right image after rectification", Right_rectified)

        cv.waitKey(0)
# else:
#     # Read maps from file
#     Left_Stereo_Map = np.load(paths['stereo']+'/leftStereoMap.npy')
#     Right_Stereo_Map = np.load(paths['stereo']+'/rightStereoMap.npy')
#     print('Read stereo maps from file...')   


print('.')
print('.')
print('__________ Finished stereo rectification and mapping __________')



## Step 5: Create disparity and depth map from scenery
print('__________ Starting disparity and depth mapping __________')
print('.')
print('.')

## Take one image of the scene with both cameras
path_depth_testing = '/Users/niklas/Virtual_Environment/Version_5/projectAutonomous/Depth_measurement'
takeTestPhoto = False
if takeTestPhoto is True:
    while True:

        key = cv.waitKey(5)

        if key == 27:
            print('Program was terminated by user..')
            sys.exit()

        isTrueL, frameL = capL.read()
        isTrueR, frameR = capR.read()
        cv.imshow('Cam R ', frameR)
        # temp2 = frameL
        # temp = frameL

        # x1 = temp.shape[1]//2
        # y1 = 0
        # x2 = temp.shape[1]//2
        # y2 = temp.shape[0]

        # x3 = 0
        # y3 = temp.shape[0]//2
        # x4 = temp.shape[1]
        # y4 = temp.shape[0]//2

        # yellow = [0, 255, 255]

        # cv.line(temp, (x1,y1), (x2,y2), yellow, thickness=2)
        # cv.line(temp, (x3,y3), (x4,y4), yellow, thickness=2)
        
        cv.imshow('Cam L ', frameL)
        

        if key == ord(' '):
            imgPathL = path_depth_testing+'/camL5.png'
            imgPathR = path_depth_testing+'/camR5.png'

            cv.imwrite(imgPathL,frameL)
            cv.imwrite(imgPathR,frameR)
            sys.exit()


## Rectify images
# imgL = cv.imread(path_depth_testing+'/camL0.png')
# imgR = cv.imread(path_depth_testing+'/camR0.png')

# grayL = cv.cvtColor(imgL,cv.COLOR_BGR2GRAY)
# grayR = cv.cvtColor(imgR,cv.COLOR_BGR2GRAY)


# Left_rectified= cv.remap(grayL,Left_Stereo_Map[0],Left_Stereo_Map[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
# Right_rectified= cv.remap(grayR,Right_Stereo_Map[0],Right_Stereo_Map[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)


# cv.imshow('Camera L rectified', Left_rectified)
# cv.imshow('Camera R rectified', Right_rectified)
# cv.waitKey(0)

# New code from github: https://github.com/opencv/opencv/blob/master/samples/python/stereo_match.py
# ply_header = '''ply
# format ascii 1.0
# element vertex %(vert_num)d
# property float x
# property float y
# property float z
# property uchar red
# property uchar green
# property uchar blue
# end_header
# '''
# def write_ply(fn, verts, colors):
#     verts = verts.reshape(-1, 3)
#     colors = colors.reshape(-1, 3)
#     verts = np.hstack([verts, colors])
#     with open(fn, 'wb') as f:
#         f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
#         np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
# # End new code

## DISPARITY TESTING______________________________________________

minDisparity = 144
maxDisparity = 272
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
# left_matcher = cv.StereoBM_create(numDisparities = numDisparities, blockSize=blockSize)

sigma = 1.5
lmbda = 8000.0

right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
wls_filter = cv.ximgproc.createDisparityWLSFilter(left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

##________________________________________________________________

# ## Show disparity map
# minDisparity = 64
# maxDisparity = 128
# numDisparities = maxDisparity-minDisparity
# blockSize = 3
# P1 = 8*3*blockSize**2 
# P2 = 32*3*blockSize**2 
# disp12MaxDiff = 50
# uniquenessRatio = 5
# speckleWindowSize = 16*3 #256
# speckleRange = 64

# left_matcher = cv.StereoSGBM_create(minDisparity = minDisparity,
#         numDisparities = numDisparities,
#         blockSize = blockSize,
#         P1 = P1,
#         P2 = P2,
#         #disp12MaxDiff = disp12MaxDiff,
#         uniquenessRatio = uniquenessRatio,
#         # speckleWindowSize = speckleWindowSize,
#         # speckleRange = speckleRange,
# )


def printCoordinates(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(filtered_disp, (x, y), 50, (0, 255, 255), -1)   
        print('Coordinates on screen x={xscreen}, y={yscreen} [pixels]'.format(xscreen=x, yscreen=y))

        xglobal = points_3D[y][x][0]
        yglobal = points_3D[y][x][1]
        zglobal = points_3D[y][x][2]

        print('Coordinates as seen from left camera x={:.2f}mm, y={:.2f}mm, z={:.2f}mm'.format(xglobal, yglobal , zglobal))

        print('Disparity: {}'.format(filtered_disp[y][x]))
        
        

# cv.namedWindow('Filtered Disparity Display')
# cv.setMouseCallback('Filtered Disparity Display', printCoordinates)

# sigma = 1.5
# lmbda = 8000.0

# right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
# wls_filter = cv.ximgproc.createDisparityWLSFilter(left_matcher)
# wls_filter.setLambda(lmbda)
# wls_filter.setSigmaColor(sigma)




# print('Distance between camR and cam L: x={:.2f}mm (pos to the right), y={:.2f} mm, z={:.2f}mm (pos. along optical axis'.format(Trns[0][0], Trns[1][0], Trns[2][0]))

#colors = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)
#mask = temp > temp.min()
#out_points = points_3D[mask]
#out_colors = colors[mask]
#out_fn = 'out.ply'
#write_ply(out_fn, out_points, out_colors)
#print('%s saved' % out_fn)



while True:

    isTrueL, frameL = capL.read()
    isTrueR, frameR = capR.read()

    key = cv.waitKey(50)

    if key == 27:
        print('Program was terminated by user..')
        sys.exit()

    grayL = cv.cvtColor(frameL,cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(frameR,cv.COLOR_BGR2GRAY)
    Left_rectified= cv.remap(grayL,Left_Stereo_Map[0],Left_Stereo_Map[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    Right_rectified= cv.remap(grayR,Right_Stereo_Map[0],Right_Stereo_Map[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)

    # Left frame centre for depth measurement
    img_center = [frameL.shape[0]//2, frameL.shape[1]//2]
    x1 = img_center[1]
    y1 = img_center[0]-75
    x2 = img_center[1]
    y2 = img_center[0]+75

    x3 = img_center[1]-75
    y3 = img_center[0]
    x4 = img_center[1]+75
    y4 = img_center[0]

    yellow = [0, 255, 255]

    cv.line(frameL, (x1,y1), (x2,y2), yellow, thickness=2)
    cv.line(frameL, (x3,y3), (x4,y4), yellow, thickness=2)

    # Disparity just for visual check
    left_disp = left_matcher.compute(Left_rectified, Right_rectified)
    right_disp = right_matcher.compute(Right_rectified,Left_rectified)
    filtered_disp = wls_filter.filter(left_disp, Left_rectified, disparity_map_right=right_disp)
    cv.imshow('Disparity for visual check', filtered_disp) 

    # Manual disparity and depth calculation (for center of frame)
    disp = filtered_disp[frameL.shape[0]//2][frameL.shape[1]//2]

    z = ((-Trns[0][0])*cameraMatrixL[0][0])/(disp)

    print('Disparity: {}, depth: {}'.format(disp, z))

    z = int(z)
    z = str(z)+' mm'
    font = cv.FONT_HERSHEY_PLAIN
    fontScale = 3

    x = frameL.shape[1]//2 +2
    y = frameL.shape[0]//2 -2

    cv.putText(frameL, z, (x, y), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
    cv.imshow('Camera L', frameL)

    # Disparity for 3Dreprojection -> Why dont simply convert from int16 to unit8?
    # temp = filtered_disp.astype(np.float32) / 16
    # disp8forPCL = np.uint8(temp)
    # points_3D = cv.reprojectImageTo3D(disp8forPCL, Q)

    # Disparity for colored map
    left_disp_col = left_disp.astype(np.float32) / 16
    right_disp_col = right_disp.astype(np.float32) / 16
    filtered_disp_col = wls_filter.filter(left_disp_col, Left_rectified, disparity_map_right=right_disp_col)
    #cv.imshow('Disparity for colormap', filtered_disp_col) 

    # Use 8bit unsigned integer disparity display (0...255) for color mapping
    disp8forColor = np.uint8(filtered_disp_col)
    colored = cv.applyColorMap(disp8forColor, cv.COLORMAP_JET)
    cv.imshow('Disparity Coloured', colored) 

    # xglobal = points_3D[img_center[0]][img_center[1]][0]
    # yglobal = points_3D[img_center[0]][img_center[1]][1]
    # zglobal = points_3D[img_center[0]][img_center[1]][2]
    #print('Coordinates as seen from left camera x={:.2f}mm, y={:.2f}mm, z={:.2f}mm'.format(xglobal, yglobal , zglobal))


    #filtered_disp_test = cv.normalize(src=filtered_disp_test, dst=filtered_disp_test, beta=0, alpha=255, norm_type=cv.NORM_MINMAX);
    #filtered_disp_test = np.uint8(filtered_disp_test)
    

    # Reproject 3D points using NOT NORMALIZED version of disparity map that is scaled to float and divided by 16
    # temp = filtered_disp.astype(np.float32) / 16
    # points_3D = cv.reprojectImageTo3D(temp, Q)

    # Normalize the values to a range from 0..255 for a grayscale image
    # #cv.normalize(filtered_disp, filtered_disp, alpha=255, beta=0, norm_type=cv.NORM_MINMAX)
    # disp_8 = (filtered_disp/256).astype(np.uint8)
    # colored = cv.applyColorMap(disp_8, cv.COLORMAP_JET)
    
    # filtered_disp = np.uint8(filtered_disp)
    # colored = cv.applyColorMap(filtered_disp, cv.COLORMAP_JET)


    

    
    if cv.waitKey(10) & 0xFF == 27:
        break
        
cv.destroyAllWindows()


#cv.imshow('Disparity Display', disparity_map)   


sys.exit()