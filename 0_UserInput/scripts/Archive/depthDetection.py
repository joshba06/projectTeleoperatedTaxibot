import cv2 as cv
import numpy as np
import os
import cameraCalibration as camcal
import sys


#%% Settings

dst_chessboard_corners = 2.4 #cm

boardSize = (6,9)
subpix_criteria = (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)
winSize = (3,3)

numImgsGenerateSingle = 70
numImgsGenerateStereo = 70

home_path = '/Users/niklas/Virtual_Environment/Version_4/Object_Detection'
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
singleLGenImages = True
singleRGenImages = False

stereoTestCombinations = False
singleLTestCombinations = False
singleRTestCombinations = False

singleLCalibration = False
singleRCalibration = False
stereoCalibration = False

newRectificationMapping = False

showAllSteps = False
showCams = False


#%% Main

## Step 0: Initiate camera capture

capL = cv.VideoCapture(0)
capR = cv.VideoCapture(2)


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
        lsBestCombination = np.load(paths['individual']+'/bestCombinationCamL.npy')

        # Compute intrinsics
        cameraMatrixL, distortionCoefficientsL, newCameraMatrixL = camcal.calibrateSingle('L', lsBestCombination, paths['individual'], subpix_criteria, calibration_flags, boardSize, winSize)
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
        lsBestCombination = np.load(paths['individual']+'/bestCombinationCamR.npy')

        # Compute intrinsics
        cameraMatrixR, distortionCoefficientsR, newCameraMatrixR = camcal.calibrateSingle('R', lsBestCombination, paths['individual'], subpix_criteria, calibration_flags, boardSize, winSize)
        print('Camera R intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}'.format(fx=cameraMatrixR[0][0], fy=cameraMatrixR[1][1], cx=cameraMatrixR[0][2], cy=cameraMatrixR[1][2]))
    
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
        lsCombination = np.load(paths['stereo']+'/bestCombinationStereo.npy')

        # Load intrinsics for left and right camera
        cameraMatrixL = np.load(paths['individual']+'/cameraMatrixL.npy')
        newCameraMatrixL = np.load(paths['individual']+'/newCameraMatrixL.npy')
        distortionCoefficientsL = np.load(paths['individual']+'/distortionCoefficientsL.npy')
        cameraMatrixR = np.load(paths['individual']+'/cameraMatrixR.npy')
        newCameraMatrixR = np.load(paths['individual']+'/newCameraMatrixR.npy')
        distortionCoefficientsR = np.load(paths['individual']+'/distortionCoefficientsR.npy')
        print('Read individual camera matrices from file...')

        # Compute stereo intrinsics
        retS, newCameraMatrixL, distortionCoefficientsL, newCameraMatrixR, distortionCoefficientsR, Rot, Trns, Emat, Fmat = camcal.calibrateStereo(lsCombination, paths['stereo'], newCameraMatrixL, distortionCoefficientsL, newCameraMatrixR, distortionCoefficientsR, boardSize)

        """
        -> cameraMatrix, newCameraMatrix and distortionCoefficients remain the same as flag=KeepIntrinsics is set. Hence, they wont be saved to file below!
        """

        # Get baseline after calibration process
        coord_x = Trns[0][0]*dst_chessboard_corners
        coord_y = Trns[1][0]*dst_chessboard_corners
        coord_z = Trns[2][0]*dst_chessboard_corners
        print('Distance between camR and cam L: x={x} (pos to the right), y={y}, z={z} (pos. along optical axis'.format(x=coord_x, y=coord_y, z=coord_z))
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
    lsCombination = list(np.load(paths['stereo']+'/bestCombinationStereo.npy'))
    indices = []
    for imgNum in lsCombination:
        temp = imgNum.split('_')[0]
        indices.append(temp)

    path_img_L = paths['stereo']+'/camL/'+indices[1]+'_L.png'
    path_img_R = paths['stereo']+'/camR/'+indices[1]+'_R.png'
    imgL = cv.imread(path_img_L)
    imgR = cv.imread(path_img_R)
    grayL = cv.cvtColor(imgL,cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR,cv.COLOR_BGR2GRAY)

    rectify_scale= 1
    rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR= cv.stereoRectify(newCameraMatrixL, distortionCoefficientsL, newCameraMatrixR, distortionCoefficientsR, grayL.shape[::-1], Rot, Trns, rectify_scale,(0,0))

    Left_Stereo_Map= cv.initUndistortRectifyMap(newCameraMatrixL, distortionCoefficientsL, rect_l, proj_mat_l, grayL.shape[::-1], cv.CV_16SC2)
    Right_Stereo_Map= cv.initUndistortRectifyMap(newCameraMatrixR, distortionCoefficientsR, rect_r, proj_mat_r, grayR.shape[::-1], cv.CV_16SC2)
    print('Computed stereo maps-.-.')

    # Save maps to file - DOES NOT WORK - REDO THIS WHEN THERE IS TIME
    # np.save(paths['stereo']+'/leftStereoMap', Left_Stereo_Map)
    # np.save(paths['stereo']+'/rightStereoMap', Right_Stereo_Map)
    # print('Saved stereo maps to file...')


    ### Add drawing of epipolar lines!!! for checking purposes

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

sys.exit()


## Step 5: Create disparity and depth map from scenery
print('__________ Starting disparity and depth mapping __________')
print('.')
print('.')

# Creating an object of StereoSGBM algorithm
minDisparity = 0
maxDisparity = 16*9
numDisparities = maxDisparity-minDisparity
blockSize = 3
disp12MaxDiff = 5
uniquenessRatio = 7
speckleWindowSize = 0
speckleRange = 2

stereo = cv.StereoSGBM_create(minDisparity = minDisparity,
        numDisparities = numDisparities,
        blockSize = blockSize,
        disp12MaxDiff = disp12MaxDiff,
        uniquenessRatio = uniquenessRatio,
        speckleWindowSize = speckleWindowSize,
        speckleRange = speckleRange
    )

while True:

    key = cv.waitKey(40)

    if key == 27:
        print('Program was terminated by user..')
        sys.exit()

    isTrueL, frameL = capL.read()
    isTrueR, frameR = capR.read()
    Left_rectified= cv.remap(frameL,Left_Stereo_Map[0],Left_Stereo_Map[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    Right_rectified= cv.remap(frameR,Right_Stereo_Map[0],Right_Stereo_Map[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    grayL  = cv.cvtColor(Left_rectified,cv.COLOR_BGR2GRAY)
    grayR  = cv.cvtColor(Right_rectified,cv.COLOR_BGR2GRAY)

    # Show rectified left frame as basis for distance analysis
    img_center = [Left_rectified.shape[0]//2, Left_rectified.shape[1]//2]
    x1 = img_center[1]
    y1 = img_center[0]-75
    x2 = img_center[1]
    y2 = img_center[0]+75

    x3 = img_center[1]-75
    y3 = img_center[0]
    x4 = img_center[1]+75
    y4 = img_center[0]

    yellow = [0, 255, 255]

    cv.line(Left_rectified, (x1,y1), (x2,y2), yellow, thickness=2)
    cv.line(Left_rectified, (x3,y3), (x4,y4), yellow, thickness=2)

    cv.imshow('Camera L rectified', Left_rectified)

    # Calculating disparity using the StereoSGBM algorithm
    disparity_map = stereo.compute(grayL, grayR).astype(np.float32) / 16
    cv.imshow('Disparity Map Basis', disparity_map)

    dsp_map_normed = (disparity_map - minDisparity) / numDisparities
    cv.imshow('Disparity Map Normed', dsp_map_normed)

    disparity_map_colored = cv.applyColorMap((dsp_map_normed * (256. / maxDisparity)).astype(np.uint8),cv.COLORMAP_HOT)
    cv.imshow('Disparity Map Colored', disparity_map_colored)

    # cv.filterSpeckles(disparity_map, 0, 40, max_disparity)
    # _, disparity_map = cv.threshold(disparity_map, 0, max_disparity * 16, cv.THRESH_TOZERO)
    # disparity_scaled = (disparity_map / 16.).astype(np.uint8)

    # cv.imshow('Disparity Map', (disparity_scaled * (256. / max_disparity)).astype(np.uint8))

    #cv.imshow('Disparity Map', (disparity_map - minDisparity) / numDisparities)


    # ## Step 6: Reprojecting image to 3D
    # points_3D = cv.reprojectImageTo3D(disparity_map, Q)


    # coordinates_center = points_3D[img_center[0], img_center[1]]
    # print('Coordinates in center: {}'.format(coordinates_center))

    # returns a 3 channel matrix. Third channel are the x, y, z coordiantes
    