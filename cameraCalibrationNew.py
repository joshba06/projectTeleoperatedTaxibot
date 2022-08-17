import cv2 as cv
import numpy as np
import time
import os
import csv
import sys
from pprint import pprint
from datetime import datetime

numImgsGenerateSingle = 30
numImgsGenerateStereo = 20

calibration_criteria = (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 100, 1e-06)
winSize = (11,11)
squareSize = 2.2 #cm
boardSize = (6,9)

#%% Sub functions

def generateImgsSingle(capture, camPosition, pathImgs, numImgsGenerateSingle, calibration_criteria, boardSize):
        
    print('Starting generation of image and objectpoints for camera {}...'.format(camPosition))

    winSize = (5, 5)

    ## Elementary settings
    value_timer = int(1)

    # (Re)set timer
    timer = value_timer

    # (Re)set number of valid calibration images
    validImg = 0

    # Set flag to run program until calibration and correction was successfully executed
    flag = 0

    while (True and flag==0):

        key = cv.waitKey(25)

        if key == 27:
            print('Program was terminated by user.')
            break
        
        # Display livestream of camera
        isTrue, frame = capture.read()
        font = cv.FONT_HERSHEY_PLAIN
        fontScale = 1
        frame_height, frame_width = frame.shape[:2]


        # Add information about keys and valid images on livestream feed
        cv.putText(frame, 'Press "ESC" to stop the program',(10, frame_height-60), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
        cv.putText(frame, 'Press SPACE to start the photo countdown',(10, frame_height-20), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
        cv.putText(frame, 'Number of valid photos for calibration: '+str(validImg)+'/'+str(numImgsGenerateSingle),(frame_width//3, 40), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
        cv.imshow('Calibration Camera', frame)
        
        # As long as the required number of images has not been reached, take new photos in which a pattern can be identified
        while (validImg < numImgsGenerateSingle):
            
            isTrue, frame = capture.read()
            
            # Add information about keys and valid images on livestream feed
            cv.putText(frame, 'Press "ESC" to stop the program',(10, frame_height-60), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
            cv.putText(frame, 'Press SPACE to start the photo countdown',(10, frame_height-20), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
            cv.putText(frame, 'Number of valid photos for calibration: '+str(validImg)+'/'+str(numImgsGenerateSingle),(frame_width//3, 40), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
            cv.imshow('Calibration Camera', frame)
            
            key = cv.waitKey(125)

            # Start countdown, when SPACE is pressed
            if key == ord(' '):
                
                # Record time when timer key was pressed
                timer_start = time.time()

                # Wait for timer to finish
                while timer >= 0:
                    
                    isTrue, frame = capture.read()

                    cv.putText(frame, str(timer),(10, 50), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
                    cv.imshow('Calibration Camera', frame)
                    cv.waitKey(125)

                    # Display seconds left on timer as soon as 1s elapses
                    current_time = time.time()
                    elapsed_time = current_time - timer_start

                    if elapsed_time >= 1:
                        timer = timer - 1
                        timer_start = current_time
                    else: 
                        isTrue, frame = capture.read()
                        cv.imshow('Calibration Camera', frame)
                
                # Save image to folder
                imgPath = pathImgs+'/'+str(validImg)+'_'+str(camPosition)+'.png'

                # Check if path exists, if it does, increase image number
                exists = 1
                tempNum = validImg
                while exists == 1:
                    if os.path.exists(imgPath):
                        tempNum +=1
                        imgPath = pathImgs+'/'+str(tempNum)+'_'+str(camPosition)+'.png'
                    else: 
                        exists = 2

                cv.imwrite(imgPath,frame)
                
                # Check whether chessboard was identified

                # Try to locate approximate corners on the chessboard image
                img = cv.imread(imgPath)
                gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

                found_pattern, corners_aprx = cv.findChessboardCorners(gray, boardSize, None)


                # If corner-pattern was found, locate exact corners of 2D image and store information
                if (found_pattern == True):
                    
                    corners_exct = cv.cornerSubPix(gray, corners_aprx, winSize, (-1,-1), calibration_criteria)
                    
                    # Draw and display the corners
                    cv.drawChessboardCorners(img, boardSize, corners_exct, found_pattern)
                    cv.putText(img, 'Valid pattern found in image...',(frame_width//3, 80), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
                    cv.imshow('Calibration Camera', img)
                    cv.waitKey(2000)
                    
                    # Add information to main screen
                    validImg += 1
                    
                
                # If corner-pattern was not found, delete image
                elif (found_pattern == False):
                    
                    cv.putText(frame, 'No pattern found. Please retake photo...',(frame_width//3, 80), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
                    cv.imshow('Calibration Camera', frame)


                    cv.waitKey(2000)

                    # Delete image that was taken and for which pattern was not found
                    try: 
                        os.remove(imgPath)
                    except:
                        print('Could not delete image.')
                        pass
            
                # Reset timer
                timer = value_timer

            # Exit program, when letter ESC is pressed
            elif key == 27:
                print('Program was terminated by user.')
                flag = 1
                break
        
        # Exit livestream when required number of images exists
        flag = 1
        cv.destroyAllWindows()

def generateImagesStereo(capL, capR, path_cam_L, path_cam_R, numImgsGenerateStereo, calibration_criteria, boardSize):
    
    print('Starting generation of images...')

    winSize = (11,11)

    ## Elementary settings
    value_timer = int(1)

    # (Re)set timer
    timer = value_timer

    # (Re)set number of valid calibration images
    validImg = 0

    # Set flag to run program until calibration and correction was successfully executed
    flag = 0

    while (True and flag==0):

        key = cv.waitKey(125)

        if key == 27:
            print('Program was terminated by user.')
            break
        
        # Display livestream of camera
        isTrueL, frameL = capL.read()
        isTrueR, frameR = capR.read()
        font = cv.FONT_HERSHEY_PLAIN
        fontScale = 1
        frame_heightL, frame_widthL = frameL.shape[:2]
        frame_heightR, frame_widthR = frameR.shape[:2]

        #Syntax: cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])

        # Add information about keys and valid images on livestream feed
        cv.putText(frameL, 'Press "ESC" to stop the program',(10, frame_heightL-60), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
        cv.putText(frameL, 'Press SPACE to start the photo countdown',(10, frame_heightL-20), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
        cv.putText(frameL, 'Number of valid photos for calibration: '+str(validImg)+'/'+str(numImgsGenerateStereo),(frame_widthL//3, 40), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
        cv.imshow('Calibration Camera L', frameL)

        cv.putText(frameR, 'Press "ESC" to stop the program',(10, frame_heightR-60), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
        cv.putText(frameR, 'Press SPACE to start the photo countdown',(10, frame_heightR-20), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
        cv.putText(frameR, 'Number of valid photos for calibration: '+str(validImg)+'/'+str(numImgsGenerateStereo),(frame_widthR//3, 40), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
        cv.imshow('Calibration Camera R', frameR)
        
        # As long as the required number of images has not been reached, take new photos in which a pattern can be identified
        while (validImg < numImgsGenerateStereo):
            
            isTrueL, frameL = capL.read()
            isTrueR, frameR = capR.read()
            
            # Add information about keys and valid images on livestream feed
            cv.putText(frameL, 'Press "ESC" to stop the program',(10, frame_heightL-60), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
            cv.putText(frameL, 'Press SPACE to start the photo countdown',(10, frame_heightL-20), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
            cv.putText(frameL, 'Number of valid photos for calibration: '+str(validImg)+'/'+str(numImgsGenerateStereo),(frame_widthL//3, 40), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
            cv.imshow('Calibration Camera L', frameL)
            
            cv.putText(frameR, 'Press "ESC" to stop the program',(10, frame_heightR-60), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
            cv.putText(frameR, 'Press SPACE to start the photo countdown',(10, frame_heightR-20), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
            cv.putText(frameR, 'Number of valid photos for calibration: '+str(validImg)+'/'+str(numImgsGenerateStereo),(frame_widthR//3, 40), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
            cv.imshow('Calibration Camera R', frameR)


            key = cv.waitKey(125)

            # Start countdown, when SPACE is pressed
            if key == ord(' '):
                
                # Record time when timer key was pressed
                timer_start = time.time()

                # Wait for timer to finish
                while timer >= 0:
                    isTrueL, frameL = capL.read()
                    isTrueR, frameR = capR.read()

                    cv.putText(frameL, str(timer),(10, 50), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
                    cv.imshow('Calibration Camera L', frameL)

                    cv.putText(frameR, str(timer),(10, 50), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
                    cv.imshow('Calibration Camera R', frameR)
                    cv.waitKey(125)

                    # Display seconds left on timer as soon as 1s elapses
                    current_time = time.time()
                    elapsed_time = current_time - timer_start

                    if elapsed_time >= 1:
                        timer = timer - 1
                        timer_start = current_time
                    else: 
                        isTrueL, frameL = capL.read()
                        isTrueR, frameR = capR.read()
                        cv.imshow('Calibration Camera L', frameL)
                        cv.imshow('Calibration Camera R', frameR)
                
                # Save image to folder
                imgPathL = path_cam_L+'/'+str(validImg)+'_L.png'
                cv.imwrite(imgPathL,frameL)
                imgPathR = path_cam_R+'/'+str(validImg)+'_R.png'
                cv.imwrite(imgPathR,frameR)
                
                # Check whether chessboard was identified

                # Try to locate approximate corners on the chessboard image
                imgL = cv.imread(imgPathL)
                grayL = cv.cvtColor(imgL,cv.COLOR_BGR2GRAY)
                imgR = cv.imread(imgPathR)
                grayR = cv.cvtColor(imgR,cv.COLOR_BGR2GRAY)

                found_patternL, corners_aprxL = cv.findChessboardCorners(grayL, boardSize, None)
                found_patternR, corners_aprxR = cv.findChessboardCorners(grayR, boardSize, None)


                # If corner-pattern was found, locate exact corners of 2D image and store information
                if (found_patternL == True) and (found_patternR == True):
                    
                    corners_exctL = cv.cornerSubPix(grayL, corners_aprxL, winSize, (-1,-1), calibration_criteria)
                    corners_exctR = cv.cornerSubPix(grayR, corners_aprxR, winSize, (-1,-1), calibration_criteria)
                    
                    # Draw and display the corners
                    cv.drawChessboardCorners(imgL, boardSize, corners_exctL, found_patternL)
                    cv.drawChessboardCorners(imgR, boardSize, corners_exctR, found_patternR)
                    cv.putText(imgL, 'Valid pattern found in image...',(frame_widthL//3, 80), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
                    cv.putText(imgR, 'Valid pattern found in image...',(frame_widthR//3, 80), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
                    cv.imshow('Calibration Camera L', imgL)
                    cv.imshow('Calibration Camera R', imgR)
                    cv.waitKey(2000)
                    
                    # Add information to main screen
                    validImg += 1
                    
                
                # If corner-pattern was not found, delete image
                elif (found_patternL == False) or (found_patternR == False):
                    
                    cv.putText(frameL, 'No pattern found. Please retake photo...',(frame_widthL//3, 80), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
                    cv.imshow('Calibration Camera L', frameL)
                    cv.putText(frameR, 'No pattern found. Please retake photo...',(frame_widthR//3, 80), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
                    cv.imshow('Calibration Camera R', frameR)

                    cv.waitKey(2000)

                    # Delete image that was taken and for which pattern was not found
                    try: 
                        os.remove(imgPathL)
                        os.remove(imgPathR)
                    except:
                        print('Could not delete image.')
                        pass
            
                # Reset timer
                timer = value_timer

            # Exit program, when letter ESC is pressed
            elif key == 27:
                print('Program was terminated by user.')
                flag = 1
                break
        
        # Exit livestream when required number of images exists
        flag = 1
        cv.destroyAllWindows()
    
    print('Finished taking all required images for camera calibration...')


def checkArray(array_indices, variable, lsHistory):
    """Determine whether image to be added is already or has already been checked in this combination.
    Return: testedBefore is True -> Combination was already tested

    Keyword arguments:
    array_indices -- Array containing indices to images in dict to be checked.
    variable -- Index of new image to be added to combination.
    lsHistory -- List containing all combinations that have been checked so far.
    """    
  
    if variable in array_indices:
        testedBefore = True
    else:
        array_indices.append(variable)
        temp = array_indices.copy()
        temp.sort()
        if temp in lsHistory:
            array_indices.remove(variable)
            testedBefore = True
        else:
            lsHistory.append(temp.copy())
            testedBefore = False

    return testedBefore, array_indices, lsHistory


# Given a list of image points, calculate reprojection error and roi
def getROIandError(listPoints, gray, boardSize):

    calibration_flags = cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv.fisheye.CALIB_CHECK_COND+cv.fisheye.CALIB_FIX_SKEW
    objp = np.zeros((1, boardSize[0]*boardSize[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:boardSize[0], 0:boardSize[1]].T.reshape(-1, 2)
    objectPoints = []
    imagePoints = [] 

    # Add imagePoints from dictionary to local array
    for point in listPoints:
        objectPoints.append(objp)
        imagePoints.append(point)

    # Prepare matrices to be filled by calibration function
    N_OK = len(objectPoints)
    mat_intrinsics_in = np.zeros((3, 3))
    vec_distortion_in = np.zeros((4, 1))

    mat_Rotation = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    vec_Translation = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

    # Calibrate camera
    ret, cameraMatrix, distortionCoefficients, mat_Rotation, vec_Translation= \
        cv.fisheye.calibrate(
            objectPoints,
            imagePoints,
            gray.shape[::-1],
            mat_intrinsics_in,
            vec_distortion_in,
            mat_Rotation,
            vec_Translation,
            calibration_flags,
            (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )

    # NewCameraMatrix 
    h,  w = gray.shape[:2]
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, distortionCoefficients, (w,h), 1, (w,h))

    # Region of interest (ROI) Has the format (x,y,w,h)
    roi_area = roi[2]*roi[3] #width * height

    mean_error = 0
    for i in range(len(objectPoints)):
        imgpoints2, _ = cv.projectPoints(objectPoints[i], mat_Rotation[i], vec_Translation[i], cameraMatrix, distortionCoefficients)
        error = cv.norm(imagePoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    rep_error = mean_error/len(objectPoints)

    return roi_area, rep_error

# Compute error and roi for specific indices
def determineErrorCurrentLevel(mode, dict_pathsPoints, array_indices, max_result, grayTemp, error_history, boardSize):
    """Determine error and roi for specific array of images.

    Keyword arguments:
    mode -- 'Single' or 'Stereo'
    """   
     
    array_names = []
    
    if mode == 'Single':
        
        # Extract image points for current array of indices to local list
        listPoints = []

         # Get points and names
        for indx in array_indices:
        
            # Left camera
            if dict_pathsPoints[0][0][0] is not None:
                imgPointsL = dict_pathsPoints[indx][0][1]
                listPoints.append(imgPointsL.copy())
                name = dict_pathsPoints[indx][0][0]
                temp = name.split('/')[-1]
                name = temp.split('.')[0]
                array_names.append(name)

            # Right camera
            elif dict_pathsPoints[0][1][0] is not None:
                imgPointsR = dict_pathsPoints[indx][1][1]
                listPoints.append(imgPointsR.copy())
                name = dict_pathsPoints[indx][1][0]
                temp = name.split('/')[-1]
                name = temp.split('.')[0]
                array_names.append(name)

        #print('Checking {indices}, aka {names}'.format(names=array_names, indices=array_indices))
    

        try:
            # CHECK ROI and ERROR for current level
            roi, rep_error = getROIandError(listPoints, grayTemp, boardSize)

            temp = array_indices.copy() 
            num_imgs_local = len(listPoints)
            oldMinError = max_result[num_imgs_local][1][1]

            if oldMinError > rep_error:
                max_result[num_imgs_local][1][1] = rep_error
                max_result[num_imgs_local][0][1] = roi
                max_result[num_imgs_local][2][1] = array_names.copy()
                print('({numImgs}) Images: {indices}. Error: {error}______ New minimum ______'.format(numImgs=num_imgs_local, indices=str(temp), error=str(rep_error)))
            else:
                print('({numImgs}) Images: {indices}. Error: {error}'.format(numImgs=num_imgs_local, indices=str(temp), error=str(rep_error)))

        except:
            rep_error = 1000
            print('__Computation error for combination {combination}'.format(combination=array_indices))
            error_history.append(array_names.copy())

    elif mode == 'Stereo':

        # Extract needed range from main dictionary with paths to images
        listPointsL = []
        listPointsR = []
    
        for indx in array_indices:
            imgPointsL = dict_pathsPoints[indx][0][1]
            imgPointsR = dict_pathsPoints[indx][1][1]
            listPointsL.append(imgPointsL.copy())
            listPointsR.append(imgPointsR.copy())

            name = dict_pathsPoints[indx][0][0]
            temp = name.split('/')[-1]
            name = temp.split('.')[0]
            array_names.append(name)
        
        #print('Checking {indices}, aka {names}'.format(names=array_names, indices=array_indices))

        try:
            # CHECK ROI and ERROR for current level
            roiL, rep_errorL = getROIandError(listPointsL, grayTemp, boardSize)
            roiR, rep_errorR = getROIandError(listPointsR, grayTemp, boardSize)
            roi_res = 0.5*(roiL+roiR)
            rep_error_res = 0.5*(rep_errorL+rep_errorR)
      
            temp = array_indices.copy() 

            num_imgs_local = len(listPointsR)
            oldMinError = max_result[num_imgs_local][1][1]

            if oldMinError > rep_error_res:
                max_result[num_imgs_local][1][1] = rep_error_res
                max_result[num_imgs_local][0][1] = roi_res
                max_result[num_imgs_local][2][1] = array_names.copy()
                print('({numImgs}) Images: {indices}. MEAN Error: {error}______ New minimum ______'.format(numImgs=num_imgs_local, indices=str(temp), error=str(rep_error_res)))
            else:
                print('({numImgs}) Images: {indices}. MEAN Error: {error}'.format(numImgs=num_imgs_local, indices=str(temp), error=str(rep_error_res)))

        except:
            rep_error_res = 1000
            print('__Computation error for combination {combination}'.format(combination=array_indices))
            error_history.append(array_names.copy())

        rep_error = rep_error_res

    return rep_error, max_result, error_history

# Given any dictionary with image paths and image points, return error and ROI for current indices
def findBestCombination(mode, dict_pathsPoints, array_indices, num_imgs_available, bestCombGlobal, lsHistory, max_result, grayTemp, error_history, boardSize):
  
  error_current_level = 1000
  bestCombLocal = []
  
  for j in range(num_imgs_available):
    
    testedBefore, bestCombGlobal, lsHistory = checkArray(bestCombGlobal, j, lsHistory)
    if testedBefore:
        continue
    else:
        #print('Testing indices {indices}'.format(indices=bestCombGlobal))
        rep_error, max_result, error_history = determineErrorCurrentLevel(mode, dict_pathsPoints, bestCombGlobal, max_result, grayTemp, error_history, boardSize)
                    
        # If this combination results in high errors, check another image
        if rep_error > error_current_level:
            bestCombGlobal.remove(j)             
            continue
        else:
            error_current_level = rep_error
            bestCombLocal = bestCombGlobal.copy()
            bestCombGlobal.remove(j)

  return bestCombLocal, lsHistory, max_result, error_history

# Given an image path, compute corners
def computeCorners(imgPath, boardSize):
    
    _img_shape = None
    
    print('Accessing image: '+imgPath)
    img = cv.imread(imgPath)

    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."

    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, boardSize, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)

    return ret, corners, gray

def loadImgsandImgPoints(mode, boardSize, winSize, calibration_criteria, pathImgsL, pathImgsR):
    """ Returns dictionary with paths to images and their image points.

    Keyword arguments:
    mode -- 'Single' or 'Stereo'
    pathImgs -- path to the image folder for one camera
    """       
    # Define dictionary to store all paths and image points
    dict_pathsPoints = {}
    i = 0

    # Load images for left camera
    if (mode == 'Single') and (pathImgsL is not None):

        # Remove unwanted files from directory
        paths_images_L = os.listdir(pathImgsL)
        if ".DS_Store" in paths_images_L: 
            paths_images_L.remove('.DS_Store')

        for imgPathL in paths_images_L:

            imgPath = pathImgsL+'/'+imgPathL
            
            retL, cornersL, grayL = computeCorners(imgPath, boardSize)
            
            # Refine corner search to subpixel accuracy if some corners were found and save to dictionary
            if retL is True:
                cv.cornerSubPix(grayL, cornersL, winSize, (-1,-1), calibration_criteria)
                dict_pathsPoints[i] = [[imgPath, cornersL], [None, None]]
                i+=1

            else: 
                continue

    # Load images for right camera
    elif (mode == 'Single') and (pathImgsR is not None):
       
       # Remove unwanted files from directory
        paths_images_R = os.listdir(pathImgsR)
        if ".DS_Store" in paths_images_R: 
            paths_images_R.remove('.DS_Store')
        
        for imgPathR in paths_images_R:

            imgPath = pathImgsR+'/'+imgPathR
            
            retR, cornersR, grayR = computeCorners(imgPath, boardSize)
            
            # Refine corner search to subpixel accuracy if some corners were found and save to dictionary
            if retR is True:
                cv.cornerSubPix(grayR, cornersR, (3,3), (-1,-1), calibration_criteria)
                dict_pathsPoints[i] = [[None, None], [imgPath, cornersR]]
                i+=1

            else: 
                continue

    # Load iamges for both cameras
    elif mode == 'Stereo':

        # Remove unwanted files from directory
        paths_images_L = os.listdir(pathImgsL)
        if ".DS_Store" in paths_images_L: 
            paths_images_L.remove('.DS_Store')

        paths_images_R = os.listdir(pathImgsR)
        if ".DS_Store" in paths_images_R: 
            paths_images_R.remove('.DS_Store')
    
        for pathL in paths_images_L:

            # Get left image and find path to the same image for right camera
            temp = pathL.split('_')[0]
            pathR = temp+'_R.png'

            imgPathL = pathImgsL+'/'+pathL
            imgPathR = pathImgsR+'/'+pathR

            retR, cornersR, grayR = computeCorners(imgPathR, boardSize)
            retL, cornersL, grayL = computeCorners(imgPathL, boardSize)
            
            # Refine corner search to subpixel accuracy if some corners were found and save to dictionary
            if (retL is True) and (retR is True):
                cv.cornerSubPix(grayL, cornersL, (3,3), (-1,-1), calibration_criteria)
                cv.cornerSubPix(grayR, cornersR, (3,3), (-1,-1), calibration_criteria)

                dict_pathsPoints[i] = [[imgPathL, cornersL], [imgPathR, cornersR]]
                i+=1

            else: 
                continue
 
    print('Finished loading images points...')
    return dict_pathsPoints  


def testAllImgCombinations(mode, dict_pathsPoints, camPosition, path_calibration, boardSize):
    """ Given a dictionary with paths and image points, save dictionary with best combinations for all numbers of images to file.
    Return list containing image names that result in lowest reprojection error

    Keyword arguments:
    mode -- 'Single' or 'Stereo'
    dict_pathsPoints -- Dictionary containing image paths and points [[imgPathL, cornersL], [imgPathR, cornersR]]
    camPosition -- 'L' or 'R' or None for Stereo
    """   
    ## Define lists and parameters for local functions 

    num_imgs_base = 5
    num_imgs_available = len(dict_pathsPoints.keys())
    num_levels = num_imgs_available - num_imgs_base

    max_result = {}
    for level in range(num_imgs_base,num_imgs_available):
        max_result[level] = [['ROI', 0],['Error', 1000],['Images', []]]

    lsHistory = []
    error_history = []
    error_L1 = 2000

    # Get one gray image for use in subfunctions
    if mode == 'Single' and camPosition == 'R':
        pathTemp = dict_pathsPoints[5][1][0]

    elif mode == 'Single' and camPosition == 'L':
        pathTemp = dict_pathsPoints[5][0][0]
    else:
        pathTemp = dict_pathsPoints[5][0][0]

    imgTemp = cv.imread(pathTemp)
    grayTemp = cv.cvtColor(imgTemp,cv.COLOR_BGR2GRAY)

    ## For a base number of images, determine the grouped sequence of indices that leads to lowest error
    print('Testing base combination of images...')
    for i in range(num_imgs_available):

        # Determine basis range for analyis
        first = i
        last = i+num_imgs_base

        if last > num_imgs_available:
            rest = last-num_imgs_available
            last = rest 
            array_indices = list(range(first, num_imgs_available))
            temp = range(0,last)
            array_indices += temp

        else: 
            array_indices = list(range(first, last))

        # Check if this combination was tested before
        temp = array_indices.copy()
        temp.sort()
        if temp in lsHistory:
            continue
        else:
            lsHistory.append(temp.copy())

        # Get error for current combination
        #print('Testing indices {indices}'.format(indices=array_indices))
        rep_error, max_result, error_history = determineErrorCurrentLevel(mode, dict_pathsPoints, array_indices, max_result, grayTemp, error_history, boardSize)
        
        if rep_error > error_L1:
            continue
        else:
            maxL1 = array_indices.copy()
            error_L1 = rep_error



    ## Using base combination as starting point, add images to combination and check for lowest error combination
    bestCombGlobal = maxL1.copy()
    print('Best combination base {combination}'.format(combination=maxL1))

    for n in range(num_imgs_base):
        bestCombGlobal, lsHistory, max_result, error_history = findBestCombination(mode, dict_pathsPoints, array_indices, num_imgs_available, bestCombGlobal, lsHistory, max_result, grayTemp, error_history, boardSize)

    ## NEW______________________________________________________________________________
    # When 5 (num_imgs_base) additional images are computed, since they are the optimal images from all availalable ones, use these 5 as the new base and repeat the process
    
    # Reset max_results
    max_result = {}
    for level in range(num_imgs_base,num_imgs_available):
        max_result[level] = [['ROI', 0],['Error', 1000],['Images', []]]
    
    new_base = bestCombGlobal[num_imgs_base:].copy()
    bestCombGlobal = new_base
    for n in range(num_levels):
        bestCombGlobal, lsHistory, max_result, error_history = findBestCombination(mode, dict_pathsPoints, array_indices, num_imgs_available, bestCombGlobal, lsHistory, max_result, grayTemp, error_history, boardSize)
    
    ## END NEW_________________________________________________________________________


    ## For all results in max_result find the combination with the lowest reprojection error___________________________
    lsBestCombination = []
    values = list(max_result.values())
    lowestError = 20000
    for value in values:
        error = value[1][1]
        if error < lowestError:
            lowestError = error
            lsBestCombination = value[2][1]

    # Just for checking of the printed result: Get indices of img names that were printed as best combination
    keys = list(dict_pathsPoints.keys())
    temp = []
    for name in lsBestCombination:
        name  = name.split('_')[0]
        for key in keys:
            pathL = dict_pathsPoints[key][0][0]
            if pathL == None: 
                pathL=''    
            elif pathL is not None:
                pathL = pathL.split('/')[-1]
                pathL = pathL.split('_')[0]
            
            pathR = dict_pathsPoints[key][1][0]
            if pathR == None:
                pathR=''
            elif pathR is not None:
                pathR = pathR.split('/')[-1]
                pathR = pathR.split('_')[0]
            
            if name == pathR or name == pathL:
                temp.append(key)

    if mode == 'Single':
        print('Lowest error of {error} found for combination {combination}.'.format(error=lowestError, combination=temp))
    elif mode == 'Stereo':
        print('Lowest MEAN error of {error} found for combination {combination}.'.format(error=lowestError, combination=temp))


    ## Save data to disk as csv and npy________________________________________________
    
    # Format date to be added to filename
    now = datetime.now()
    year = now.strftime('%Y')
    month = now.strftime('%m')
    day = now.strftime('%d')
    hour = now.strftime('%H')
    minutes = now.strftime('%M')
    date = year+month+day+'_'+hour+minutes

    # CSV: Export entire max_results list and entire error history
    # NPY: Export combination leading to lowest reprojection error

    header = ['NumberOfImages', 'ReprojectionError', 'ROI', 'Images']
    all_keys = list(max_result.keys())

    if mode == 'Single':
        os.chdir(path_calibration)
        np.save('bestCombinationCam'+camPosition, lsBestCombination)
        np.save('max_results_cam'+camPosition+'_'+date, max_result)

        with open('max_results_cam'+camPosition+'_'+date+'.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(header)
            
            for key in all_keys:
                valError = max_result[key][1][1]
                roi = max_result[key][0][1]
                images = max_result[key][2][1]
                data = [key, valError, roi, images]
                # write the data
                writer.writerow(data)
        print('Best combinations for camera {camPosition} saved to file.'.format(camPosition=camPosition))

        with open('error_history_cam'+camPosition+'_'+date+'.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            for element in error_history:
                writer.writerow(element)
        print('Error history for camera {camPosition} saved to file.'.format(camPosition=camPosition))


    elif mode == 'Stereo':
        os.chdir(path_calibration)
        np.save('bestCombinationStereo', lsBestCombination)
        np.save('max_results_stereo'+date, max_result)

        with open('max_results_stereo.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(header)
            
            for key in all_keys:
                valError = max_result[key][1][1]
                roi = max_result[key][0][1]
                images = max_result[key][2][1]
                data = [key, valError, roi, images]
                # write the data
                writer.writerow(data)
        print('Best combinations for stereo setup saved to file.')
    
        with open('error_history_stereo_'+date+'.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            for element in error_history:
                writer.writerow(element)
        print('Error history for for stereo setup saved to file.')

    
    return lsBestCombination


def calibrateSingle(camPosition, lsCombinations, path_calibration, calibration_criteria, squareSize, boardSize, winSize):
    
    path_cam = path_calibration+'/cam'+camPosition

    objp = np.zeros((1, boardSize[0]*boardSize[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:boardSize[0], 0:boardSize[1]].T.reshape(-1, 2)

    objp = objp * squareSize

    _img_shape = None
    objectPoints = []
    imagePoints = []

    imgNumbers = lsCombinations.copy()
    for a in range(len(imgNumbers)):
        imgNumbers[a] = imgNumbers[a].split('_')[0]

    path_images = []
    for imgNum in imgNumbers:
        path_images.append(path_cam+'/'+imgNum+'_'+camPosition+'.png')


    # Get imagePoints and objectPoints
    for i in range(len(path_images)):

        imgPath = path_images[i]
        print('Accessing image: '+imgPath)
        img = cv.imread(imgPath)

        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."

        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, boardSize, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)

        # If found, add object points, image points (after refining them)
        if ret is True:
            objectPoints.append(objp)
            cv.cornerSubPix(gray, corners, winSize, (-1,-1), calibration_criteria)
            imagePoints.append(corners)

    # Prepare matrices to be filled by calibration function
    N_OK = len(objectPoints)
    mat_intrinsics_in = np.zeros((3, 3))
    vec_distortion_in = np.zeros((4, 1))

    #vec_distortion_in only yields distortion coefficients (k1,k2,k3,k4).

    mat_Rotation = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    vec_Translation = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

    # Calibrate camera
    ret, cameraMatrix, distortionCoefficients, mat_Rotation, vec_Translation, _, _, perViewErrors= \
        cv.calibrateCameraExtended(
            objectPoints,
            imagePoints,
            gray.shape[::-1],
            mat_intrinsics_in,
            vec_distortion_in,
            mat_Rotation,
            vec_Translation,
            calibration_criteria,
        )

    #pprint(perViewErrors)

    # NewCameraMatrix (optimum depending on whether original pixels should be kept or not)
    h,  w = gray.shape[:2]
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, distortionCoefficients, (w,h), 1, (w,h))

    mean_error = 0
    for i in range(len(objectPoints)):
        imgpoints2, _ = cv.projectPoints(objectPoints[i], mat_Rotation[i], vec_Translation[i], cameraMatrix, distortionCoefficients)
        error = cv.norm(imagePoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    print( "Reprojection total error: {}".format(mean_error/len(objectPoints)))

    return cameraMatrix, distortionCoefficients, newCameraMatrix

    
def calibrateStereo(lsCombinationStereo, path_stereo, newCameraMatrixL, distortionCoefficientsL, newCameraMatrixR, distortionCoefficientsR, squareSize, boardSize):
    
    path_cam_L = path_stereo+'/camL'
    path_cam_R = path_stereo+'/camR'

    calibration_criteria = (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 100, 10e-06)
    objp = np.zeros((1, boardSize[0]*boardSize[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:boardSize[0], 0:boardSize[1]].T.reshape(-1, 2)
    
    objp = objp * squareSize

    _img_shapeR = None
    _img_shapeL = None
    objectPoints = []
    imagePointsL = [] 
    imagePointsR = [] 

    imgNumbers = lsCombinationStereo.copy()
    for a in range(len(imgNumbers)):
        imgNumbers[a] = imgNumbers[a].split('_')[0]

    path_images_L = []
    path_images_R = []
    for imgNum in imgNumbers:
        path_images_L.append(path_cam_L+'/'+imgNum+'_L.png')
        path_images_R.append(path_cam_R+'/'+imgNum+'_R.png')
    
    # Get object and image points for every image    
    for i in range(len(path_images_L)):

        imgPathL = path_images_L[i]
        imgPathR = path_images_R[i]
        
        print('Reading image: '+imgPathR)
        print('Reading image: '+imgPathL)
        imgR = cv.imread(imgPathR)
        imgL = cv.imread(imgPathL)

        if _img_shapeR == None or _img_shapeL == None:
            _img_shapeR = imgR.shape[:2]
            _img_shapeL = imgL.shape[:2]
        else:
            assert _img_shapeR == imgR.shape[:2], "All images must share the same size."

        grayR = cv.cvtColor(imgR,cv.COLOR_BGR2GRAY)
        grayL = cv.cvtColor(imgL,cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        retR, cornersR = cv.findChessboardCorners(grayR, boardSize, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)
        retL, cornersL = cv.findChessboardCorners(grayL, boardSize, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)

        # If found, add object points, image points (after refining them)
        if retR is True and retL is True:
            objectPoints.append(objp)
            cv.cornerSubPix(grayR, cornersR,(3,3),(-1,-1),calibration_criteria)
            cv.cornerSubPix(grayL, cornersL,(3,3),(-1,-1),calibration_criteria)
            imagePointsR.append(cornersR)
            imagePointsL.append(cornersL)


    # Calibrate stereo
    flags = 0
    flags |= cv.CALIB_FIX_INTRINSIC
    criteria_stereo= (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-06)

    retS, newCameraMatrixL, distortionCoefficientsL, newCameraMatrixR, distortionCoefficientsR, Rot, Trns, Emat, Fmat, perViewErrors = cv.stereoCalibrateExtended(objectPoints, imagePointsL, imagePointsR, newCameraMatrixL, distortionCoefficientsL, newCameraMatrixR, distortionCoefficientsR, grayL.shape[::-1], criteria_stereo, flags)

    error = 0
    for indx in range(len(perViewErrors)):
        x = perViewErrors[indx][0]
        y = perViewErrors[indx][1]
        error += (x**2+y**2)**(1/2)
    error = error/len(perViewErrors)
    print('Reprojectione error for stereo calibration of {} pixels'.format(error))

    return retS, newCameraMatrixL, distortionCoefficientsL, newCameraMatrixR, distortionCoefficientsR, Rot, Trns, Emat, Fmat

#%% Main functions


def getIntrinsicsLeftCamera(capture, paths, singleLGenImages, singleLTestCombinations, singleLCalibration, debuggingMode):

    # If any of the steps to calibration must be completed, enter loop until they finish
    while (singleLGenImages is True) or (singleLTestCombinations is True) or (singleLCalibration is True):
    
        if singleLGenImages is True:
            
            print('Starting image generation for camera L...')
            generateImgsSingle(capture, 'L', paths['indCamL'], numImgsGenerateSingle, calibration_criteria, boardSize)
            print('Images successfully generated and saved to file...')
            singleLGenImages is False

        elif singleLTestCombinations is True:
        
            print('Computing best image combination for camera L')
            # Load all image paths and their image points
            dict_pathsPoints = loadImgsandImgPoints('Single', boardSize, winSize, calibration_criteria, paths['indCamL'], None)

            # Find best combination for all images
            lsCombination = testAllImgCombinations('Single', dict_pathsPoints, 'L', paths['individual'], boardSize)
            print('Best image combination for camera L '+str(lsCombination))
            singleLTestCombinations is False

        elif singleLCalibration == True:

            # Load list with best combination from file
            lsBestCombination = np.load(paths['individual']+'/bestCombinationCamL.npy')

            # Compute intrinsics
            cameraMatrixL, distortionCoefficientsL, newCameraMatrixL = calibrateSingle('L', lsBestCombination, paths['individual'], calibration_criteria, squareSize, boardSize, winSize)

            #Save matrices to folder
            os.chdir(paths['individual'])
            np.save('cameraMatrixL', cameraMatrixL)
            np.save('distortionCoefficientsL', distortionCoefficientsL)
            np.save('newcameraMatrixL', newCameraMatrixL)
            print('CameraMatrixL, distortionCoefficientsL and newCameraMatrixL computed and saved to file...')

            singleLCalibration is False
            

    # After the steps were completed or right at the beginning, load intrinsics from file
    try: 
        cameraMatrixL = np.load(paths['individual']+'/cameraMatrixL.npy')
        newCameraMatrixL = np.load(paths['individual']+'/newCameraMatrixL.npy')
        distortionCoefficientsL = np.load(paths['individual']+'/distortionCoefficientsL.npy')
    except:
        print('Error loading camera intrinsics from file. Please restart calibration')
    else:   
        print('Camera L intrinsics: fx={fx:.2f}px, fy={fy:.2f}px, cx={cx:.2f}px, cy={cy:.2f}px'.format(fx=cameraMatrixL[0][0], fy=cameraMatrixL[1][1], cx=cameraMatrixL[0][2], cy=cameraMatrixL[1][2]))

     # Optional: Show result of calibration for cam
    if debuggingMode is True:
                
        #Undistort livestream to check whether calibration was successfull (this function is heavy on resources and shall not be used for constant undistortion)
        while True:
            key = cv.waitKey(25)
            if key == 27:
                print('Program was terminated by user.')
                break

            isTrue, frame = capture.read()

            dst = cv.undistort(frame, cameraMatrixL, distortionCoefficientsL, None, newCameraMatrixL)

            cv.imshow('Camera L - Undistored', dst)



def getIntrinsicsRightCamera(capture, paths, singleRGenImages, singleRTestCombinations, singleRCalibration, debuggingMode):

   # If any of the steps to calibration must be completed, enter loop until they finish
    while (singleRGenImages is True) or (singleRTestCombinations is True) or (singleRCalibration is True):
    
        if singleRGenImages is True:
            
            print('Starting image generation for camera R...')
            generateImgsSingle(capture, 'R', paths['indCamR'], numImgsGenerateSingle, calibration_criteria, boardSize)
            print('Images successfully generated and saved to file...')
            singleRGenImages is False

        elif singleRTestCombinations is True:
        
            print('Computing best image combination for camera R')
            # Load all image paths and their image points
            dict_pathsPoints = loadImgsandImgPoints('Single', boardSize, winSize, calibration_criteria, None, paths['indCamR'])

            # Find best combination for all images
            lsCombination = testAllImgCombinations('Single', dict_pathsPoints, 'R', paths['individual'], boardSize)
            print('Best image combination for camera R '+str(lsCombination))
            singleRTestCombinations is False

        elif singleRCalibration == True:

            # Load list with best combination from file
            lsBestCombination = np.load(paths['individual']+'/bestCombinationCamR.npy')

            # Compute intrinsics
            cameraMatrixR, newCameraMatrixR, distortionCoefficientsR = calibrateSingle('R', lsBestCombination, paths['individual'], calibration_criteria, squareSize, boardSize, winSize)

            #Save matrices to folder
            os.chdir(paths['individual'])
            np.save('cameraMatrixR', cameraMatrixR)
            np.save('distortionCoefficientsR', distortionCoefficientsR)
            np.save('newcameraMatrixR', newCameraMatrixR)
            print('CameraMatrix, distortionCoefficients and newCameraMatrix computed and saved to file...')
            singleRCalibration is False
            

    # After the steps were completed or right at the beginning, load intrinsics from file
    try: 
        cameraMatrixR = np.load(paths['individual']+'/cameraMatrixR.npy')
        newCameraMatrixR = np.load(paths['individual']+'/newCameraMatrixR.npy')
        distortionCoefficientsR = np.load(paths['individual']+'/distortionCoefficientsR.npy')
    except:
        print('Error loading camera intrinsics from file. Please restart calibration')
    else:   
        print('Camera R intrinsics: fx={fx:.2f}px, fy={fy:.2f}px, cx={cx:.2f}px, cy={cy:.2f}px'.format(fx=cameraMatrixR[0][0], fy=cameraMatrixR[1][1], cx=cameraMatrixR[0][2], cy=cameraMatrixR[1][2]))
        
     # Optional: Show result of calibration for cam
    if debuggingMode is True:
                
        #Undistort livestream to check whether calibration was successfull (this function is heavy on resources and shall not be used for constant undistortion)
        while True:
            key = cv.waitKey(25)
            if key == 27:
                print('Program was terminated by user.')
                break

            isTrue, frame = capture.read()

            dst = cv.undistort(frame, cameraMatrixR, distortionCoefficientsR, None, newCameraMatrixR)

            cv.imshow('Camera R - Undistored', dst)




def calibrateStereoSetup(capL, capR, paths, stereoGenImages, stereoTestCombinations, stereoCalibration, debuggingMode):
    # If any of the steps to calibration must be completed, enter loop until they finish
    while (stereoGenImages is True) or (stereoTestCombinations is True) or (stereoCalibration is True):
    
        if stereoGenImages is True:
            print('Starting image generation for stereo setup...')
            generateImagesStereo(capL, capR, paths['stCamL'], paths['stCamR'], numImgsGenerateStereo, calibration_criteria, boardSize)
            print('Images successfully generated and saved to file...')
            stereoGenImages is False

        elif stereoTestCombinations is True:
        
            print('Computing best image combination for stereo setup')
            # Load all image paths and their image points
            dict_pathsPoints = loadImgsandImgPoints('Stereo', boardSize, winSize, calibration_criteria, paths['indCamL'], paths['indCamR']) 

            # Find best combination for all images
            lsCombination = testAllImgCombinations('Stereo', dict_pathsPoints, None, paths['stereo'], boardSize)
            print('Best image combination for stereo setup '+str(lsCombination))
            stereoTestCombinations is False


        elif stereoCalibration is True:

            # Load intrinsics
            newCameraMatrixL = np.load(paths['individual']+'/newCameraMatrixL.npy')
            distortionCoefficientsL = np.load(paths['individual']+'/distortionCoefficientsL.npy')
            newCameraMatrixR = np.load(paths['individual']+'/newCameraMatrixR.npy')
            distortionCoefficientsR = np.load(paths['individual']+'/distortionCoefficientsR.npy')

            # Load list with best combination from file
            lsCombination = np.load(paths['stereo']+'/bestCombinationStereo.npy')

            # Compute stereo parameters
            retS, newCameraMatrixL, distortionCoefficientsL, newCameraMatrixR, distortionCoefficientsR, Rot, Trns, Emat, Fmat = calibrateStereo(lsCombination, paths['stereo'], newCameraMatrixL, distortionCoefficientsL, newCameraMatrixR, distortionCoefficientsR, squareSize, boardSize)

            # Save stereo parameters
            np.save(paths['stereo']+'/rotationVector', Rot)
            np.save(paths['stereo']+'/translationVector', Trns)
            np.save(paths['stereo']+'/essentialMatrixE', Emat)
            np.save(paths['stereo']+'/fundamentalMatrixF', Fmat)
            print('Saved stereo matrices to file...')
            stereoCalibration is False
            
            print('Translation from cameraR to cameraL: x={:.2f}cm, y={:.2f}cm, z={:.2f}cm'.format(Trns[0][0], Trns[1][0], Trns[2][0]))
            #print('Reminder: z is positive towards the front of the cameras. X is positive to the right and Y is positive downwards. Hence, x must be negative in the above result')



def getRectificationMap(capL, capR, paths, newRectificationMapping, debuggingMode):

    if newRectificationMapping is True:
        # Load intrinsics and extrinics from calibration process
        newCameraMatrixL = np.load(paths['individual']+'/newCameraMatrixL.npy')
        distortionCoefficientsL = np.load(paths['individual']+'/distortionCoefficientsL.npy')
        newCameraMatrixR = np.load(paths['individual']+'/newCameraMatrixR.npy')
        distortionCoefficientsR = np.load(paths['individual']+'/distortionCoefficientsR.npy')


        Rot = np.load(paths['stereo']+'/rotationVector.npy')
        Trns = np.load(paths['stereo']+'/translationVector.npy')
        Emat = np.load(paths['stereo']+'/essentialMatrixE.npy')
        Fmat = np.load(paths['stereo']+'/fundamentalMatrixF.npy')
        print('Loaded camera matrices')

        '''
        Change this so that all bad images are removed from the folder and then select one image by random
        '''
        # Load images for mapping
        path_img_L = '/Users/niklas/Virtual_Environment/Version_5/projectAutonomous/1_Calibration/stereo/camL/7_L.png'
        path_img_R = '/Users/niklas/Virtual_Environment/Version_5/projectAutonomous/1_Calibration/stereo/camR/7_R.png'
        imgL = cv.imread(path_img_L)
        imgR = cv.imread(path_img_R)

        gray = cv.cvtColor(imgL,cv.COLOR_BGR2GRAY)
        imgSize = gray.shape[::-1]

        R_L, R_R, proj_mat_l, proj_mat_r, Q, roiL, roiR= cv.stereoRectify(newCameraMatrixL, distortionCoefficientsL, newCameraMatrixR, distortionCoefficientsR, imgSize, Rot, Trns, flags=cv.CALIB_ZERO_DISPARITY , alpha=1)

        leftMapX, leftMapY = cv.initUndistortRectifyMap(newCameraMatrixL, distortionCoefficientsL, R_L, proj_mat_l, imgSize, cv.CV_32FC1)
        rightMapX, rightMapY = cv.initUndistortRectifyMap(newCameraMatrixR, distortionCoefficientsR, R_R, proj_mat_r, imgSize, cv.CV_32FC1)    

        # Save rectification maps, Q to file
        np.save(paths['stereo']+'/rectificationMapCamLX', leftMapX)
        np.save(paths['stereo']+'/rectificationMapCamLY', leftMapY)
        np.save(paths['stereo']+'/rectificationMapCamRX', rightMapX)
        np.save(paths['stereo']+'/rectificationMapCamRY', rightMapY)
        np.save(paths['stereo']+'/reprojectionMatrixQ', Q)


    else:
        # Load from file
        leftMapX = np.load(paths['stereo']+'/rectificationMapCamLX.npy')
        leftMapY = np.load(paths['stereo']+'/rectificationMapCamLY.npy')
        rightMapX = np.load(paths['stereo']+'/rectificationMapCamRX.npy')
        rightMapY = np.load(paths['stereo']+'/rectificationMapCamRY.npy')
        Q = np.load(paths['stereo']+'/reprojectionMatrixQ.npy')

    
    if debuggingMode is True:

        newCameraMatrixL = np.load(paths['individual']+'/newCameraMatrixL.npy')
        distortionCoefficientsL = np.load(paths['individual']+'/distortionCoefficientsL.npy')
        newCameraMatrixR = np.load(paths['individual']+'/newCameraMatrixR.npy')
        distortionCoefficientsR = np.load(paths['individual']+'/distortionCoefficientsR.npy')

        # Take an image of a horizontal chessboard for debugging purposes
        path_testing = paths['stereo']+'/testing'
        os.mkdir(path_testing)

        takeTestPhoto = False
        while True and (takeTestPhoto is True) :

            key = cv.waitKey(25)

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
                imgPathL = path_testing+'/camLTesting.png'
                imgPathR = path_testing+'/camRTesting.png'

                cv.imwrite(imgPathL,frameL)
                cv.imwrite(imgPathR,frameR)
                break

        cv.destroyAllWindows()

        # Load and rectify the image pair
        imgL = cv.imread(path_testing+'/camLTesting.png')
        imgR = cv.imread(path_testing+'/camRTesting.png')

        Left_rectified = cv.remap(imgL,leftMapX,leftMapY, cv.INTER_LINEAR, cv.BORDER_CONSTANT)
        Right_rectified = cv.remap(imgR,rightMapX,rightMapY, cv.INTER_LINEAR, cv.BORDER_CONSTANT)

        grayL = cv.cvtColor(Left_rectified,cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(Right_rectified,cv.COLOR_BGR2GRAY)

        # Settings for chessboard corners
        font = cv.FONT_HERSHEY_PLAIN
        fontScale = 2
        boardSize = (9,6)
        subpix_criteria = (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 100, 10e-06)

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
                depthcamL  = ((-Trns[0][0])*newCameraMatrixL[0][0])/(disp_x)
                disp_x_text = ('Camera L disparity: {}px, depth: {:.2f}cm'.format(disp_x, depthcamL))
                cv.putText(vis_afterRectification, disp_x_text, (x_l + int((x_r+Left_rectified.shape[1]-x_l)/2), y_r -10), font, fontScale, (255, 0, 0), 2, cv.LINE_AA)

                slope = (y_l-y_r)/(x_r+Left_rectified.shape[1]-x_l)
                slopes.append(slope)

            avg = sum(slopes)/len(slopes)
            cv.putText(vis_afterRectification, 'Average slope '+str(avg),(vis_afterRectification.shape[1]//3, (vis_afterRectification.shape[0]//5)*4), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
            
        else:
            print('No chessboard found!')

        cv.imshow('Rectification check - before rectification', vis_beforeRectification)
        cv.imshow('Rectification check - after rectification', vis_afterRectification)
        cv.waitKey(0)

    return leftMapX, leftMapY, rightMapX, rightMapY, Q


def initiateDepthDetection(Left_rectified, Right_rectified, Q, paths, debuggingMode):

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


    win_left_disp = 'Left disparity map'
    win_filtered_disp = 'Filtered disparity map'
    cv.namedWindow(win_left_disp)
    cv.namedWindow(win_filtered_disp)

    Left_rectified = cv.cvtColor(Left_rectified,cv.COLOR_BGR2GRAY)
    Right_rectified = cv.cvtColor(Right_rectified,cv.COLOR_BGR2GRAY)

    font = cv.FONT_HERSHEY_PLAIN
    fontScale = 2

    if debuggingMode is True:
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

    disparityMap = left_disp_forVis
    pointCloud = points_3D_LM

    return disparityMap, pointCloud