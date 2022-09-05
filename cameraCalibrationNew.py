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
        
    print('Starting generation of images for camera {}...'.format(camPosition))

    winSize = (5, 5)

    ## Elementary settings
    value_timer = int(1)

    # (Re)set timer
    timer = value_timer

    # (Re)set number of valid calibration images
    validImg = 0

    # Set flag to run program until calibration and correction was successfully executed
    flagAborted = False

    while (True and flagAborted==False):

        key = cv.waitKey(25)

        if key == 27:
            print('Program was terminated by user.')
            flagAborted = True
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
                flagAborted = True
                break
        
        # Exit livestream when required number of images exists
        break

    cv.destroyAllWindows()

    if flagAborted == False:
        print('All images successfully generated and saved to file...')
    else:
        print('Generating images aborted...')
    
    return False

def generateImagesStereo(capL, capR, path_cam_L, path_cam_R, numImgsGenerateStereo, calibration_criteria, boardSize):
    
    print('Starting generation of images for stereo...')

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

# Given a list of image points, calculate reprojection error
def getRepError(listPoints, gray, boardSize):
    '''
    The same procedure is used for single and stereo setup because 
    '''

    objp = np.zeros((1, boardSize[0]*boardSize[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:boardSize[0], 0:boardSize[1]].T.reshape(-1, 2)
    objp = objp * squareSize
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
    _, cameraMatrix, distortionCoefficients, mat_Rotation, vec_Translation= \
        cv.calibrateCamera(
            objectPoints,
            imagePoints,
            gray.shape[::-1],
            mat_intrinsics_in,
            vec_distortion_in,
            mat_Rotation,
            vec_Translation,
            0,
            calibration_criteria,
        )

    ## Compute overall mean reprojection error for image combination
    overallMeanRMS = 0

    for i in range(len(objectPoints)):
        projectedImgPoints, _ = cv.projectPoints(objectPoints[i], mat_Rotation[i], vec_Translation[i], cameraMatrix, distortionCoefficients)

        sumRMS = 0

        # For every corner in a chessboard view, calculate RMS error
        for j in range(len(projectedImgPoints)):
            deltaX = projectedImgPoints[j][0][0] - imagePoints[i][j][0][0]
            deltaY = projectedImgPoints[j][0][1] - imagePoints[i][j][0][1]
            #print('Delta: x={}, y={}'.format(deltaX, deltaY))

            RMS = (deltaX**2+deltaY**2)**(0.5)
            sumRMS += RMS

        # Calculate mean error per image
        meanRMSImg = sumRMS / len(projectedImgPoints)
        overallMeanRMS += meanRMSImg

    # Calculate mean error for all images
    overallMeanRMS = overallMeanRMS / len(objectPoints)

    return overallMeanRMS

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

        #print('Checking {indices}, aka {names}'.format(indices=array_indices, names=array_names))

        try:
            # Check error for current level
            rep_error = getRepError(listPoints, grayTemp, boardSize)

            temp = array_indices.copy() 
            num_imgs_local = len(listPoints)
            oldMinError = max_result[num_imgs_local][0][1]
            
            if oldMinError > rep_error:
                max_result[num_imgs_local][0][1] = rep_error
                max_result[num_imgs_local][1][1] = array_names.copy()
                print('({numImgs}) Images: {indices}. Re-projection error: {error}______ New minimum ______'.format(numImgs=num_imgs_local, indices=str(temp), error=str(rep_error)))
            else:
                print('({numImgs}) Images: {indices}. Re-projection error: {error}'.format(numImgs=num_imgs_local, indices=str(temp), error=str(rep_error)))

        except:
            rep_error = 1000
            print('__Computation error for combination {combination}'.format(combination=array_indices))
            errorImg = array_names.copy()
            errorImg = errorImg[-1] # Sure that this is correct?
            error_history.append(errorImg)

    elif mode == 'Stereo':
        '''
        Accuracy is ranked using the mean reprojection error for both images. Technically, this could be done using one call to 
        stereoCalibrate() instead, but the procedure below is accurate enough.
        '''

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
            # Compute mean reprojection error for two images of the same view
            rep_errorL = getRepError(listPointsL, grayTemp, boardSize)
            rep_errorR = getRepError(listPointsR, grayTemp, boardSize)
            rep_error_res = 0.5*(rep_errorL+rep_errorR)
      
            temp = array_indices.copy() 

            num_imgs_local = len(listPointsR)
            oldMinError = max_result[num_imgs_local][0][1]

            if oldMinError > rep_error_res:
                max_result[num_imgs_local][0][1] = rep_error_res
                max_result[num_imgs_local][1][1] = array_names.copy()
                print('({numImgs}) Images: {indices}. MEAN Error: {error}______ New minimum ______'.format(numImgs=num_imgs_local, indices=str(array_names), error=str(rep_error_res)))
            else:
                print('({numImgs}) Images: {indices}. MEAN Error: {error}'.format(numImgs=num_imgs_local, indices=str(array_names), error=str(rep_error_res)))

        except:
            rep_error = 1000
            print('__Computation error for combination {combination}'.format(combination=array_indices))
            errorImg = array_names.copy()
            errorImg = errorImg[-1] # Sure that this is correct?
            error_history.append(errorImg)

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
    """ Returns dictionary with paths to images and their image points

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

            _img_shape = None
    
            print('Accessing image: '+imgPath)
            img = cv.imread(imgPath)

            if _img_shape == None:
                _img_shape = img.shape[:2]
            else:
                assert _img_shape == img.shape[:2], "All images must share the same size."

            grayL = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

            retL, cornersL = cv.findChessboardCorners(grayL, boardSize, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)

            
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
                cv.cornerSubPix(grayR, cornersR, winSize, (-1,-1), calibration_criteria)
                dict_pathsPoints[i] = [[None, None], [imgPath, cornersR]]
                i+=1

            else: 
                continue

    # Load images for both cameras
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
                cv.cornerSubPix(grayL, cornersL, winSize, (-1,-1), calibration_criteria)
                cv.cornerSubPix(grayR, cornersR, winSize, (-1,-1), calibration_criteria)

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
    for level in range(num_imgs_base,num_imgs_available+1):
        max_result[level] = [['Error', 1000],['Images', []]]

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
    for i in range(num_imgs_available+1):

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
    print('Best combination base: {combination}'.format(combination=maxL1))

    for n in range(num_imgs_base):
        bestCombGlobal, lsHistory, max_result, error_history = findBestCombination(mode, dict_pathsPoints, array_indices, num_imgs_available, bestCombGlobal, lsHistory, max_result, grayTemp, error_history, boardSize)


    # Keep max result for 5 images
    minErrorTemp = max_result[num_imgs_base][0][1]
    minErrorCombTemp = max_result[num_imgs_base][1][1]

    # Reset max_results
    max_result = {}
    for level in range(num_imgs_base,num_imgs_available+1):
        max_result[level] = [['Error', 1000],['Images', []]]
    
    # Save previously calculated max result for base level
    max_result[num_imgs_base][0][1] = minErrorTemp
    max_result[num_imgs_base][1][1] = minErrorCombTemp
    
    new_base = bestCombGlobal[num_imgs_base:].copy()
    bestCombGlobal = new_base
    for n in range(num_levels):
        bestCombGlobal, lsHistory, max_result, error_history = findBestCombination(mode, dict_pathsPoints, array_indices, num_imgs_available, bestCombGlobal, lsHistory, max_result, grayTemp, error_history, boardSize)

    ## For all results in max_result find the combination with the lowest reprojection error
    lsBestCombination = []
    values = list(max_result.values())
    lowestError = 20000
    for value in values:
        error = value[0][1]
        if error < lowestError:
            lowestError = error
            lsBestCombination = value[1][1]

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
        print('Lowest reprojection error of {error:.5f} pixels found for combination {combination}.'.format(error=lowestError, combination=lsBestCombination))
    elif mode == 'Stereo':
        print('Lowest MEAN reprojection error of {error:.5f} pixels found for combination {combination}.'.format(error=lowestError, combination=lsBestCombination))


    ## Save data to disk as csv and npy
    header = ['NumberOfImages', 'ReprojectionError', 'Images']
    all_keys = list(max_result.keys())

    if mode == 'Single':
        os.chdir(path_calibration)
        np.save('bestCombinationCam'+camPosition, lsBestCombination)
        np.save('max_results_cam'+camPosition, max_result)

        with open('max_results_cam'+camPosition+'.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(header)
            
            for key in all_keys:
                valError = max_result[key][0][1]
                images = max_result[key][1][1]
                data = [key, valError, images]
               
                writer.writerow(data)
        print('Best combination for camera {camPosition} saved to file.'.format(camPosition=camPosition))

        with open('error_history_cam'+camPosition+'.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            for element in error_history:
                writer.writerow(element)
        print('Error history for camera {camPosition} saved to file.'.format(camPosition=camPosition))


    elif mode == 'Stereo':
        os.chdir(path_calibration)
        np.save('bestCombinationStereo', lsBestCombination)
        np.save('max_results_stereo', max_result)

        with open('max_results_stereo.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            writer.writerow(header)
            
            for key in all_keys:
                valError = max_result[key][0][1]
                images = max_result[key][1][1]
                data = [key, valError, images]
                writer.writerow(data)
        print('Best combinations for stereo setup saved to file.')
    
        with open('error_history_stereo.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            for element in error_history:
                writer.writerow(element)
        print('Error history for for stereo setup saved to file.')
    
    return lsBestCombination

def calibrateSingle(camPosition, lsBestCombination, path_calibration, calibration_criteria, squareSize, boardSize, winSize):
    
    print('Starting calibration for camera {}'.format(camPosition))
    
    path_cam = path_calibration+'/cam'+camPosition

    # Setup matrices for objectpoints and imagepoints
    objp = np.zeros((1, boardSize[0]*boardSize[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:boardSize[0], 0:boardSize[1]].T.reshape(-1, 2)
    objp = objp * squareSize
    objectPoints = []
    imagePoints = []

    _img_shape = None

    # Get full paths to the images listed in 'lsBestCombination'
    imgPaths = [path_cam+'/'+imgNumber.split('_')[0]+'_'+camPosition+'.png' for imgNumber in lsBestCombination]


    # Compute imagePoints and objectPoints for the respective images
    for i in range(len(imgPaths)):

        imgPath = imgPaths[i]
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
    mat_Rotation = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    vec_Translation = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

    # Calibrate camera
    _, cameraMatrix, distortionCoefficients, mat_Rotation, vec_Translation = \
        cv.calibrateCamera(
            objectPoints,
            imagePoints,
            gray.shape[::-1],
            mat_intrinsics_in,
            vec_distortion_in,
            mat_Rotation,
            vec_Translation,
            0,
            calibration_criteria,
        )

    # Optimise camera matrix (optimum depending on whether original pixels should be kept or not)
    h,  w = gray.shape[:2]
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, distortionCoefficients, (w,h), 1, (w,h))


    ## Compute overall mean reprojection error for image combination
    overallMeanRMS = 0
    errorVis = []
    errorVisHeader = ['Image', 'Corner', 'xProjected', 'yProjected', 'xOriginal', 'yOriginal', 'deltaX', 'deltaY', 'RMS error']

    for i in range(len(objectPoints)):
        projectedImgPoints, _ = cv.projectPoints(objectPoints[i], mat_Rotation[i], vec_Translation[i], cameraMatrix, distortionCoefficients)

        sumRMS = 0

        # For every corner in a chessboard view, calculate RMS error
        for j in range(len(projectedImgPoints)):
            deltaX = projectedImgPoints[j][0][0] - imagePoints[i][j][0][0]
            deltaY = projectedImgPoints[j][0][1] - imagePoints[i][j][0][1]
            #print('Delta: x={}, y={}'.format(deltaX, deltaY))

            RMS = (deltaX**2+deltaY**2)**(0.5)
            sumRMS += RMS

            imgName = lsBestCombination[i]

            temp = [imgName, j, projectedImgPoints[j][0][0], projectedImgPoints[j][0][1], imagePoints[i][j][0][0], imagePoints[i][j][0][1], deltaX, deltaY, RMS]
            errorVis.append(temp.copy())

        # Calculate mean error per image
        meanRMSImg = sumRMS / len(projectedImgPoints)
        print('Mean RMS error for current image: {} pixels'.format(meanRMSImg))
        overallMeanRMS += meanRMSImg

    # Calculate mean error for all images
    overallMeanRMS = overallMeanRMS / len(imgPaths)

    print('Overall mean RMS error: {} pixels'.format(overallMeanRMS))
    
    # Output error vis to csv
    with open(path_calibration+'/calibrationError_bestCombination_cam'+camPosition+'.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(errorVisHeader)
        for element in errorVis:
            writer.writerow(element)

    return cameraMatrix, distortionCoefficients, newCameraMatrix

    
def calibrateStereo(lsBestCombinationtereo, paths, newCameraMatrixL, distortionCoefficientsL, newCameraMatrixR, distortionCoefficientsR):
    
    print('Starting stereo calibration')
    path_cam_L = paths['stereo']+'/camL'
    path_cam_R = paths['stereo']+'/camR'

    # Setup matrices for objectpoints and imagepoints
    objp = np.zeros((1, boardSize[0]*boardSize[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:boardSize[0], 0:boardSize[1]].T.reshape(-1, 2)
    objp = objp * squareSize
    objectPoints = []
    imagePointsL = [] 
    imagePointsR = [] 

    _img_shapeR = None
    _img_shapeL = None

    imgNumbers = lsBestCombinationtereo.copy()
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
            cv.cornerSubPix(grayR, cornersR, winSize, (-1,-1),calibration_criteria)
            cv.cornerSubPix(grayL, cornersL, winSize, (-1,-1),calibration_criteria)
            imagePointsR.append(cornersR)
            imagePointsL.append(cornersL)


    # Calibrate stereo
    flags = 0
    flags |= cv.CALIB_FIX_INTRINSIC

    retS, _, _, _, _, Rot, Trns, Emat, Fmat = cv.stereoCalibrate(objectPoints, imagePointsL, imagePointsR, newCameraMatrixL, distortionCoefficientsL, newCameraMatrixR, distortionCoefficientsR, grayL.shape[::-1], calibration_criteria, flags)

    # Save stereo parameters
    np.save(paths['stereo']+'/rotationVector', Rot)
    np.save(paths['stereo']+'/translationVector', Trns)
    print('Saved stereo matrices T and R to file...')
    
    
    ## Compute overall mean reprojection error for graphic display comparison with previous single camera calibration________________
    N_OK = len(objectPoints)
 
    errorVis = []
    errorVisHeader = ['Camera', 'Image', 'Corner', 'xProjected', 'yProjected', 'xOriginal', 'yOriginal', 'deltaX', 'deltaY', 'RMS error']

    # Calibrate camera L
    mat_Rotation = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    vec_Translation = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    mat_intrinsics_in = np.zeros((3, 3))
    vec_distortion_in = np.zeros((4, 1))
    _, cameraMatrixL, distortionCoefficientsL, mat_RotationL, vec_TranslationL= \
        cv.calibrateCamera(
            objectPoints,
            imagePointsL,
            grayL.shape[::-1],
            mat_intrinsics_in,
            vec_distortion_in,
            mat_Rotation,
            vec_Translation,
            0,
            calibration_criteria,
        )
    
    overallMeanRMS_L = 0
    for i in range(len(objectPoints)):
        projectedImgPointsL, _ = cv.projectPoints(objectPoints[i], mat_RotationL[i], vec_TranslationL[i], cameraMatrixL, distortionCoefficientsL)

        sumRMS = 0

        # For every corner in a chessboard view, calculate RMS error
        for j in range(len(projectedImgPointsL)):
            deltaX = projectedImgPointsL[j][0][0] - imagePointsL[i][j][0][0]
            deltaY = projectedImgPointsL[j][0][1] - imagePointsL[i][j][0][1]
            #print('Delta: x={}, y={}'.format(deltaX, deltaY))

            RMS = (deltaX**2+deltaY**2)**(0.5)
            sumRMS += RMS

            imgName = lsBestCombinationtereo[i].split("_")[0]+'_L'

            temp = ['Camera L', imgName, j, projectedImgPointsL[j][0][0], projectedImgPointsL[j][0][1], imagePointsL[i][j][0][0], imagePointsL[i][j][0][1], deltaX, deltaY, RMS]
            errorVis.append(temp.copy())

        # Calculate mean error per image
        meanRMSImg = sumRMS / len(projectedImgPointsL)
        #print('Mean RMS error for image {}: {:.5f} pixels'.format(imgName, meanRMSImg))
        overallMeanRMS_L += meanRMSImg

    # Calculate mean error for all images
    overallMeanRMS_L = overallMeanRMS_L / len(lsBestCombinationtereo)
    print('Overall mean RMS error camera L: {:.5f} pixels'.format(overallMeanRMS_L))
    
    
    # Calibrate camera R
    mat_Rotation = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    vec_Translation = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    mat_intrinsics_in = np.zeros((3, 3))
    vec_distortion_in = np.zeros((4, 1))
    _, cameraMatrixR, distortionCoefficientsR, mat_RotationR, vec_TranslationR= \
        cv.calibrateCamera(
            objectPoints,
            imagePointsR,
            grayR.shape[::-1],
            mat_intrinsics_in,
            vec_distortion_in,
            mat_Rotation,
            vec_Translation,
            0,
            calibration_criteria,
        )
    
    overallMeanRMS_R = 0
    for i in range(len(objectPoints)):
        projectedImgPointsR, _ = cv.projectPoints(objectPoints[i], mat_RotationR[i], vec_TranslationR[i], cameraMatrixR, distortionCoefficientsR)

        sumRMS = 0

        # For every corner in a chessboard view, calculate RMS error
        for j in range(len(projectedImgPointsR)):
            deltaX = projectedImgPointsR[j][0][0] - imagePointsR[i][j][0][0]
            deltaY = projectedImgPointsR[j][0][1] - imagePointsR[i][j][0][1]
            #print('Delta: x={}, y={}'.format(deltaX, deltaY))

            RMS = (deltaX**2+deltaY**2)**(0.5)
            sumRMS += RMS

            imgName = lsBestCombinationtereo[i].split("_")[0]+'_R'

            temp = ['Camera R', imgName, j, projectedImgPointsR[j][0][0], projectedImgPointsR[j][0][1], imagePointsR[i][j][0][0], imagePointsR[i][j][0][1], deltaX, deltaY, RMS]
            errorVis.append(temp.copy())

        # Calculate mean error per image
        meanRMSImg = sumRMS / len(projectedImgPointsL)
        #print('Mean RMS error for image {}: {:.5f} pixels'.format(imgName, meanRMSImg))
        overallMeanRMS_R += meanRMSImg

    # Calculate mean error for all images
    overallMeanRMS_R = overallMeanRMS_R / len(lsBestCombinationtereo)
    print('Overall mean RMS error camera R: {:.5f} pixels'.format(overallMeanRMS_R))
    
    # Calculate mean error for both cameras
    bothCamsMeanRMS = 0.5*(overallMeanRMS_L+overallMeanRMS_R)
    print('Overall mean RMS error for stereo setup: {:.5f} pixels'.format(bothCamsMeanRMS))
    
    # Output error vis to csv
    with open(paths['stereo']+'/evaluationRepErrorStereo.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(errorVisHeader)
        for element in errorVis:
            writer.writerow(element)

#%% Main functions

def getIntrinsicsLeftCamera(capture, paths, singleLGenImages, singleLTestCombinations, singleLCalibration, debuggingMode):

    print('{:^75s}'.format('_____ Computing intrinsics for left camera _____'))

    # If any of the steps to calibration must be completed, enter loop until they finish
    while (singleLGenImages is True) or (singleLTestCombinations is True) or (singleLCalibration is True):
    
        if singleLGenImages is True: #Works

            print('Starting image generation for camera L...')
            generateImgsSingle(capture, 'L', paths['indCamL'], numImgsGenerateSingle, calibration_criteria, boardSize)
            singleLGenImages = False

        elif singleLTestCombinations is True: #Works
        
            print('Computing best image combination for camera L')
            # Create a dictionary with paths to all images and their respective image points 
            dict_pathsPoints = loadImgsandImgPoints('Single', boardSize, winSize, calibration_criteria, paths['indCamL'], None)

            # Find best combination for all images
            lsBestCombination = testAllImgCombinations('Single', dict_pathsPoints, 'L', paths['individual'], boardSize)
            
            singleLTestCombinations = False

        elif singleLCalibration == True: #Works

            # Load list with best combination from file
            try:
                lsBestCombination = np.load(paths['individual']+'/bestCombinationCamL.npy')
            except:
                print('Failed to load best image combination. Please compute best combination first')

            cameraMatrixL, distortionCoefficientsL, newCameraMatrixL = calibrateSingle('L', lsBestCombination, paths['individual'], calibration_criteria, squareSize, boardSize, winSize)

            #Save matrices to folder
            os.chdir(paths['individual'])
            np.save('distortionCoefficientsL', distortionCoefficientsL)
            np.save('cameraMatrixL', cameraMatrixL)
            np.save('newcameraMatrixL', newCameraMatrixL)
            print('CameraMatrixL, distortionCoefficientsL and newCameraMatrixL computed and saved to file...')

            singleLCalibration = False
            

    ## After the steps were completed or right at the beginning, load intrinsics from file
    print('Attempting to load intrinsics from file...')
    try: 
        cameraMatrixL = np.load(paths['individual']+'/cameraMatrixL.npy')
        newCameraMatrixL = np.load(paths['individual']+'/newCameraMatrixL.npy')
        distortionCoefficientsL = np.load(paths['individual']+'/distortionCoefficientsL.npy')
    except:
        print('Error loading camera intrinsics from file. Please restart calibration')
    else:   
        print('Camera L intrinsics: fx={fx:.2f}px, fy={fy:.2f}px, cx={cx:.2f}px, cy={cy:.2f}px'.format(fx=newCameraMatrixL[0][0], fy=newCameraMatrixL[1][1], cx=newCameraMatrixL[0][2], cy=newCameraMatrixL[1][2]))
        print('Camera L distortion coefficients: k1={}, k2={}, p1={}, p2={}, k3={} '.format(distortionCoefficientsL[0][0], distortionCoefficientsL[1][0], distortionCoefficientsL[2][0], distortionCoefficientsL[3][0], distortionCoefficientsL[4][0]))
     

     # Optional: Show result of calibration for cam
    if debuggingMode is True: #Works
                
        # Undistort livestream to provide plausability check whether calibration was successfull
        while True:
            key = cv.waitKey(25)
            if key == 27:
                print('User closed undistorted view.')
                break

            isTrue, frame = capture.read()
          
            dst = cv.undistort(frame, cameraMatrixL, distortionCoefficientsL, None, newCameraMatrixL)
            
            cv.imshow('Camera L - Original', frame)
            cv.imshow('Camera L - Undistored', dst)

    print('{:^75s}'.format('_____ Finished computing intrinsics for left camera _____'))

    return newCameraMatrixL

def getIntrinsicsRightCamera(capture, paths, singleRGenImages, singleRTestCombinations, singleRCalibration, debuggingMode):

    print('{:^75s}'.format('_____ Computing intrinsics for right camera _____'))

   ## If any of the steps to calibration must be completed, enter loop until they finish
    while (singleRGenImages is True) or (singleRTestCombinations is True) or (singleRCalibration is True):
    
        if singleRGenImages is True: #Works
            
            print('Starting image generation for camera R...')
            generateImgsSingle(capture, 'R', paths['indCamR'], numImgsGenerateSingle, calibration_criteria, boardSize)
            singleRGenImages = False

        elif singleRTestCombinations is True: #Works
        
            print('Computing best image combination for camera R')
            # Create a dictionary with paths to all images and their respective image points 
            dict_pathsPoints = loadImgsandImgPoints('Single', boardSize, winSize, calibration_criteria, None, paths['indCamR'])

            # Find best combination for all images
            lsBestCombination = testAllImgCombinations('Single', dict_pathsPoints, 'R', paths['individual'], boardSize)

            singleRTestCombinations = False

        elif singleRCalibration == True: #Works

            # Load list with best combination from file
            try:
                lsBestCombination = np.load(paths['individual']+'/bestCombinationCamR.npy')
            except:
                print('Failed to load best image combination. Please compute best combination first')

            # Compute intrinsics
            cameraMatrixR, distortionCoefficientsR, newCameraMatrixR = calibrateSingle('R', lsBestCombination, paths['individual'], calibration_criteria, squareSize, boardSize, winSize)

            
            # Save matrices to folder
            os.chdir(paths['individual'])
            np.save('distortionCoefficientsR', distortionCoefficientsR)
            np.save('cameraMatrixR', cameraMatrixR)
            np.save('newcameraMatrixR', newCameraMatrixR)
            print('CameraMatrix, distortionCoefficients and newCameraMatrix computed and saved to file...')

            singleRCalibration = False
            

    ## After the steps were completed or right at the beginning, load intrinsics from file
    print('Attempting to load intrinsics from file...')
    try: 
        cameraMatrixR = np.load(paths['individual']+'/cameraMatrixR.npy')
        newCameraMatrixR = np.load(paths['individual']+'/newCameraMatrixR.npy')
        distortionCoefficientsR = np.load(paths['individual']+'/distortionCoefficientsR.npy')
    except:
        print('Error loading camera intrinsics from file. Please restart calibration')
    else: 
        print('Camera R intrinsics: fx={fx:.2f}px, fy={fy:.2f}px, cx={cx:.2f}px, cy={cy:.2f}px'.format(fx=newCameraMatrixR[0][0], fy=newCameraMatrixR[1][1], cx=newCameraMatrixR[0][2], cy=newCameraMatrixR[1][2]))
        print('Camera R distortion coefficients: k1={}, k2={}, p1={}, p2={}, k3={} '.format(distortionCoefficientsR[0][0], distortionCoefficientsR[1][0], distortionCoefficientsR[2][0], distortionCoefficientsR[3][0], distortionCoefficientsR[4][0]))
     
    # Optional: Show result of calibration for cam
    if debuggingMode is True: #Works
                
        # Undistort livestream to provide plausability check whether calibration was successfull
        while True:
            key = cv.waitKey(25)
            if key == 27:
                print('User closed undistorted view.')
                break

            isTrue, frame = capture.read()

            dst = cv.undistort(frame, cameraMatrixR, distortionCoefficientsR, None, newCameraMatrixR)

            cv.imshow('Camera R - Original', frame)
            cv.imshow('Camera R - Undistored', dst)

    print('{:^75s}'.format('_____ Finished computing intrinsics for right camera _____'))

def calibrateStereoSetup(capL, capR, paths, stereoGenImages, stereoTestCombinations, stereoCalibration, debuggingMode):

    print('{:^75s}'.format('_____ Computing stereo parameters _____'))
    
    # If any of the steps to calibration must be completed, enter loop until they finish
    while (stereoGenImages is True) or (stereoTestCombinations is True) or (stereoCalibration is True):
    
        if stereoGenImages is True: #Works 

            generateImagesStereo(capL, capR, paths['stCamL'], paths['stCamR'], numImgsGenerateStereo, calibration_criteria, boardSize)
            print('Images successfully generated and saved to file...')
            stereoGenImages = False

        elif stereoTestCombinations is True: #Works
        
            print('Computing best image combination for stereo setup')
            # Load all image paths and their image points
            dict_pathsPoints = loadImgsandImgPoints('Stereo', boardSize, winSize, calibration_criteria, paths['stCamL'], paths['stCamR']) 

            # Find best combination for all images
            lsBestCombination = testAllImgCombinations('Stereo', dict_pathsPoints, None, paths['stereo'], boardSize)
            
            stereoTestCombinations = False

        elif stereoCalibration is True: #Works

            # Load list with best combination from file
            lsBestCombination = np.load(paths['stereo']+'/bestCombinationStereo.npy')
            print('Best image combination for stereo setup '+str(lsBestCombination))

            # Load intrinsics from file
            newCameraMatrixL = np.load(paths['individual']+'/newCameraMatrixL.npy')
            distortionCoefficientsL = np.load(paths['individual']+'/distortionCoefficientsL.npy')
            newCameraMatrixR = np.load(paths['individual']+'/newCameraMatrixR.npy')
            distortionCoefficientsR = np.load(paths['individual']+'/distortionCoefficientsR.npy')

            # Compute stereo parameters
            calibrateStereo(lsBestCombination, paths, newCameraMatrixL, distortionCoefficientsL, newCameraMatrixR, distortionCoefficientsR)

            stereoCalibration = False
            
    # Otherwise just load the stereo parameters
    try:
        print('Attempting to load stereo parameters from file..')
        Rot = np.load(paths['stereo']+'/rotationVector.npy')
        Trns = np.load(paths['stereo']+'/translationVector.npy')
        print('Successfully loaded stereo parameters from file..')
        print('Translation from camera R to camera L: x={:.2f}cm, y={:.2f}cm, z={:.2f}cm'.format(Trns[0][0], Trns[1][0], Trns[2][0]))
    except:
        print('Could not load stereo parameters from file. Please recalibrate stereo setup..')

    print('{:^75s}'.format('_____ Finished computing stereo parameters _____'))
    
    return Trns

def getRectificationMap(capL, capR, paths, newRectificationMapping, debuggingMode): #works

    if newRectificationMapping is True:
        # Load intrinsics and extrinics from calibration process
        newCameraMatrixL = np.load(paths['individual']+'/newCameraMatrixL.npy')
        distortionCoefficientsL = np.load(paths['individual']+'/distortionCoefficientsL.npy')
        newCameraMatrixR = np.load(paths['individual']+'/newCameraMatrixR.npy')
        distortionCoefficientsR = np.load(paths['individual']+'/distortionCoefficientsR.npy')

        Rot = np.load(paths['stereo']+'/rotationVector.npy')
        Trns = np.load(paths['stereo']+'/translationVector.npy')
        print('Loaded camera matrices...')

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
        print('Saved rectification maps to file...')

    
    # Load from file
    leftMapX = np.load(paths['stereo']+'/rectificationMapCamLX.npy')
    leftMapY = np.load(paths['stereo']+'/rectificationMapCamLY.npy')
    rightMapX = np.load(paths['stereo']+'/rectificationMapCamRX.npy')
    rightMapY = np.load(paths['stereo']+'/rectificationMapCamRY.npy')
    Q = np.load(paths['stereo']+'/reprojectionMatrixQ.npy')
    print('Loaded rectification maps from file...')
    
    if debuggingMode is True: #work

        newCameraMatrixL = np.load(paths['individual']+'/newCameraMatrixL.npy')
        distortionCoefficientsL = np.load(paths['individual']+'/distortionCoefficientsL.npy')
        newCameraMatrixR = np.load(paths['individual']+'/newCameraMatrixR.npy')
        distortionCoefficientsR = np.load(paths['individual']+'/distortionCoefficientsR.npy')
        Trns = np.load(paths['stereo']+'/translationVector.npy')

        # Take an image of a horizontal chessboard for debugging purposes

        while True:

            key = cv.waitKey(25)

            if key == 27:
                print('Program was terminated by user..')
                sys.exit()

            isTrueL, frameL = capL.read()
            isTrueR, frameR = capR.read()
            cv.imshow('Cam L', frameL)
            cv.imshow('Cam R', frameR)
            
            tempL = frameL.copy()
            tempR = frameR.copy()

            grayL = cv.cvtColor(tempL,cv.COLOR_BGR2GRAY)
            grayR = cv.cvtColor(tempR,cv.COLOR_BGR2GRAY)
            found_patternL, corners_aprxL = cv.findChessboardCorners(grayL, boardSize, None)
            found_patternR, corners_aprxR = cv.findChessboardCorners(grayR, boardSize, None)

            if (found_patternL == True) and (found_patternR == True):
                cv.drawChessboardCorners(tempL, boardSize, corners_aprxL, found_patternL)
                cv.drawChessboardCorners(tempR, boardSize, corners_aprxR, found_patternR)
                cv.imshow('Cam L', tempL)
                cv.imshow('Cam R', tempR)

                if key == ord(' '):
                    imgPathL = paths['stereoTesting']+'/camLTesting.png'
                    imgPathR = paths['stereoTesting']+'/camRTesting.png'

                    cv.imwrite(imgPathL,frameL)
                    cv.imwrite(imgPathR,frameR)
                    break

        cv.destroyAllWindows()

        # Load and rectify the image pair
        imgL = cv.imread(paths['stereoTesting']+'/camLTesting.png')
        imgR = cv.imread(paths['stereoTesting']+'/camRTesting.png')

        Left_rectified = cv.remap(imgL,leftMapX,leftMapY, cv.INTER_LINEAR, cv.BORDER_CONSTANT)
        Right_rectified = cv.remap(imgR,rightMapX,rightMapY, cv.INTER_LINEAR, cv.BORDER_CONSTANT)

        grayL = cv.cvtColor(Left_rectified,cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(Right_rectified,cv.COLOR_BGR2GRAY)

        # Settings for chessboard corners
        font = cv.FONT_HERSHEY_PLAIN
        fontScale = 2

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
            cv.cornerSubPix(grayR, cornersR,(3,3),(-1,-1),calibration_criteria)
            cv.cornerSubPix(grayL, cornersL,(3,3),(-1,-1),calibration_criteria)
            imagePointsR.append(cornersR)
            imagePointsL.append(cornersL)
            
            # Get points in 4th row (vertical centre) and display them
            for i in range(0,41,7):
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
                cv.putText(vis_afterRectification, disp_x_text, (x_l + int((x_r+Left_rectified.shape[1]-x_l)/3), y_r -10), font, fontScale, (255, 0, 0), 2, cv.LINE_AA)

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


def initialiseCorrespondence():
        minDisparity = 0 #192
        maxDisparity = 272
        numDisparities = maxDisparity-minDisparity
        blockSize = 3
        disp12MaxDiff = 5
        uniquenessRatio = 15

        left_matcher = cv.StereoSGBM_create(minDisparity = minDisparity,numDisparities = numDisparities,blockSize = blockSize,disp12MaxDiff = disp12MaxDiff, uniquenessRatio = uniquenessRatio)
    
        # left_matcher = cv.StereoBM_create(numDisparities = numDisparities, blockSize=blockSize)
        sigma = 1.5
        lmbda = 8000.0

        right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
        wls_filter = cv.ximgproc.createDisparityWLSFilter(left_matcher)
        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)

        return left_matcher, right_matcher, wls_filter

def getDepth(Left_rectified, Right_rectified, Q, debuggingMode, left_matcher, right_matcher, wls_filter):


    Left_rectified_gray = cv.cvtColor(Left_rectified,cv.COLOR_BGR2GRAY)
    Right_rectified_gray = cv.cvtColor(Right_rectified,cv.COLOR_BGR2GRAY)

    if debuggingMode == True:

        win_left_dispSGBM = 'Left disparity map - StereoSGBM'
        win_filtered_dispSGBM = 'Filtered left disparity map - StereoSGBM'
        cv.namedWindow(win_left_dispSGBM)
        cv.namedWindow(win_filtered_dispSGBM)

        font = cv.FONT_HERSHEY_PLAIN
        fontScale = 2

        print('____________Checking left disparity map & reprojected depth____________')
        left_disp = left_matcher.compute(Left_rectified_gray, Right_rectified_gray).astype(np.float32) / 16
        print('Data type: '+str(left_disp.dtype))
        print('Shape: '+str(left_disp.shape))

        points_3D_LM = cv.reprojectImageTo3D(left_disp, Q)

        print('____________Checking filtered disparity map & reprojected depth____________')
        right_disp = right_matcher.compute(Right_rectified_gray,Left_rectified_gray).astype(np.float32) / 16
        filtered_disp = wls_filter.filter(left_disp, Left_rectified_gray, disparity_map_right=right_disp)
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

        cv.setMouseCallback(win_left_dispSGBM, printCoordinatesdispLM)
        cv.setMouseCallback(win_filtered_dispSGBM, printCoordinatesdispFM)
        
        vis_right = right_matcher.compute(Right_rectified_gray,Left_rectified_gray)
        vis_left = left_matcher.compute(Left_rectified_gray, Right_rectified_gray)
        filtered_disp_forVis = wls_filter.filter(vis_left, Left_rectified_gray, disparity_map_right=vis_right)

        dispForColor = filtered_disp.copy()
        dispForColor = cv.normalize(src=dispForColor, dst=dispForColor, alpha=255, beta=0 , norm_type=cv.NORM_MINMAX)
        disp8 = np.uint8(dispForColor)

        colored = cv.applyColorMap(disp8, cv.COLORMAP_JET)

        left_disp_forVis = left_matcher.compute(Left_rectified_gray, Right_rectified_gray)

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

            cv.imshow(win_left_dispSGBM, left_disp_forVis)
            cv.imshow(win_filtered_dispSGBM, filtered_disp_forVis)
            cv.imshow('Coloured Disparity', colored)

        cv.destroyAllWindows()
    else:
        #print('Computing filtered disparity map with Stereo SGBM')
        left_disp = left_matcher.compute(Left_rectified_gray, Right_rectified_gray).astype(np.float32) / 16
        right_disp = right_matcher.compute(Right_rectified_gray,Left_rectified_gray).astype(np.float32) / 16
        filtered_disp = wls_filter.filter(left_disp, Left_rectified_gray, disparity_map_right=right_disp)
        points_3D = cv.reprojectImageTo3D(filtered_disp, Q)

        dispForColor = filtered_disp.copy()
        dispForColor = cv.normalize(src=dispForColor, dst=dispForColor, alpha=255, beta=0 , norm_type=cv.NORM_MINMAX)
        disp8 = np.uint8(dispForColor)
        colored_disp = cv.applyColorMap(disp8, cv.COLORMAP_JET)


    return points_3D, colored_disp