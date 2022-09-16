import cv2 as cv
import numpy as np
import time
import os
import csv
import sys
from pprint import pprint

# Global parameters used in all functions

numImgsGenerateSingle = 30
numImgsGenerateStereo = 20

calibration_criteria = (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 100, 1e-06)
winSize = (11,11)
squareSize = 2.2 #cm
boardSize = (9,6)

font = cv.FONT_HERSHEY_PLAIN
fontScale = 2

#%% Sub functions

def listdirVisible(path):
    items = os.listdir(path)
    visible = [item for item in items if not item.startswith('.')]
    return visible

def generateImgsSingle(capture, camPosition, pathImgs):
        
    print('.\nStarting image generation for camera {}\n.'.format(camPosition))

    # (Re)set timer
    value_timer = int(1)
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
                gray = cv.cvtColor(cv.imread(imgPath),cv.COLOR_BGR2GRAY)
                found_pattern, corners_aprx = cv.findChessboardCorners(gray, boardSize, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)

                # If all corners were found, locate exact corners of 2D image and store information
                if (found_pattern == True):
                    
                    corners_exct = cv.cornerSubPix(gray, corners_aprx, winSize, (-1,-1), calibration_criteria)
                    
                    # Draw and display the corners
                    img = cv.imread(imgPath)
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
        print('.\nFinished image generation for camera {}\n.'.format(camPosition))
    else:
        print('Generating images aborted...')

def generateImagesStereo(capL, capR, path_cam_L, path_cam_R):
    
    print('.\nStarting image generation for stereo\n.')
    
    # (Re)set timer
    value_timer = int(1)
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
                imgL = cv.imread(imgPathL)
                grayL = cv.cvtColor(imgL,cv.COLOR_BGR2GRAY)
                found_patternL, corners_aprxL = cv.findChessboardCorners(grayL, boardSize, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)
                
                imgR = cv.imread(imgPathR)
                grayR = cv.cvtColor(imgR,cv.COLOR_BGR2GRAY)
                found_patternR, corners_aprxR = cv.findChessboardCorners(grayR, boardSize, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)


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
    
    print('.\nFinished image generation for stereo\n.')



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

# Given a list of image points, compute overall mean reprojection error for this image combination
def getRepError(camPosition, listPoints, listNames, imgSize):

    objp = np.zeros((1, boardSize[0]*boardSize[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:boardSize[0], 0:boardSize[1]].T.reshape(-1, 2)
    objp = objp * squareSize
    objectPoints = []
    imagePoints = [] 

    # Add imagePoints from dictionary to local array
    for point in listPoints:
        objectPoints.append(objp)
        imagePoints.append(point)

    # Necessary to set notation for at least distCoeffs as 5x1. source: page 394
    cameraMatrixTemplate = np.zeros((3, 3))
    distCoefficientsTemplate = np.zeros((5, 1))


    # Calibrate camera
    _, cameraMatrix, distortionCoefficients, rotationR, translationT= \
        cv.calibrateCamera(
            objectPoints,
            imagePoints,
            imgSize,
            cameraMatrixTemplate,
            distCoefficientsTemplate,
            flags = 0,
            criteria = calibration_criteria,
        )


    # Compute overall mean reprojection error for image combination
    # errorVisHeader = ['Camera', 'Image', 'Corner', 'xProjected', 'yProjected', 'xOriginal', 'yOriginal', 'deltaX', 'deltaY', 'RMS error']
    overallMeanRMS = 0
    errorVis = []

    for i in range(len(objectPoints)):
        
        projectedImgPoints, _ = cv.projectPoints(objectPoints[i], rotationR[i], translationT[i], cameraMatrix, distortionCoefficients)

        # For every corner in a chessboard view, calculate RMS error
        sumRMS = 0
        for j in range(len(projectedImgPoints)):
            deltaX = projectedImgPoints[j][0][0] - imagePoints[i][j][0][0]
            deltaY = projectedImgPoints[j][0][1] - imagePoints[i][j][0][1]

            RMS = (deltaX**2+deltaY**2)**(0.5)
            sumRMS += RMS

            imgNumber = listNames[i].split('_')[0]
            imgName = imgNumber+'_'+camPosition

            temp = ['Camera '+camPosition,imgName, j, projectedImgPoints[j][0][0], projectedImgPoints[j][0][1], imagePoints[i][j][0][0], imagePoints[i][j][0][1], deltaX, deltaY, RMS]
            errorVis.append(temp.copy())

        # Calculate mean error per image
        meanRMSImg = sumRMS / len(projectedImgPoints)
        overallMeanRMS += meanRMSImg

    # Calculate mean error for all images
    overallMeanRMS = overallMeanRMS / len(objectPoints)
    

    return overallMeanRMS, errorVis


def determineErrorCurrentLevel(mode, dict_pathsPoints, array_indices, best_result, imgSize, error_history):
    """
    Computes overall mean reprojection error for all provided image points 
    """    
    
    if mode == 'Single':

        ## Extract image points and names for the image combination to be checked in this loop
        
        # Left camera
        if dict_pathsPoints[0][0][0] is not None:
            imgPointsCurComb = [dict_pathsPoints[indx][0][1] for indx in array_indices]
            imgNamesCurComb = [dict_pathsPoints[indx][0][0].split('/')[-1].split('.')[0] for indx in array_indices]
            camPosition = 'L'

        # Right camera
        elif dict_pathsPoints[0][1][0] is not None:
            imgPointsCurComb = [dict_pathsPoints[indx][1][1] for indx in array_indices]
            imgNamesCurComb = [dict_pathsPoints[indx][1][0].split('/')[-1].split('.')[0] for indx in array_indices]
            camPosition = 'R'

        #print('Checking {indices}, aka {names}'.format(indices=array_indices, names=imgNamesCurComb))


        try:
            # Check error for current level
            rep_error, _ = getRepError(camPosition, imgPointsCurComb, imgNamesCurComb, imgSize)

            # Compare error for current level with current minimum error for current level (from previous computations)
            num_imgs_local = len(imgPointsCurComb)
            oldMinError = best_result[num_imgs_local][0][1]
            
            if oldMinError > rep_error:
                best_result[num_imgs_local][0][1] = rep_error
                best_result[num_imgs_local][1][1] = imgNamesCurComb.copy()
                print('({numImgs}) Images: {names}. Re-projection error: {error}______ New minimum ______'.format(numImgs=num_imgs_local, names=str(imgNamesCurComb), error=str(rep_error)))
            else:
                print('({numImgs}) Images: {names}. Re-projection error: {error}'.format(numImgs=num_imgs_local, names=str(imgNamesCurComb), error=str(rep_error)))

        except:
            rep_error = 1000
            print('__Computation error for combination {names}'.format(names=str(imgNamesCurComb)))
            error_history.append(imgNamesCurComb.copy())

    elif mode == 'Stereo':

        ## Extract image points and names for the image combination to be checked in this loop
        imgPointsCurCombL = [dict_pathsPoints[indx][0][1] for indx in array_indices]
        imgPointsCurCombR = [dict_pathsPoints[indx][1][1] for indx in array_indices]

        imgNamesCurComb = [dict_pathsPoints[indx][0][0].split('/')[-1].split('.')[0] for indx in array_indices]

        try:
            # Compute mean reprojection error for two images of the same view
            rep_errorL, _ = getRepError('L', imgPointsCurCombL, imgNamesCurComb, imgSize)
            rep_errorR, _ = getRepError('R', imgPointsCurCombR, imgNamesCurComb, imgSize)
            rep_error_res = 0.5*(rep_errorL+rep_errorR)
            

            # Compare error for current level with current minimum error for current level (from previous computations)
            num_imgs_local = len(imgPointsCurCombL)
            oldMinError = best_result[num_imgs_local][0][1]

            if oldMinError > rep_error_res:
                best_result[num_imgs_local][0][1] = rep_error_res
                best_result[num_imgs_local][1][1] = imgNamesCurComb.copy()
                print('({numImgs}) Images: {indices}. MEAN Error: {error}______ New minimum ______'.format(numImgs=num_imgs_local, indices=str(imgNamesCurComb), error=str(rep_error_res)))
            else:
                print('({numImgs}) Images: {indices}. MEAN Error: {error}'.format(numImgs=num_imgs_local, indices=str(imgNamesCurComb), error=str(rep_error_res)))

        except:
            rep_error = 1000
            print('__Computation error for combination {combination}'.format(combination=array_indices))
            error_history.append(imgNamesCurComb.copy())

        rep_error = rep_error_res

    return rep_error, best_result, error_history

# Given any dictionary with image paths and image points, return rep error
def findBestCombination(mode, dict_pathsPoints, array_indices, num_imgs_available, bestCombGlobal, lsHistory, best_result, imgSize, error_history):
  
  error_current_level = 1000 # threshold
  bestCombLocal = []
  
  for j in range(num_imgs_available):

    # Check if current combination of images was tested before
    testedBefore, bestCombGlobal, lsHistory = checkArray(bestCombGlobal, j, lsHistory)
    if testedBefore:
        continue
    else:
        # Determine rep error for current combination
        rep_error, best_result, error_history = determineErrorCurrentLevel(mode, dict_pathsPoints, bestCombGlobal, best_result, imgSize, error_history)
                    
        # If this combination results in high errors, check a different image
        if rep_error > error_current_level:
            bestCombGlobal.remove(j)             
            continue
        else:
            error_current_level = rep_error
            bestCombLocal = bestCombGlobal.copy()
            bestCombGlobal.remove(j)

  return bestCombLocal, lsHistory, best_result, error_history


def loadImgsandImgPoints(mode, pathImgsL, pathImgsR):
    """ 
    Returns dictionary with paths to images and their image points
    """       
    # Define dictionary to store all paths and image points
    dict_pathsPoints = {}
    i = 0

    # Load images for left camera
    if (mode == 'Single') and (pathImgsL is not None):

        # Remove unwanted files from array
        paths_images = listdirVisible(pathImgsL)

        # Compute image points for all images in directory
        for path in paths_images:

            imgPath = pathImgsL+'/'+path
            print('Accessing image: '+imgPath)
            gray = cv.cvtColor(cv.imread(imgPath),cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, boardSize, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)

            # Check if all images have the same size
            if i == 0:
                imgSize = gray.shape[::-1]
            else:
                if gray.shape[::-1] != imgSize:
                    print('All images must have the same size!')
                    sys.exit()
            
            # Refine corner search to subpixel accuracy if all corners were found and save to dictionary
            if ret is True:
                cv.cornerSubPix(gray, corners, winSize, (-1,-1), calibration_criteria)
                dict_pathsPoints[i] = [[imgPath, corners], [None, None]]
                i+=1

            else: 
                continue

    # Load images for right camera
    elif (mode == 'Single') and (pathImgsR is not None):
       
       # Remove unwanted files from directory
        paths_images = listdirVisible(pathImgsR)
        
        for path in paths_images:

            imgPath = pathImgsR+'/'+path
            
            print('Accessing image: '+imgPath)
            gray = cv.cvtColor(cv.imread(imgPath),cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, boardSize, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)
            
            # Refine corner search to subpixel accuracy if some corners were found and save to dictionary
            if ret is True:
                cv.cornerSubPix(gray, corners, winSize, (-1,-1), calibration_criteria)
                dict_pathsPoints[i] = [[None, None], [imgPath, corners]]
                i+=1

            else: 
                continue

    # Load images for both cameras
    elif mode == 'Stereo':

        # Remove unwanted files from directory
        paths_images_L = listdirVisible(pathImgsL)
        paths_images_R = listdirVisible(pathImgsR)        
    
        for pathL in paths_images_L:
            

            # Get left image and find path to the same image for right camera
            pathR = pathL.split('_')[0]+'_R.png'

            # Compute image points for the two images
            imgPathL = pathImgsL+'/'+pathL
            imgPathR = pathImgsR+'/'+pathR

            print('Accessing image: '+imgPathL)
            grayL = cv.cvtColor(cv.imread(imgPathL),cv.COLOR_BGR2GRAY)
            retL, cornersL = cv.findChessboardCorners(grayL, boardSize, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)

            print('Accessing image: '+imgPathR)
            grayR = cv.cvtColor(cv.imread(imgPathR),cv.COLOR_BGR2GRAY)
            retR, cornersR = cv.findChessboardCorners(grayR, boardSize, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)
            
            # Refine corner search to subpixel accuracy if all corners were found and save to dictionary
            if (retL is True) and (retR is True):
                cv.cornerSubPix(grayL, cornersL, winSize, (-1,-1), calibration_criteria)
                cv.cornerSubPix(grayR, cornersR, winSize, (-1,-1), calibration_criteria)

                dict_pathsPoints[i] = [[imgPathL, cornersL], [imgPathR, cornersR]]
                i+=1

            else: 
                continue
 
    print('Finished loading images points...')
    
    return dict_pathsPoints  


def testAllImgCombinations(mode, dict_pathsPoints, camPosition, path_calibration):
    """ Given a dictionary with paths and image points, save dictionary with best combinations for all numbers of images to file.
    Return list containing image names that result in lowest reprojection error

    Keyword arguments:
    mode -- 'Single' or 'Stereo'
    dict_pathsPoints -- Dictionary containing image paths and points [[imgPathL, cornersL], [imgPathR, cornersR]]
    camPosition -- 'L' or 'R' or None for Stereo
    """   
    
    num_imgs_base = 5
    num_imgs_available = len(dict_pathsPoints.keys())
    num_levels = num_imgs_available - num_imgs_base # 

    # Setup dictionary that stores lowest reprojection error for all combination sizes
    best_result = {}
    for level in range(num_imgs_base,num_imgs_available+1):
        best_result[level] = [['Error', 1000],['Images', []]]

    lsHistory = [] # List saving all tested image combinations to avoid double checks
    error_history = [] # Saves image combinations with errors (to identify images that are not good for calibration)
    error_L1 = 2000 # Threshold for first error check (any value higher than the expected found max)

    # Get image size for the use in subfunctions
    if mode == 'Single' and camPosition == 'R':
        pathTemp = dict_pathsPoints[5][1][0]
    elif mode == 'Single' and camPosition == 'L':
        pathTemp = dict_pathsPoints[5][0][0]
    else:
        pathTemp = dict_pathsPoints[5][0][0]

    imgSize = cv.cvtColor(cv.imread(pathTemp),cv.COLOR_BGR2GRAY).shape[::-1]

    # For a base number of images, determine the grouped sequence of indices that leads to lowest error
    print('Testing base combination of images...')
    for i in range(num_imgs_available+1):

        # Determine base range for analyis
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

        # Get rep error for current combination
        rep_error, best_result, error_history = determineErrorCurrentLevel(mode, dict_pathsPoints, array_indices, best_result, imgSize, error_history)
        
        # Determine whether current combination produces new rep minimum
        if rep_error > error_L1:
            continue
        else:
            minL1 = array_indices.copy()
            error_L1 = rep_error

    ## Using the base combination as image sequence that produced lowest rep error as starting point, check which base number of non sequence images have lowest rep error
    bestCombGlobal = minL1.copy()
    print('Best combination base: {combination}'.format(combination=minL1))

    for n in range(num_imgs_base):
        bestCombGlobal, lsHistory, best_result, error_history = findBestCombination(mode, dict_pathsPoints, array_indices, num_imgs_available, bestCombGlobal, lsHistory, best_result, imgSize, error_history)

    # Keep info for best base combination
    minErrorTemp = best_result[num_imgs_base][0][1]
    minErrorCombTemp = best_result[num_imgs_base][1][1]

    # Reset best_results
    best_result = {}
    for level in range(num_imgs_base,num_imgs_available+1):
        best_result[level] = [['Error', 1000],['Images', []]]
    
    # Save previously calculated best result for base level
    best_result[num_imgs_base][0][1] = minErrorTemp
    best_result[num_imgs_base][1][1] = minErrorCombTemp
    
    # Use best base as starting point to find overall best combination
    new_base = bestCombGlobal[num_imgs_base:].copy()
    bestCombGlobal = new_base
    for n in range(num_levels):
        bestCombGlobal, lsHistory, best_result, error_history = findBestCombination(mode, dict_pathsPoints, array_indices, num_imgs_available, bestCombGlobal, lsHistory, best_result, imgSize, error_history)

    # For all results in best_result find the combination with the lowest reprojection error
    lsBestCombination = []
    values = list(best_result.values())
    lowestError = 20000 # random treshold
    for value in values:
        error = value[0][1]
        if error < lowestError:
            lowestError = error
            lsBestCombination = value[1][1]

    '''
    Check if this is necessary
    '''
    # Plausability check: Get indices of img names that were printed as best combination
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
    all_keys = list(best_result.keys())

    if mode == 'Single':
        os.chdir(path_calibration)
        np.save('best_combination_cam'+camPosition, lsBestCombination)

        with open('reperrors_all_combinations_cam'+camPosition+'.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(header)
            
            for key in all_keys:
                valError = best_result[key][0][1]
                images = best_result[key][1][1]
                data = [key, valError, images]
               
                writer.writerow(data)
        print('Best combination for camera {camPosition} saved to file.'.format(camPosition=camPosition))

        with open('error_history_cam'+camPosition+'.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            for element in error_history:
                writer.writerow(element)
        print('Error history for camera {camPosition} saved to file.'.format(camPosition=camPosition))

        print('.\nFinished computing best image combination for camera {} .\n'.format(camPosition))


    elif mode == 'Stereo':
        os.chdir(path_calibration)
        np.save('best_combination_stereo', lsBestCombination)

        with open('reperrors_all_combinations_stereo.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            writer.writerow(header)
            
            for key in all_keys:
                valError = best_result[key][0][1]
                images = best_result[key][1][1]
                data = [key, valError, images]
                writer.writerow(data)
        print('Best combinations for stereo setup saved to file.')
    
        with open('error_history_stereo.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            for element in error_history:
                writer.writerow(element)
        print('Error history for for stereo setup saved to file.')

        print('.\nFinished computing best image combination for stereo setup {} .\n'.format(camPosition))

   
    
    return lsBestCombination

def calibrateSingle(camPosition, lsBestCombination, path_calibration, squareSize):
    
    print('.\nStarting calibration of camera {}\n.'.format(camPosition))

    # Setup matrices for object points and image points
    objp = np.zeros((1, boardSize[0]*boardSize[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:boardSize[0], 0:boardSize[1]].T.reshape(-1, 2)
    objp = objp * squareSize
    objectPoints = []
    imagePoints = []

    # Get full paths to the images listed in 'lsBestCombination'
    path_cam = path_calibration+'/cam'+camPosition
    imgPaths = [path_cam+'/'+imgNumber.split('_')[0]+'_'+camPosition+'.png' for imgNumber in lsBestCombination]

    imgSize = cv.cvtColor(cv.imread(imgPaths[1]),cv.COLOR_BGR2GRAY).shape[::-1] # Get size of any of the available images (all the same size, checked before!)

    # Compute imagePoints and objectPoints for the respective images
    for i in range(len(imgPaths)):

        # Load image
        imgPath = imgPaths[i]
        print('Accessing image: '+imgPath)
        img = cv.imread(imgPath)

        # Find chessboard corners
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, boardSize, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)

        # If all corners were found extend objectPoints matrix and add refined image points to list
        if ret is True:
            objectPoints.append(objp)
            cv.cornerSubPix(gray, corners, winSize, (-1,-1), calibration_criteria)
            imagePoints.append(corners)

    cameraMatrixTemplate = np.zeros((3, 3))
    distCoefficientsTemplate = np.zeros((5, 1))

    # Niklas

    rotationMatTemplate = [np.zeros((1, 1, 3)) for i in range(len(imgPaths))]
    translationVecTemplate = [np.zeros((1, 1, 3)) for i in range(len(imgPaths))]

    # Niklas

    # Calibrate camera
    _, cameraMatrix, distortionCoefficients, rotationR, translationT = \
        cv.calibrateCamera(
            objectPoints,
            imagePoints,
            imgSize,
            cameraMatrixTemplate,
            distCoefficientsTemplate,
            rotationMatTemplate,
            translationVecTemplate,
            flags = 0,
            criteria = calibration_criteria,
        )


    # Optimise camera matrix (optimum depending on whether original pixels should be kept or not)
    h,  w = gray.shape[:2]
    newCameraMatrix, _ = cv.getOptimalNewCameraMatrix(cameraMatrix, distortionCoefficients, (w,h), 1, (w,h))


    # Compute overall mean reprojection error for this image combination to (1) check plausability with previous error and (2) visualise
    overallMeanRMS, errorVis = getRepError(camPosition,imagePoints,lsBestCombination,imgSize)
    errorVisHeader = ['Camera', 'Image', 'Corner', 'xProjected', 'yProjected', 'xOriginal', 'yOriginal', 'deltaX', 'deltaY', 'RMS error']
    
    print('Plausability check: Lowest reprojection error for combination {}: {:.5f} pixels '.format(lsBestCombination, overallMeanRMS))
    
    #Output error vis to csv
    with open(path_calibration+'/reperrors_best_combination_cam'+camPosition+'.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(errorVisHeader)
        for element in errorVis:
            writer.writerow(element)

    print('.\nFinished calibration of camera {}\n.'.format(camPosition))
    return cameraMatrix, distortionCoefficients, newCameraMatrix

    
def calibrateStereo(lsBestCombinationtereo, paths, newCameraMatrixL, distortionCoefficientsL, newCameraMatrixR, distortionCoefficientsR):
    
    print('Starting stereo calibration')

    # Setup matrices for objectpoints and imagepoints
    objp = np.zeros((1, boardSize[0]*boardSize[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:boardSize[0], 0:boardSize[1]].T.reshape(-1, 2)
    objp = objp * squareSize
    objectPoints = []
    imagePointsL = [] 
    imagePointsR = [] 

    # Get full paths to the images listed in 'lsBestCombination'
    imgNumbers = lsBestCombinationtereo.copy()
    for a in range(len(imgNumbers)):
        imgNumbers[a] = imgNumbers[a].split('_')[0]

    path_cam_L = paths['stereo']+'/camL'
    path_cam_R = paths['stereo']+'/camR'

    # Get image paths to left and right image given the name of the left image
    path_images_L = []
    path_images_R = []
    for imgNum in imgNumbers:
        path_images_L.append(path_cam_L+'/'+imgNum+'_L.png')
        path_images_R.append(path_cam_R+'/'+imgNum+'_R.png')

    imgSize = cv.cvtColor(cv.imread(path_images_L[1]),cv.COLOR_BGR2GRAY).shape[::-1] # Get size of any of the available images (all the same size, checked before!)

    
    # Get object and image points for every image    
    for i in range(len(path_images_L)):

        # Load images
        imgPathL = path_images_L[i]
        imgPathR = path_images_R[i]
        print('Accessing image: '+imgPathR)
        print('Accessing image: '+imgPathL)
        imgR = cv.imread(imgPathR)
        imgL = cv.imread(imgPathL)
        grayR = cv.cvtColor(imgR,cv.COLOR_BGR2GRAY)
        grayL = cv.cvtColor(imgL,cv.COLOR_BGR2GRAY)

        # Find chessboard corners
        retR, cornersR = cv.findChessboardCorners(grayR, boardSize, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)
        retL, cornersL = cv.findChessboardCorners(grayL, boardSize, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)

        # If all corners were found extend objectPoints matrix and add refined image points to list
        if retR is True and retL is True:
            objectPoints.append(objp)
            cv.cornerSubPix(grayR, cornersR, winSize, (-1,-1),calibration_criteria)
            cv.cornerSubPix(grayL, cornersL, winSize, (-1,-1),calibration_criteria)
            imagePointsR.append(cornersR)
            imagePointsL.append(cornersL)


    # Calibrate stereo
    flags = cv.CALIB_FIX_INTRINSIC

    _, _, _, _, _, stereoRotation, stereoTranslation, _, _ = cv.stereoCalibrate(objectPoints, imagePointsL, imagePointsR, newCameraMatrixL, distortionCoefficientsL, newCameraMatrixR, distortionCoefficientsR, imgSize, calibration_criteria, flags)

    # Save stereo parameters
    np.save(paths['stereo']+'/rotationVector', stereoRotation)
    np.save(paths['stereo']+'/translationVector', stereoTranslation)
    print('Saved stereo matrices T and R to file...')

    # Compute overall mean reprojection error for this image combination to (1) check plausability with previous error and (2) visualise
    overallMeanRMSL, errorVisL = getRepError('L',imagePointsL,lsBestCombinationtereo,imgSize)
    overallMeanRMSR, errorVisR = getRepError('R',imagePointsR,lsBestCombinationtereo,imgSize)
    medianError = 0.5*(overallMeanRMSL + overallMeanRMSR)

    errorVis = np.concatenate((errorVisL, errorVisR))
    errorVisHeader = ['Camera', 'Image', 'Corner', 'xProjected', 'yProjected', 'xOriginal', 'yOriginal', 'deltaX', 'deltaY', 'RMS error']
    
    # Output error vis to csv
    with open(paths['stereo']+'/reperrors_best_combination_stereo.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(errorVisHeader)
        for element in errorVis:
            writer.writerow(element)

    print('Finished stereo calibration')

    

#%% Main functions

def getIntrinsicsLeftCamera(capture, paths, singleLGenImages, singleLTestCombinations, singleLCalibration, debuggingMode):
    
    if singleLGenImages is True: #Works
        generateImgsSingle(capture, 'L', paths['indCamL'])

    elif singleLTestCombinations is True: #Works
    
        print('Computing best image combination for camera L')
        # Create a dictionary with paths to all images and their respective image points 
        dict_pathsPoints = loadImgsandImgPoints('Single', paths['indCamL'], None)

        # Find best combination for all images
        lsBestCombination = testAllImgCombinations('Single', dict_pathsPoints, 'L', paths['individual'])

    elif singleLCalibration == True: # Works

        # Load list with best combination from file
        try:
            lsBestCombination = np.load(paths['individual']+'/best_combination_camL.npy')
        except:
            print('Failed to load best image combination. Please compute best combination first')
        else:
            # Compute intrinsics
            cameraMatrixL, distortionCoefficientsL, newCameraMatrixL = calibrateSingle('L', lsBestCombination, paths['individual'], squareSize)

            #Save matrices to folder
            os.chdir(paths['individual'])
            np.save('distortionCoefficientsL', distortionCoefficientsL)
            np.save('cameraMatrixL', cameraMatrixL)
            np.save('newcameraMatrixL', newCameraMatrixL)
            # Changed below from newCam to cam
            print('Camera L intrinsics: fx={fx:.2f}px, fy={fy:.2f}px, cx={cx:.2f}px, cy={cy:.2f}px'.format(fx=cameraMatrixL[0][0], fy=cameraMatrixL[1][1], cx=newCameraMatrixL[0][2], cy=newCameraMatrixL[1][2]))
            print('Camera L distortion coefficients: k1={:.5f}, k2={:.5f}, p1={:.5f}, p2={:.5f}, k3={:.5f} '.format(distortionCoefficientsL[0][0], distortionCoefficientsL[1][0], distortionCoefficientsL[2][0], distortionCoefficientsL[3][0], distortionCoefficientsL[4][0]))
            print('CameraMatrixL, distortionCoefficientsL and newCameraMatrixL computed and saved to file...')

    else:
            print('Attempting to load cam L intrinsics from file...')
            try: 
                cameraMatrixL = np.load(paths['individual']+'/cameraMatrixL.npy')
                newCameraMatrixL = np.load(paths['individual']+'/newCameraMatrixL.npy')
                distortionCoefficientsL = np.load(paths['individual']+'/distortionCoefficientsL.npy')
            except:
                print('Error loading camera intrinsics from file. Please restart calibration')
            else:   
                print('Camera L intrinsics: fx={fx:.2f}px, fy={fy:.2f}px, cx={cx:.2f}px, cy={cy:.2f}px'.format(fx=newCameraMatrixL[0][0], fy=newCameraMatrixL[1][1], cx=newCameraMatrixL[0][2], cy=newCameraMatrixL[1][2]))
                print('Camera L distortion coefficients: k1={:.5f}, k2={:.5f}, p1={:.5f}, p2={:.5f}, k3={:.5f} '.format(distortionCoefficientsL[0][0], distortionCoefficientsL[1][0], distortionCoefficientsL[2][0], distortionCoefficientsL[3][0], distortionCoefficientsL[4][0]))

    # Try to run debugging mode
    if debuggingMode is True: #Works
        print('Starting debugging for camera L')
        print('Attempting to load cam L intrinsics from file...')
        try: 
            cameraMatrixL = np.load(paths['individual']+'/cameraMatrixL.npy')
            newCameraMatrixL = np.load(paths['individual']+'/newCameraMatrixL.npy')
            distortionCoefficientsL = np.load(paths['individual']+'/distortionCoefficientsL.npy')
        except:
            print('Error loading camera intrinsics from file. Please restart calibration')
        else:   
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


def getIntrinsicsRightCamera(capture, paths, singleRGenImages, singleRTestCombinations, singleRCalibration, debuggingMode):
    
    if singleRGenImages is True: #Works

        generateImgsSingle(capture, 'R', paths['indCamR'])

    elif singleRTestCombinations is True: #Works
    
        print('Computing best image combination for camera R')
        # Create a dictionary with paths to all images and their respective image points 
        dict_pathsPoints = loadImgsandImgPoints('Single', None, paths['indCamR'])

        # Find best combination for all images
        lsBestCombination = testAllImgCombinations('Single', dict_pathsPoints, 'R', paths['individual'])


    elif singleRCalibration == True: #Works

        # Load list with best combination from file
        try:
            lsBestCombination = np.load(paths['individual']+'/best_combination_camR.npy')
        except:
            print('Failed to load best image combination. Please compute best combination first')

        else:
            # Compute intrinsics
            cameraMatrixR, distortionCoefficientsR, newCameraMatrixR = calibrateSingle('R', lsBestCombination, paths['individual'], squareSize)

            # Save matrices to folder
            os.chdir(paths['individual'])
            np.save('distortionCoefficientsR', distortionCoefficientsR)
            np.save('cameraMatrixR', cameraMatrixR)
            np.save('newcameraMatrixR', newCameraMatrixR)

            print('Camera R intrinsics: fx={fx:.2f}px, fy={fy:.2f}px, cx={cx:.2f}px, cy={cy:.2f}px'.format(fx=newCameraMatrixR[0][0], fy=newCameraMatrixR[1][1], cx=newCameraMatrixR[0][2], cy=newCameraMatrixR[1][2]))
            print('Camera R distortion coefficients: k1={:.5f}, k2={:.5f}, p1={:.5f}, p2={:.5f}, k3={:.5f} '.format(distortionCoefficientsR[0][0], distortionCoefficientsR[1][0], distortionCoefficientsR[2][0], distortionCoefficientsR[3][0], distortionCoefficientsR[4][0]))
            print('CameraMatrix, distortionCoefficients and newCameraMatrix computed and saved to file...')

    else:

        print('Attempting to load cam R intrinsics from file...')
        try: 
            cameraMatrixR = np.load(paths['individual']+'/cameraMatrixR.npy')
            newCameraMatrixR = np.load(paths['individual']+'/newCameraMatrixR.npy')
            distortionCoefficientsR = np.load(paths['individual']+'/distortionCoefficientsR.npy')
        except:
            print('Error loading camera intrinsics from file. Please restart calibration')
        else: 
            print('Camera R intrinsics: fx={fx:.2f}px, fy={fy:.2f}px, cx={cx:.2f}px, cy={cy:.2f}px'.format(fx=newCameraMatrixR[0][0], fy=newCameraMatrixR[1][1], cx=newCameraMatrixR[0][2], cy=newCameraMatrixR[1][2]))
            print('Camera R distortion coefficients: k1={:.5f}, k2={:.5f}, p1={:.5f}, p2={:.5f}, k3={:.5f} '.format(distortionCoefficientsR[0][0], distortionCoefficientsR[1][0], distortionCoefficientsR[2][0], distortionCoefficientsR[3][0], distortionCoefficientsR[4][0]))

    # Try to run debugging mode

    if debuggingMode is True: #Works
        
        print('Attempting to load cam R intrinsics from file...')
        try: 
            cameraMatrixR = np.load(paths['individual']+'/cameraMatrixR.npy')
            newCameraMatrixR = np.load(paths['individual']+'/newCameraMatrixR.npy')
            distortionCoefficientsR = np.load(paths['individual']+'/distortionCoefficientsR.npy')
        except:
            print('Error loading camera intrinsics from file. Please restart calibration')
        else: 
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


def calibrateStereoSetup(capL, capR, paths, stereoGenImages, stereoTestCombinations, stereoCalibration, debuggingMode):
    
    stereoTranslation = 0

    if stereoGenImages is True: #Works 

        generateImagesStereo(capL, capR, paths['stCamL'], paths['stCamR'])

    elif stereoTestCombinations is True: #Works
    
        print('Computing best image combination for stereo setup')
        # Load all image paths and their image points
        dict_pathsPoints = loadImgsandImgPoints('Stereo', paths['stCamL'], paths['stCamR']) 

        # Find best combination for all images
        lsBestCombination = testAllImgCombinations('Stereo', dict_pathsPoints, None, paths['stereo'])
        

    elif stereoCalibration is True: #Works

        # Load list with best combination from file
        try:
            lsBestCombination = np.load(paths['stereo']+'/best_combination_stereo.npy')

            # Load intrinsics from file
            newCameraMatrixL = np.load(paths['individual']+'/newCameraMatrixL.npy')
            distortionCoefficientsL = np.load(paths['individual']+'/distortionCoefficientsL.npy')
            newCameraMatrixR = np.load(paths['individual']+'/newCameraMatrixR.npy') 
            distortionCoefficientsR = np.load(paths['individual']+'/distortionCoefficientsR.npy')

        
        except: 
            print('Failed to load best image combination or camera intrinsics. Please restart calibration')

        else: 
            # Compute stereo parameters
            calibrateStereo(lsBestCombination, paths, newCameraMatrixL, distortionCoefficientsL, newCameraMatrixR, distortionCoefficientsR)
            stereoTranslation = np.load(paths['stereo']+'/translationVector.npy')
            print('Translation from camera R to camera L: x={:.5f}cm, y={:.5f}cm, z={:.5f}cm'.format(stereoTranslation[0][0], stereoTranslation[1][0], stereoTranslation[2][0]))
   
    else:
        print('Attempting to load stereo parameters from file..')
        try:
            stereoRotation = np.load(paths['stereo']+'/rotationVector.npy')
            stereoTranslation = np.load(paths['stereo']+'/translationVector.npy')
            print('Translation from camera R to camera L: x={:.2f}cm, y={:.2f}cm, z={:.2f}cm'.format(stereoTranslation[0][0], stereoTranslation[1][0], stereoTranslation[2][0]))
        except:
            print('Could not load stereo parameters from file. Please recalibrate stereo setup..')
    

    return stereoTranslation

def getRectificationMap(capL, capR, paths, newRectificationMapping, debuggingMode): #works

    if newRectificationMapping is True:

        # Load intrinsics and extrinics from calibration process
        try: 
            newCameraMatrixL = np.load(paths['individual']+'/newCameraMatrixL.npy')
            distortionCoefficientsL = np.load(paths['individual']+'/distortionCoefficientsL.npy')
            newCameraMatrixR = np.load(paths['individual']+'/newCameraMatrixR.npy')
            distortionCoefficientsR = np.load(paths['individual']+'/distortionCoefficientsR.npy')
        except: 
            print('Please compute intrinics for both cameras before starting rectification.')
        else:
            try:
                # Load stereo parameters
                stereoRotation = np.load(paths['stereo']+'/rotationVector.npy')
                stereoTranslation = np.load(paths['stereo']+'/translationVector.npy')

                # Load best stereo combination 
                lsBestCombination = np.load(paths['stereo']+'/best_combination_stereo.npy')
            except:
                print('Please compute stereo parameters before starting rectification.')
            else:
        
                # Load images for mapping
                imgNum = lsBestCombination[1].split('_')[0] # get any image from best combination (proven to be good for calibration purposes)
                
                path_img_L = paths['stereo']+'/camL/'+imgNum+'_L.png'
                path_img_R = paths['stereo']+'/camR/'+imgNum+'_R.png'
                imgL = cv.imread(path_img_L)
                imgR = cv.imread(path_img_R)
                gray = cv.cvtColor(imgL,cv.COLOR_BGR2GRAY)
    
                imgSize = gray.shape[::-1]

                # Compute rotation and projection matrices
                R_L, R_R, proj_mat_l, proj_mat_r, Q, _, _= cv.stereoRectify(newCameraMatrixL, distortionCoefficientsL, newCameraMatrixR, distortionCoefficientsR, imgSize, stereoRotation, stereoTranslation, flags=cv.CALIB_ZERO_DISPARITY , alpha=1)

                # Compute rectified camera matrices
                rectifiedCameraMatrixL = proj_mat_l[:,[0, 1, 2]]
                rectifiedCameraMatrixR = proj_mat_r[:,[0, 1, 2]]

                # Compute rectification maps
                leftMapX, leftMapY = cv.initUndistortRectifyMap(newCameraMatrixL, distortionCoefficientsL, R_L, proj_mat_l, imgSize, cv.CV_32FC1)
                rightMapX, rightMapY = cv.initUndistortRectifyMap(newCameraMatrixR, distortionCoefficientsR, R_R, proj_mat_r, imgSize, cv.CV_32FC1)    

                # Save rectification maps and Q to file
                np.save(paths['stereo']+'/rectificationMapCamLX', leftMapX)
                np.save(paths['stereo']+'/rectificationMapCamLY', leftMapY)
                np.save(paths['stereo']+'/rectificationMapCamRX', rightMapX)
                np.save(paths['stereo']+'/rectificationMapCamRY', rightMapY)
                np.save(paths['stereo']+'/reprojectionMatrixQ', Q)
                np.save(paths['stereo']+'/rectifiedCameraMatrixL', rectifiedCameraMatrixL)
                np.save(paths['stereo']+'/rectifiedCameraMatrixR', rectifiedCameraMatrixR)
                print('Saved rectification maps to file...')

    else:
        print('Attempting to load rectification maps from file...')
        try:
            # Load from file
            leftMapX = np.load(paths['stereo']+'/rectificationMapCamLX.npy')
            leftMapY = np.load(paths['stereo']+'/rectificationMapCamLY.npy')
            rightMapX = np.load(paths['stereo']+'/rectificationMapCamRX.npy')
            rightMapY = np.load(paths['stereo']+'/rectificationMapCamRY.npy')
            rectifiedCameraMatrixL = np.load(paths['stereo']+'/rectifiedCameraMatrixL.npy')
            rectifiedCameraMatrixR = np.load(paths['stereo']+'/rectifiedCameraMatrixR.npy')
            Q = np.load(paths['stereo']+'/reprojectionMatrixQ.npy')
            print('Loaded rectification maps from file...')
        except:
            print('Please initiate new rectification mapping.')
        
    
    if debuggingMode is True: #work
        print('Starting debugging mode for stereo setup')

        print('Attempting to load camera intrinsics and rectification maps from file...')
        try:
            # Load from file
            leftMapX = np.load(paths['stereo']+'/rectificationMapCamLX.npy')
            leftMapY = np.load(paths['stereo']+'/rectificationMapCamLY.npy')
            rightMapX = np.load(paths['stereo']+'/rectificationMapCamRX.npy')
            rightMapY = np.load(paths['stereo']+'/rectificationMapCamRY.npy')
            rectifiedCameraMatrixL = np.load(paths['stereo']+'/rectifiedCameraMatrixL.npy')
            rectifiedCameraMatrixR = np.load(paths['stereo']+'/rectifiedCameraMatrixR.npy')
            stereoTranslation = np.load(paths['stereo']+'/translationVector.npy')
            Q = np.load(paths['stereo']+'/reprojectionMatrixQ.npy')
            print('Loaded rectification maps from file...')
        except:
            print('Please initiate new rectification mapping.')

        # Take an image of a horizontal chessboard for debugging purposes
        
        while True:
            key = cv.waitKey(25)

            if key == 27:
                print('Program was terminated by user..')
                sys.exit()

            isTrueL, frameL = capL.read()
            isTrueR, frameR = capR.read()
            cv.imshow('Camera L', frameL)
            cv.imshow('Camera R', frameR)
            
            tempL = frameL.copy()
            tempR = frameR.copy()

            grayL = cv.cvtColor(tempL,cv.COLOR_BGR2GRAY)
            grayR = cv.cvtColor(tempR,cv.COLOR_BGR2GRAY)
            imgSize = grayL.shape[::-1]
            retL, cornersL = cv.findChessboardCorners(grayL, boardSize, None)
            retR, cornersR = cv.findChessboardCorners(grayR, boardSize, None)

            if (retL == True) and (retR == True):
                cv.drawChessboardCorners(tempL, boardSize, cornersL, retL)
                cv.drawChessboardCorners(tempR, boardSize, cornersR, retR)
                cv.putText(tempL, 'Pattern found. Press SPACE key to check rectification',(imgSize[0]//3, 40), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)
                cv.putText(tempR, 'Pattern found. Press SPACE key to check rectification',(imgSize[0]//3, 40), font, fontScale, (0, 255, 255), 2, cv.LINE_AA)

                cv.imshow('Camera L', tempL)
                cv.imshow('Camera R', tempR)

                if key == ord(' '):
                    imgPathL = paths['stereoTesting']+'/camLTesting.png'
                    imgPathR = paths['stereoTesting']+'/camRTesting.png'

                    cv.imwrite(imgPathL,frameL)
                    cv.imwrite(imgPathR,frameR)
                    break

        cv.destroyAllWindows()
        print('Press any key to close this view')

        # Load and rectify the image pair
        imgL = cv.imread(paths['stereoTesting']+'/camLTesting.png')
        imgR = cv.imread(paths['stereoTesting']+'/camRTesting.png')

        Left_rectified = cv.remap(imgL,leftMapX,leftMapY, cv.INTER_LINEAR, cv.BORDER_CONSTANT)
        Right_rectified = cv.remap(imgR,rightMapX,rightMapY, cv.INTER_LINEAR, cv.BORDER_CONSTANT)

        grayL = cv.cvtColor(imgL,cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(imgR,cv.COLOR_BGR2GRAY)      
        grayL_rect = cv.cvtColor(Left_rectified,cv.COLOR_BGR2GRAY)
        grayR_rect = cv.cvtColor(Right_rectified,cv.COLOR_BGR2GRAY)
        imgSize = grayL.shape[::-1]
        imgSize_rect = grayL_rect.shape[::-1]

        # Find chessboard corners in rectified image pair and original image pair
        retL, cornersL = cv.findChessboardCorners(grayL, boardSize, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)
        retR, cornersR = cv.findChessboardCorners(grayR, boardSize, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)
        retL_rect, cornersL_rect = cv.findChessboardCorners(grayL_rect, boardSize, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)
        retR_rect, cornersR_rect = cv.findChessboardCorners(grayR_rect, boardSize, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)

        # Draw epipolar lines on rectified image
        vis_beforeRectification = np.concatenate((imgL, imgR), axis=1)
        vis_afterRectification = np.concatenate((Left_rectified, Right_rectified), axis=1)

        if retR_rect is True and retL is True:
            imagePointsL = [] 
            imagePointsR = [] 
            imagePointsL_rect = [] 
            imagePointsR_rect = [] 

            cv.cornerSubPix(grayR_rect, cornersR_rect, winSize,(-1,-1), calibration_criteria)
            cv.cornerSubPix(grayL_rect, cornersL_rect, winSize,(-1,-1), calibration_criteria)
            imagePointsR_rect.append(cornersR_rect)
            imagePointsL_rect.append(cornersL_rect)

            cv.cornerSubPix(grayR, cornersR, winSize,(-1,-1), calibration_criteria)
            cv.cornerSubPix(grayL, cornersL, winSize,(-1,-1), calibration_criteria)
            imagePointsR.append(cornersR)
            imagePointsL.append(cornersL)

            fontcolor = (0,165,255)
            
            # Get points in 4th row (vertical centre) and display them
            for i in range(0,41,7):

                ## Display horizontal coordinates, connecting lines and horizontal lines at chessboard corners for original image
                x_l = int(round(imagePointsL[0][i][0][0]))
                x_l_text = 'x_l= '+str(x_l)+'px'
                y_l = int(round(imagePointsL[0][i][0][1]))
                cv.circle(vis_beforeRectification, (x_l, y_l), 5, (0,255,255), -1)
                cv.putText(vis_beforeRectification, x_l_text, (x_l+5, y_l-5), font, fontScale, fontcolor, 2, cv.LINE_AA)

                x_r = int(round(imagePointsR[0][i][0][0]))
                x_r_text = 'x_r= '+str(x_r)+'px'
                y_r = int(round(imagePointsR[0][i][0][1]))
                cv.circle(vis_beforeRectification, (x_r+Left_rectified.shape[1], y_r), 5, (0,255,255), -1)
                cv.putText(vis_beforeRectification, x_r_text, (x_r+Left_rectified.shape[1]+5, y_r-5), font, fontScale, fontcolor, 2, cv.LINE_AA)

                cv.line(vis_beforeRectification, (x_l,y_l), (x_r+Left_rectified.shape[1],y_r), (0,255,255), 2)

                
                # Draw horizontal (epipolar) lines starting at left frame's y coordinates
                cv.line(vis_beforeRectification, (0,y_l), (vis_beforeRectification.shape[1],y_l), (255,0,0), 2)   


                ## Display horizontal coordinates, connecting lines and horizontal lines at chessboard corners for rectified image
                x_l = int(round(imagePointsL_rect[0][i][0][0]))
                x_l_text = 'x_l= '+str(x_l)+'px'
                y_l = int(round(imagePointsL_rect[0][i][0][1]))
                cv.circle(vis_afterRectification, (x_l, y_l), 5, (0,255,255), -1)
                cv.putText(vis_afterRectification, x_l_text, (x_l+5, y_l-5), font, fontScale, fontcolor, 2, cv.LINE_AA)
            
                x_r = int(round(imagePointsR_rect[0][i][0][0]))
                x_r_text = 'x_r= '+str(x_r)+'px'
                y_r = int(round(imagePointsR_rect[0][i][0][1]))
                cv.circle(vis_afterRectification, (x_r+Left_rectified.shape[1], y_r), 5, (0,255,255), -1)
                cv.putText(vis_afterRectification, x_r_text, (x_r+Left_rectified.shape[1]+5, y_r-5), font, fontScale, fontcolor, 2, cv.LINE_AA)
                
                # Draw line connecting these chessboard corners
                cv.line(vis_afterRectification, (x_l,y_l), (x_r+Left_rectified.shape[1],y_r), (0,255,255), 2)
                disp_x = x_l - x_r
                depthcamL  = ((-stereoTranslation[0][0])*rectifiedCameraMatrixL[0][0])/(disp_x)
                disp_x_text = ('Camera L disparity: {}px, depth: {:.2f}cm'.format(disp_x, depthcamL))
                cv.putText(vis_afterRectification, disp_x_text, (x_l + int((x_r+Left_rectified.shape[1]-x_l)/3), y_r -10), font, fontScale, fontcolor, 2, cv.LINE_AA)

                # Draw horizontal (epipolar) lines starting at left frame's y coordinates
                cv.line(vis_afterRectification, (0,y_l), (vis_afterRectification.shape[1],y_l), (255,0,0), 2)
                    
        else:
            print('No chessboard found!')

        cv.imshow('Rectification check - before rectification', vis_beforeRectification)
        cv.imshow('Rectification check - after rectification', vis_afterRectification)
        cv.waitKey(0)

    return leftMapX, leftMapY, rightMapX, rightMapY, Q, rectifiedCameraMatrixL


def initialiseCorrespondence():
        minDisparity = 192
        maxDisparity = 272
        numDisparities = maxDisparity-minDisparity
        blockSize = 5
        #disp12MaxDiff = 5
        #uniquenessRatio = 15

        left_matcher = cv.StereoSGBM_create(minDisparity = minDisparity,numDisparities = numDisparities,blockSize = blockSize)
    
        # left_matcher = cv.StereoBM_create(numDisparities = numDisparities, blockSize=blockSize)
        # left_matcher.setMinDisparity(minDisparity)
        sigma = 1.5
        lmbda = 8000.0

        right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
        wls_filter = cv.ximgproc.createDisparityWLSFilter(left_matcher)
        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)

        return left_matcher, right_matcher, wls_filter

def getDepth(Left_rectified, Right_rectified, Q, left_matcher, right_matcher, wls_filter):


    # Left_rectified = cv.cvtColor(Left_rectified,cv.COLOR_BGR2GRAY)
    # Right_rectified = cv.cvtColor(Right_rectified,cv.COLOR_BGR2GRAY)

    left_dispBM = left_matcher.compute(Left_rectified, Right_rectified).astype(np.float32) / 16
    right_dispBM = right_matcher.compute(Right_rectified,Left_rectified).astype(np.float32) / 16
    filtered_dispBM = wls_filter.filter(left_dispBM, Left_rectified, disparity_map_right=right_dispBM)

    points_3DBM_filtered = cv.reprojectImageTo3D(filtered_dispBM, Q)




    return points_3DBM_filtered, filtered_dispBM
