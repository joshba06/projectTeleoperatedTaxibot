## Setup workstation
import os
import shutil
import platform
from pprint import pprint
from git import Repo
import subprocess
import sys
import wget
from zipfile import ZipFile

import pkg_resources
installed_packages = pkg_resources.working_set
installed_packages_list = sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])

import userSettings
userSettings = userSettings.main()
pathVenv = userSettings['pathLocalInterpreter']

def getIndex(element, array):
    for i in range(len(array)):
        if element == array[i]:
            index = i
            break   
    return index

def moveAllContent(source, destination):
    
    items = os.listdir(source)

    if '.DS_Store' in items:
        items.remove('.DS_Store')

    for item in items:
        shutil.move(source+'/'+item, destination)


def install(package):
    subprocess.call([pathVenv, "-m", "pip", "install", package])

def uninstall(package):
    subprocess.check_call([pathVenv, "-m", "pip", "uninstall", package, "-y"])


#%% Create folder structure
def createFolderStructure(labels, home_path, colab=False):

    # Create dictionary with paths to most used directories
    directories = {
        '0_UserInput': os.path.join(home_path,'0_UserInput'),
        'backgrounds': os.path.join(home_path,'0_UserInput/backgrounds'),
        'objects': os.path.join(home_path,'0_UserInput/objects'),
        'trainedModels': os.path.join(home_path,'0_UserInput/trainedModels'),
        'scripts': os.path.join(home_path,'0_UserInput/scripts'),
        'requiredPackages': os.path.join(home_path,'0_UserInput/requiredPackages'),
        
        
        '1_CameraCalibration': os.path.join(home_path,'1_Calibration'),
        'stereo': os.path.join(home_path,'1_Calibration/stereo'),
        'stereoTesting': os.path.join(home_path,'1_Calibration/stereo/testing'),
        'individual': os.path.join(home_path,'1_Calibration/individual'),
        'stCamL': os.path.join(home_path,'1_Calibration/stereo/camL'),
        'stCamR': os.path.join(home_path,'1_Calibration/stereo/camR'),
        'indCamR': os.path.join(home_path,'1_Calibration/individual/camR'),
        'indCamL': os.path.join(home_path,'1_Calibration/individual/camL'),

        '2_ObjectDetection': os.path.join(home_path,'2_ObjectDetection'),
        
        '1_Preprocessing': os.path.join(home_path,'2_ObjectDetection','1_Preprocessing'),
        'images': os.path.join(home_path,'2_ObjectDetection','1_Preprocessing/images'),

        '2_Tensorflow': os.path.join(home_path,'2_ObjectDetection','2_Tensorflow'),
        'protoc': os.path.join(home_path,'2_ObjectDetection','2_Tensorflow/protoc'),
        'workspace': os.path.join(home_path,'2_ObjectDetection','2_Tensorflow/workspace'), 
        'training': os.path.join(home_path,'2_ObjectDetection','2_Tensorflow/workspace/training'),
        'annotations': os.path.join(home_path,'2_ObjectDetection','2_Tensorflow/workspace/training/annotations'),
        'images_training': os.path.join(home_path,'2_ObjectDetection','2_Tensorflow/workspace/training/images/training'),
        'images_testing': os.path.join(home_path,'2_ObjectDetection','2_Tensorflow/workspace/training/images/testing'),
        'models': os.path.join(home_path,'2_ObjectDetection','2_Tensorflow/workspace/training/models'),
        'pre_trained_models': os.path.join(home_path,'2_ObjectDetection','2_Tensorflow/workspace/training/pre_trained_models'),

        '3_Output': os.path.join(home_path,'2_ObjectDetection','3_Output'),
    }


    # Create folder structure from dictionary
    for key in directories:
        
        # If path does not exist, create new
        if os.path.exists(directories[key]) is False:
            try:
                os.makedirs(directories[key])
            except OSError:
                print('Failed to create {} from scratch.'.format(directories[key]))
            else:
                print ('Successfully created {} from scratch.'.format(directories[key]))
            
        # If path does exist, do not replace old path
        else:        
            pass
            #print('%s already exists..' %paths[key])

    
    # Create subfolders for labels
    for label in labels:
        temp_path_prep = os.path.join(directories['images'], label)
        
        if os.path.exists(temp_path_prep) is False:
            try:
                os.makedirs(temp_path_prep)
            except OSError:
                print('Failed to create {} from scratch.'.format(temp_path_prep))
            else:
                print ('Successfully created {} from scratch.'.format(temp_path_prep))


    # Add paths for which directories shall not be created, but ones that are frequently used
    directories['home'] = home_path
    directories['research'] = directories['2_Tensorflow']+'/models/research'

        # Create dictionary with paths to most used files
    files = {
                'labelmap': os.path.join(directories['annotations']+'/label_map.pbtxt'),
                'tf_train': os.path.join(directories['annotations']+'/train.record'),
                'tf_test': os.path.join(directories['annotations']+'/test.record'),
                'generateTfRecord': os.path.join(directories['scripts']+'/generatetfrecord.py'),

    }

    return files, directories


def installBasicPackages():

    # Upgrade pip 
    subprocess.call([pathVenv, "-m", "pip", "install", "--upgrade", "pip"])

    # all other required packages will be installed and updated by TF2 object detection API
    install('wget==3.2')
    print('Successfully checked installation of wget...')
    install('matplotlib==3.5.2')
    print('Successfully checked installation of matplotlib...')
    install('pandas==1.4.3')
    print('Successfully checked installation of pandas...')
    install('GitPython')
    print('Successfully checked installation of GitPython...')
    

def installOpenCV():

    # Delete any installed version of opencv
    try:
        uninstall('opencv-python')
    except:
        nothing = 0
    else:
        try:
            uninstall('opencv-python-headless')
        except:
            nothing = 0
        else:
            try:
                uninstall('opencv-contrib-headless')
            except:
                nothing = 0
            
            else:
                try:
                    uninstall('opencv-contrib-python')
                except:
                    nothing = 0

    # Re-install openCV from scratch
    install('opencv-contrib-python==4.6.0.66')

def installAlbumentation():
    subprocess.call([pathVenv, "-m", "pip", "install", "-U", "albumentations", "--no-binary", "qudida", "albumentations"])

def installModelGarden(paths):
    
    # Clone repository only if it does not exist already
    if os.path.exists(paths['2_Tensorflow']+'/models/research') is False:

        print('Setting up model garden...')

        # Clone model garden from github
        Repo.clone_from('https://github.com/tensorflow/models.git', paths['tf_models'])

        # Copy model garden from "required packages" direcory
        # shutil.copytree(paths['requiredPackages']+'/modelGarden/models', paths['2_Tensorflow']+'/models')

        print('Setting up model garden complete...')
        
    else:
        print('Model garden is already installed...')
        
    os.chdir(paths['home'])


def installProtobuf(paths):
    
    # Install protobuf
    if os.path.exists(paths['research']+'/object_detection/protos/matcher_pb2.py') is False:
        
        systemName = platform.system()

        #Go to destination directory
        os.chdir(paths['protoc'])
        print('Setting up protos...')

        # Mac
        if systemName == 'Darwin':

            # Download from github
            # protoc_url = 'https://github.com/protocolbuffers/protobuf/releases/download/v21.1/protoc-21.1-osx-aarch_64.zip'
            # wget.download(protoc_url)

            # Copy from folder "required packages"
            shutil.copy(paths['requiredPackages']+'/protobuf/protoc-21.1-osx-aarch_64.zip', paths['protoc'])

            # Extract all content of downloaded file
            with ZipFile('protoc-21.1-osx-aarch_64.zip', 'r') as zipObj:
                zipObj.extractall()

            # Add path to file to system variable
            os.environ['Path'] = paths['protoc']+'/bin'
            os.chdir(paths['research'])

            os.system('protoc object_detection/protos/*.proto --python_out=.')

        # Linux
        elif systemName == 'Linux':
            protoc_url = 'https://github.com/protocolbuffers/protobuf/releases/download/v21.5/protoc-21.5-linux-x86_64.zip'
            wget.download(protoc_url)

            # Extract all content of downloaded file
            with ZipFile('protoc-21.5-linux-x86_64.zip', 'r') as zipObj:
                zipObj.extractall()

            os.environ['Path'] = paths['protoc']+'/bin'
            os.chdir(paths['research'])

            os.system('protoc object_detection/protos/*.proto --python_out=.')


        # Windows
        elif systemName == 'Windows':
            protoc_url = 'https://github.com/protocolbuffers/protobuf/releases/download/v21.5/protoc-21.5-win64.zip'
            wget.download(protoc_url)

            # Extract all content of downloaded file
            with ZipFile('protoc-21.5-win64.zip', 'r') as zipObj:
                zipObj.extractall()

            os.environ['Path'] = paths['protoc']+'/bin'
            os.chdir(paths['research'])

            os.system('for /f %i in ("dir /b object_detection\protos\*.proto") do protoc object_detection\protos\%i --python_out=.')
                
        else:
            pass


        files = os.listdir(paths['research']+'/object_detection/protos')
        protoFiles = [file for file in files if ('.proto' in file)]
        pyFiles = [file for file in files if ('.py' in file)]
        
        if len(protoFiles)+1 == len(pyFiles):
            print(' Protoc was set up successfully...')
        else:
            print('There has been an error while installing protoc...')
            sys.exit()
        
    else:
        print('Protobuf is already installed...')
        
    os.chdir(paths['home'])


def installCocoAPI(paths):
    
    # Clone repository only if it does not exist already
    if os.path.exists(paths['research']+'/cocoapi/README.txt') is False:

        print('Setting up cocoapi..')

        # Clone from github
        # Repo.clone_from('https://github.com/cocodataset/cocoapi.git', paths['research']+'/cocoapi')

        # Copy from required packages folder
        shutil.copytree(paths['requiredPackages']+'/coco/cocoapi', paths['research']+'/cocoapi')
        
        print('Setup completed...')
    else:
        print('Cocoapi is already installed...')

    os.chdir(paths['home'])


def installODAPI(paths):

    systemName = platform.system()

    # Check if API has already been installed
    if os.path.exists(paths['research']+'/checkAudex.txt') is False:
        
        print('Installing object detection API...')
        
        # Move to 'research' directory
        os.chdir(paths['research'])

        # Copy setup.py to current working directory
        subprocess.run(['cp','object_detection/packages/tf2/setup.py', '.' ])

        # Execute setup.py (this command installs all dependencies needed for tf2 odapi)
        subprocess.run(['python', '-m', 'pip', 'install', '.'])
        # subprocess.call([path_venv, "-m", "pip", "install", "."])

        # Customize visualisation toolbox of TF
        mod_name = 'objectDetectionModCode.py'
        file_name = 'visualization_utils.py'
        source = paths['0_UserInput']+'/scripts/objectDetectionModCode.py'

        path_to_venv= userSettings['pathLocalInterpreter'].split('/')
        index = getIndex(userSettings['nameVenv'], path_to_venv)
        prefix = '/'.join(path_to_venv[:index+1])

        dest_1 = prefix+'/lib/python3.9/site-packages/object_detection/utils'  # Check if this really is python 3.9 or 3
        dest_2 = paths['2_Tensorflow']+'/models/research/object_detection/utils'
        
        os.remove(dest_1+'/'+file_name)
        os.remove(dest_2+'/'+file_name)
        shutil.copy(source, dest_1)
        shutil.copy(source, dest_2)
        os.rename(dest_1+'/'+mod_name, dest_1+'/'+file_name)
        os.rename(dest_2+'/'+mod_name, dest_2+'/'+file_name)
        
        print('Installation complete..')

        # Create a file to check for when running this code
        with open(paths['research']+'/checkAudex.txt', 'w') as file:
            file.write('Installed ODAPI')

    else:
        print('Object Detection API is already installed...')

    # Move back to home-directory
    os.chdir(paths['home'])


def checkODAPI(paths, colab=False):

    print('Checking installation of TF2 object detection API...')

    # Move to 'research' directory
    os.chdir(paths['research'])
    # import object_detection

    if colab == False:
        subprocess.run([pathVenv, paths['research']+'/object_detection/builders/model_builder_tf2_test.py'])
        
    else:
        subprocess.run([pathVenv, '-m', 'pip', 'install', 'numpy', '--upgrade']) # This had to be added for execution on colab. Problem solved using stackoverflow
        subprocess.run([pathVenv, paths['research']+'/object_detection/builders/model_builder_tf2_test.py'])

    # Move back to home directory
    os.chdir(paths['home'])


def installPackages(home_path, labels, firstInstallation=False):

    if firstInstallation == True:

        print('.\nSetting up system for the first time...\n.')

        files, paths = createFolderStructure(labels, home_path, colab=False)

        # Install packages required for object detection (not training)
        installBasicPackages()
        installAlbumentation()
        installModelGarden(paths)
        installProtobuf(paths)
        installCocoAPI(paths)
        installODAPI(paths)
        installOpenCV()
        checkODAPI(paths)

        response = input('Continue? (y/n)')
        
        if response == 'y':
            print('.\nSystem setup completed...\n.')
        else:
            print('Program terminated by user...')
            sys.exit()

    elif firstInstallation == False:

        print('.\nChecking system setup\n.')
        files, paths = createFolderStructure(labels, home_path, colab=False)
        print('.\nSystem setup completed...\n.')

    return files, paths