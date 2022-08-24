## Setup workstation
import os
import shutil
import platform
from pprint import pprint

import subprocess
import sys

def getIndex(element, array):
    for i in range(len(array)):
        name = array[i].split('==')[0]
        if element == name:
            index = i
            break   
    return index

def moveAllContent(source, destination):
    
    items = os.listdir(source)
    # directories = [item for item in items if (os.path.isdir(source+'/'+item))]
    # files = [item for item in items if (os.path.isfile(source+'/'+item))]

    # Remove .DS Store files
    # if '.DS_Store' in directories:
    #     directories.remove('.DS_Store')

    # if '.DS_Store' in files:
    #     files.remove('.DS_Store')
    if '.DS_Store' in items:
        items.remove('.DS_Store')

    # for directory in directories:
    #     print('Copying {} from {} to {}'.format(directory, source+'/'+directory, destination))
    #     shutil.move(source+'/'+directory, destination)
        
    # for file in files:
    #     shutil.move(source+'/'+file, destination)

    for item in items:
        shutil.move(source+'/'+item, destination)

def install(package):
    subprocess.run([sys.executable, "-m", "pip", "install", package])

def uninstall(package):
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", package, "-y"])



#%% Create folder structure

def createFolderStructure(labels, home_path):
    systemName = platform.system()

    # Create dictionary with paths to most used directories
    directories = {
        '0_UserInput': os.path.join(home_path,'0_UserInput'),
        'backgrounds': os.path.join(home_path,'0_UserInput/backgrounds'),
        'objects': os.path.join(home_path,'0_UserInput/objects'),
        'trainedModels': os.path.join(home_path,'0_UserInput/trainedModels'),
        'scripts': os.path.join(home_path,'0_UserInput/scripts'),

        
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
        'tf_models': os.path.join(home_path,'2_ObjectDetection','2_Tensorflow/models'),
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
            nothing = 0
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
    # all other required packages will be installed and updated by TF2 object detection API
    install('wget==3.2')
    print('Successfully checked installation of wget...')
    install('pyqt5==5.15.7')
    print('Successfully checked installation of pyqt5...')
    install('matplotlib==3.5.2')
    print('Successfully checked installation of matplotlib...')
    install('pandas==1.4.3')
    print('Successfully checked installation of pandas...')
    install('GitPython')
    print('Successfully checked installation of GitPython...')
    

def installOpenCV():

    import pkg_resources
    installed_packages = pkg_resources.working_set
    installed_packages_list = sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])

    #pprint(installed_packages_list)


    # OpenCV
    for i in range(len(installed_packages_list)):
        package_name = installed_packages_list[i].split('==')[0]
        #print('Checking package: '+installed_packages_list[i])

        if package_name == 'opencv-contrib-python':
            package_version = installed_packages_list[i].split('==')[1]

            if package_version == '4.6.0.66':
                print('Package: "{}" version: {} is installed. Required version: {}. Ok...'.format(package_name, package_version,'4.6.0.66'))
            else:
                print('Package: "{}" version: {} is installed. Required version: {}. Uninstalling...'.format(package_name, package_version,'4.6.0.66'))
                uninstall(package_name)
                print('Package: {} version: {}. Installing...'.format(package_name,'4.6.0.66'))
                install('opencv-contrib-python==4.6.0.66') 

        elif ('opencv' in package_name) and (package_name != 'opencv-contrib-python'):
            print('Package: "{}" is installed. Package: "{}" required. Uninstalling ...'.format(package_name, 'opencv-contrib-python'))
            uninstall(package_name)
            print('Package: "{}" version: {}. Installing...'.format('opencv-contrib-python','4.6.0.66'))
            install('opencv-contrib-python==4.6.0.66')
        
        else:
            break
        
    install('opencv-contrib-python==4.6.0.66')

def installAlbumentation():
    os.system('pip install -U albumentations --no-binary qudida,albumentations')

def installModelGarden(paths):
    from git import Repo
    # Download the model garden (model garden is an environment that is necessary to train new models from scratch or to continue training existing models)
    # The model itself will be downloaded later

    # Clone repository only if it does not exist already
    if os.path.exists(paths['2_Tensorflow']+'/models/research') is False:

        # Clone git repo to temporary folder (because it cannot be cloned to not-empty directory)
        print('Cloning tensorflow model garden...')

        Repo.clone_from('https://github.com/tensorflow/models.git', paths['tf_models'])

        print('Cloning complete...')
        
    else:
        print('Model garden is already installed...')
        
    os.chdir(paths['home'])


def installProtobuf(paths):
    import wget
    
    # Install protobuf
    if os.path.exists(paths['research']+'/object_detection/protos/matcher_pb2.py') is False:

        #Go to destination directory
        os.chdir(paths['protoc'])
        protoc_url = 'https://github.com/protocolbuffers/protobuf/releases/download/v21.1/protoc-21.1-osx-aarch_64.zip'
        wget.download(protoc_url)

        # Extract all content of downloaded file
        from zipfile import ZipFile

        with ZipFile('protoc-21.1-osx-aarch_64.zip', 'r') as zipObj:
            zipObj.extractall()

        os.environ['Path'] = paths['protoc']+'/bin'
        os.chdir(paths['research'])

        os.system('protoc object_detection/protos/*.proto --python_out=.')

        files = os.listdir(paths['research']+'/object_detection/protos')
        protoFiles = [file for file in files if ('.proto' in file)]
        pyFiles = [file for file in files if ('.py' in file)]
        
        if len(protoFiles)+1 == len(pyFiles):
            print(' Protoc was installed successfully...')
        else:
            print('There has been an error while installing protoc...')
            sys.exit()
        
    else:
        print('Protobuf is already installed...')
        
    os.chdir(paths['home'])


def installCocoAPI(paths):
    from git import Repo
    
    # Clone repository only if it does not exist already
    if os.path.exists(paths['research']+'/cocoapi/README.txt') is False:

        # Create temporary folder
        os.makedirs(paths['research']+'/cocoapi')

        # Clone git repo to temporary folder (because it cannot be cloned to not-empty directory)
        print('Cloning cocoapi..')
        Repo.clone_from('https://github.com/cocodataset/cocoapi.git', paths['research']+'/cocoapi')
        
        print('Cloning complete...')
    else:
        print('Cocoapi is already installed...')

    os.chdir(paths['home'])


def installODAPI(paths):

    systemName = platform.system()

    # Check if API has already been installed
    if os.path.exists(paths['research']+'/checkAudex.txt') is False:
        print('Installing setup.py...')
        
        # Move to 'research' directory
        os.chdir(paths['research'])

        # Copy setup.py to current working directory
        subprocess.run(['cp','object_detection/packages/tf2/setup.py', '.' ])

        # Execute setup.py (this command installs all dependencies needed for tf2 odapi)
        subprocess.run(['python', '-m', 'pip', 'install', '.'])

        if systemName == 'Windows' or systemName == 'Darwin':
            # Customize visualisation toolbox of TF
            mod_name = 'objectDetectionModCode.py'
            file_name = 'visualization_utils.py'
            source = paths['0_UserInput']+'/scripts/objectDetectionModCode.py'

            dest_1 = sys.prefix+'/lib/python3.9/site-packages/object_detection/utils'
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


def checkODAPI(paths):

    systemName = platform.system()

    # Move to 'research' directory
    os.chdir(paths['research'])
    import object_detection

    # Local machine
    if systemName == 'Darwin' or systemName == 'Windows':
        print('Checking installation of TF2 object detection API...')
        #!python {paths['research']+'/object_detection/builders/model_builder_tf2_test.py'}
        subprocess.run(['python', paths['research']+'/object_detection/builders/model_builder_tf2_test.py'])
        

    # Google colab    
    elif systemName == 'Linux':
        
        subprocess.run(['python', '-m', 'pip', 'install', 'numpy', '--upgrade']) # This had to be added for execution on colab. Problem solved using stackoverflow


    else:
        print('No operating system was defined...')

    # Move back to home directory
    os.chdir(paths['home'])


def installPackages(home_path, labels, firstInstallation):

    if firstInstallation is True:
        files, paths = createFolderStructure(labels, home_path)

        # Install packages required for object detection (not training)
        installBasicPackages()
        installAlbumentation()
        installModelGarden(paths)
        installProtobuf(paths)
        installCocoAPI(paths)
        installODAPI(paths)
        installOpenCV()
        checkODAPI(paths) 

    elif firstInstallation is False:
        files, paths = createFolderStructure(labels, home_path)
        #checkODAPI(paths)
        installOpenCV()
        


    return files, paths