## Setup workstation
import os
import shutil


#%% Create folder structure

def createFolderStructure(labels, home_path, system_id):
    # Create dictionary with paths to most used directories
    paths = {
        '0_User_Input': os.path.join(home_path,'0_User_Input'),
        'backgrounds': os.path.join(home_path,'0_User_Input/backgrounds'),
        'objects': os.path.join(home_path,'0_User_Input/objects'),

        '1_Preprocessing': os.path.join(home_path,'1_Preprocessing'),
        'images': os.path.join(home_path,'1_Preprocessing/images'),

        '2_Tensorflow': os.path.join(home_path,'2_Tensorflow'),
        'protoc': os.path.join(home_path,'2_Tensorflow/protoc'),
        'workspace': os.path.join(home_path,'2_Tensorflow/workspace'),
        'scripts': os.path.join(home_path,'2_Tensorflow/workspace/scripts'),   
        'training': os.path.join(home_path,'2_Tensorflow/workspace/training'),
        'annotations': os.path.join(home_path,'2_Tensorflow/workspace/training/annotations'),
        'images_training': os.path.join(home_path,'2_Tensorflow/workspace/training/images/training'),
        'images_testing': os.path.join(home_path,'2_Tensorflow/workspace/training/images/testing'),
        'models': os.path.join(home_path,'2_Tensorflow/workspace/training/models'),
        'pre_trained_models': os.path.join(home_path,'2_Tensorflow/workspace/training/pre_trained_models'),

        '3_Output': os.path.join(home_path,'3_Output'),
    }

    ## On colab, delete entire folder structure before setting it up
    
    # Local machine
    if system_id == 0:
        flag = 0

    # Google colab    
    elif system_id == 1:
        
        # If folder structure exist, delete it first
        path = paths['2_Tensorflow']
        if (os.path.exists(path)):
            shutil.rmtree(path)
    else:
        print('No operating system was defined...')

    # Create folder structure from dictionary
    for key in paths:
        
        # If path does not exist, create new
        if os.path.exists(paths[key]) is False:
            
            try:
                os.makedirs(paths[key])
            except OSError:
                print('Failed to create %s from scratch.' % paths[key])
            else:
                print ('Successfully created %s from scratch. ' % paths[key])        
            
        # If path does exist, do not replace old path
        else:        
            print('%s already exists..' %paths[key])

    
    ## Create subfolders for labels
    for label in labels:
        temp_path_prep = os.path.join(paths['images'], label)
        
        if os.path.exists(temp_path_prep) is False:
            try:
                os.makedirs(temp_path_prep)
            except OSError:
                print('Failed to create %s from scratch.' % temp_path_prep)
            else:
                print ('Successfully created %s from scratch. ' % temp_path_prep)

    # Create dictionary with paths to most used files
    files = {}
            
    # Add paths for which directories shall not be created, but ones that are frequently used
    paths['home'] = home_path
    paths['research'] = paths['2_Tensorflow']+'/models/research'

    return files, paths