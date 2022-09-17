#%% Imports
import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import uuid
import math

from xml.etree.ElementTree import ElementTree
from xml.etree.ElementTree import Element
import xml.etree.ElementTree as etree
import xml.dom.minidom

import random

import shutil

import albumentations as A

def listdirVisible(path):
    items = os.listdir(path)
    visible = [item for item in items if not item.startswith('.')]
    return visible

def clearPreprocessing(path_preprocessing):
    
    folders = os.listdir(path_preprocessing)
    for folder in folders:
        path = path_preprocessing+'/'+folder
        if os.path.isdir(path):
            images = os.listdir(path)
            for image in images:
                subpath = path+'/'+image  
                if os.path.exists(subpath):
                    os.remove(subpath)
               
def clearTraining(path_images):               
    
    # Delete all items in workspace-testing
    folders = os.listdir(path_images)
    for folder in folders:
        path = path_images+'/'+folder
        if os.path.isdir(path):
            images = os.listdir(path)
            for image in images:
                subpath = path+'/'+image  
                if os.path.exists(subpath):
                    os.remove(subpath)



def create_annotation(object_name, pos_x, pos_y, sign_width, sign_height, img_width, img_height, img_depth, file_path, file_name):
    annotation = Element('annotation')
    tree = ElementTree(annotation)

    folder = Element('folder')
    annotation.append(folder)
    folder.text = object_name

    filename = Element('filename')
    annotation.append(filename)
    filename.text = file_name + '.jpg'

    path = Element('path')
    annotation.append(path)
    path.text = file_path

    source = Element('source')
    database = Element('database')
    source.append(database)
    annotation.append(source)
    database.text = 'SelfMade'

    size = Element('size')
    width = Element('width')
    height = Element('height')
    depth = Element('depth')
    size.append(width)
    size.append(height)
    size.append(depth)
    annotation.append(size)
    width.text = str(img_width)
    height.text = str(img_height)
    depth.text = str(img_depth)

    segmented = Element('segmented')
    annotation.append(segmented)
    segmented.text = str(0)

    object_ = Element('object')
    name = Element('name')
    object_.append(name)
    name.text = object_name

    pose = Element('pose')
    object_.append(pose)
    pose.text = str('Unspecified')

    truncated = Element('truncated')
    object_.append(truncated)
    truncated.text = str(0)

    difficult = Element('difficult')
    object_.append(difficult)
    difficult.text = str(0)

    bndbox = Element('bndbox')

    xmin = Element('xmin')
    bndbox.append(xmin)
    xmin.text = str(pos_x)

    ymin = Element('ymin')
    bndbox.append(ymin)
    ymin.text = str(pos_y)

    xmax = Element('xmax')
    bndbox.append(xmax)
    xmax.text = str(pos_x + sign_width)

    ymax = Element('ymax')
    bndbox.append(ymax)
    ymax.text = str(pos_y + sign_height)

    object_.append(bndbox)
    annotation.append(object_)
    
    
    ugly_tree = xml.etree.ElementTree.tostring(annotation)
    dom = xml.dom.minidom.parseString(ugly_tree)
    pretty_tree = dom.toprettyxml()
    
    return pretty_tree

    
BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    
    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
   
    cv.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
    ((text_width, text_height), _) = cv.getTextSize(class_name, cv.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(5, 5))
    plt.axis('off')
    plt.imshow(img)

def place_object(background, object, x_koord, y_koord, width, height):
    
    # Resize object and neutralise alpha channel
    obj_resized = cv.resize(object, (width, height))
    alpha_s = obj_resized[:, :, 3] / 255
    alpha_l = 1.0-alpha_s
    
    # Place object on background
    for c in range(0,3):
        background[y_koord:y_koord+height, x_koord:x_koord+width, c] = (alpha_s * obj_resized[:, :, c] + alpha_l * background[y_koord:y_koord+height, x_koord:x_koord+width, c])
    image_result = background
    
    return image_result

transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True, always_apply=False, p=0.33),
    A.RGBShift(always_apply=False, p=0.66, r_shift_limit=30, g_shift_limit=30, b_shift_limit=30),
    A.Rotate(limit=45, p=0.66, border_mode=cv.BORDER_CONSTANT),
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))



def main(dict_labels, path_backgrounds, path_objects, numImgs, upperScale, lowerScale, save_folder, path_training):
    '''
    @params
    @labels: dictionary with label ids and names
    @path_backgrounds: Path to 'backgrounds' folder
    @path_objects: Path to folder where all images for one label are saved
    '''


    # Get background image paths
    backgrounds = listdirVisible(path_backgrounds)
    n_backgrounds = len(backgrounds)
    print('Found '+str(n_backgrounds)+' backgrounds')

    # Get all object folder paths
    object_folders = listdirVisible(path_objects)
  
    lslabels = list(dict_labels.keys())

    for label in lslabels:

        all_items = []
    
        # Only consider active labels (not old folders)
        if label in object_folders:
            
            # Get all images for current label
            path_images = path_objects+'/'+label
            list_images = listdirVisible(path_images)
            n_images = len(list_images)
            print('Found '+str(n_images)+' objects for label: '+label+'. Multiplying by factor '+str(numImgs))

            # Loop through all images
            for i in range(0, n_images):
                
                # Get object
                path_object = path_images+'/'+list_images[i]
                
                # Open object as CMYK and get alpha-channel
                obj_cmyk = cv.imread(path_object, cv.IMREAD_UNCHANGED)
                alpha = obj_cmyk[:,:,3]

                # Open object second time as BGR and convert to RGB
                obj_bgr = cv.imread(path_object, cv.IMREAD_COLOR)
                b, g, r = cv.split(obj_bgr)

                # Add alpha-channel to RGB 
                obj_rgb = cv.merge([r, g, b, alpha])

                # Get object information
                obj_name = list_images[i].split('_')[0]
                obj_width, obj_height, _ = obj_rgb.shape
                obj_ratio = obj_width / obj_height

                print('Multiplying image: ' +list_images[i])
                
                for j in range(0,numImgs):
                    
                    # Get random background
                    x = np.random.randint(n_backgrounds)
                    bg_name = backgrounds[x].split('.')[0]
                    path_background = path_backgrounds+'/'+str(bg_name)+'.png'
                    background = cv.imread(path_background, 1)
                    background = cv.cvtColor(background, cv.COLOR_BGR2RGB)
                    bnd_width, bnd_height, _ = background.shape

                    # Generate random size of object and determine random position on background
                    rand_size = np.random.randint(lowerScale, upperScale)
                    y_rand = np.random.randint(bnd_height-(rand_size+1))
                    x_rand = np.random.randint(bnd_width-(rand_size / obj_ratio+1))
                    
                    # Place object on chosen background with random size and coordinates
                    image_result = place_object(background, obj_rgb, x_rand, y_rand, int(rand_size/obj_ratio), rand_size)
            
                    # Save chosen coordinates for bboxes
                    bboxes = [[x_rand, y_rand, int(rand_size/obj_ratio), rand_size]]

                    # Set up category ids
                    category_ids = []
                    category_ids.append(dict_labels[label])
                    category_id_to_name = {}
                    category_id_to_name[category_ids[0]]=label


                    # Modify image with Albumentation
                    random.seed()
                    transformed = transform(image=image_result, bboxes=bboxes, category_ids=category_ids)
                    # visualize(
                    #     transformed['image'],
                    #     transformed['bboxes'],
                    #     transformed['category_ids'],
                    #     category_id_to_name,
                    # )

                    image_transformed = transformed['image']
                    bbox_transformed = transformed['bboxes'][0]

                    # Save images to 'Preprocessing' folder
                    file_name = obj_name + str(uuid.uuid1())
                    img_width, img_height, img_depth = image_transformed.shape
                    save_path = save_folder+label

                    abs_path = os.path.abspath(save_path)

                    x_min = bbox_transformed[0]
                    y_min = bbox_transformed[1]
                    bbx_width = bbox_transformed[2]
                    bbx_height = bbox_transformed[3]
            
                    annotation = create_annotation(obj_name, int(x_min), int(y_min), int(bbx_width), int(bbx_height), img_width, img_height, img_depth, abs_path, file_name)
                    
                    xml_File = open(save_path + '/'+ file_name + '.xml','w')
                    xml_File.write(annotation)
                    xml_File.close()
                    #print('Saving image "'+file_name+'" and xml file to directory: '+str(save_path))
                    cv.imwrite(save_path +'/'+ file_name + '.jpg', cv.cvtColor(image_transformed, cv.COLOR_RGB2BGR))

                    # Add images and their respective xml files to array
                    all_items.append([save_path +'/'+ file_name + '.jpg', save_path +'/'+ file_name + '.xml'])


        ## Partition images to testing / training folders from 1_Preprocessing
        
        # Count number of images in each label folder
        n_items = len(all_items)

        # Use 15% of the images for testing, 85% for training
        n_testing = 2*(math.ceil(0.5*0.15*n_items))
    
        n_training = n_items - n_testing
        print('Multiplication completed. Currently available for label: '+str(label)+', total: '+str(n_items)+', testing: '+str(n_testing)+', training: '+str(n_training))      
              
        # Partition images for training and testing in a random order
        
        # Training
        count = 0
        number_history = []
        for i in range(n_training):
            randnum = np.random.randint(len(all_items))
            number_history.append(randnum)
            source_jpg = all_items[randnum][0]
            source_xml = all_items[randnum][1]
          
            # Ignore hidden files, such as .ds_store
            if not (all_items[randnum][0].startswith('.') or all_items[randnum][1].startswith('.')):
                training = path_training+'images/training'
                
                shutil.copy(source_jpg, training)
                shutil.copy(source_xml, training)
                all_items.pop(randnum)
                count +=1 

        # Testing
        count = 0
        number_history = []
        for i in range(n_testing):
            randnum = np.random.randint(len(all_items))
            number_history.append(randnum)
            source_jpg = all_items[randnum][0]
            source_xml = all_items[randnum][1]
            
            # Ignore hidden files, such as .ds_store
            if not (all_items[randnum][0].startswith('.') or all_items[randnum][1].startswith('.')) :
                testing = path_training+'images/testing'
                shutil.copy(source_jpg, testing)
                shutil.copy(source_xml, testing)
                all_items.pop(randnum)
                count +=1