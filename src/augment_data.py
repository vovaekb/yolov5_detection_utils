import os
import re
import math
import cv2
import yaml
import numpy as np
import pandas as pd
import imageio
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa
from PIL import Image
from PIL import ImageDraw

LABELS_DIR = "train_data/labels"
IMAGES_DIR = "train_data/images"
AUG_IMAGES_DIR = "train_data/aug_images"
AUG_IMAGES_BBOXES_DIR ="train_data/aug_images_bboxes"
AUG_LABELS_DIR = "train_data/aug_labels"
IMG_PREFIX = "aug1_"
IMG_WIDTH = 600
IMG_HEIGHT = 400

draw_bboxes = False


class DataAugmenter:
    """
    Class for aplying data augmentations to images and labels 
    
    Methods
    -------
    setup()
        Performs initial setup.
    
    get_class_folders()
        Obtains list of classes from folder names in training data
    """
    def __init__(self):
        self.classes = None
        self.bb_list = []
        self.bb_list_df = None
        self.aug = iaa.SomeOf(2, [
            iaa.Affine(scale=(0.5, 1.5)),
            iaa.Affine(rotate=(-60, 60)),
            iaa.Affine(translate_percent={"x":(-0.3, 0.3),"y":(-0.3, 0.3)}),
            iaa.Fliplr(1),
            iaa.Flipud(1)
        ])

    def setup(self):
        self.classes = list(self.get_class_folders(LABELS_DIR))
        # print(self.classes)
        self.classes.remove('normal')
        self.classes = sorted(self.classes, key=str.lower)
        # print(self.classes)

    def get_class_folders(self, path):
        for folder in os.listdir(path):
            if os.path.isdir(os.path.join(path, folder)):
                yield folder

    def bbs_obj_to_df(self, bbs_object):
        """
        Converts bounding box object to DataFrame
		
        Parameters
		----------
		bbs_object: imgaug.augmentables.bbs.BoundingBoxesOnImage 
			bounding box object to convert
		
        Returns
		-------
		df_bbs: pd.DataFrame
			the conversion result
        """
        # print('bbs_obj_to_df')
        #     convert BoundingBoxesOnImage object into array
        bbs_array = bbs_object.to_xyxy_array()
        # convert to a DataFrame
        df_bbs = pd.DataFrame(bbs_array, columns=['xmin', 'ymin', 'xmax', 'ymax'])
        return df_bbs

    def convert(self, size, box):
        """
        Converts bounding box to YOLO format

        Parameters
		----------
		size: tuple(int, int)
			image size (width, height) 
		
        box: tuple(int, int, int, int)
			bounding box to convert
        
        Returns
		-------
		result: tuple(int, int, int, int)
			the conversion result (x_center, y_center, width, height)
        """
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)
        
    def draw_bbox(self, image, class_name, filename, bbs_df):
        """
        Draw bounding box from DataFrame
        """
        PIL_image = Image.fromarray(image.astype('uint8'), 'RGB')
        draw = ImageDraw.Draw(PIL_image)
        draw.rectangle([(bbs_df['xmin'].iloc[0], bbs_df['ymin'].iloc[0]), (bbs_df['xmax'].iloc[0], bbs_df['ymax'].iloc[0])], outline ='red')
        # PIL_image.show()
        PIL_image.save('%s/%s/%s%s' % (AUG_IMAGES_BBOXES_DIR, class_name, IMG_PREFIX, filename))
        
    def save_labels(self, label_path, image_aug, class_name, bbs_df):
        """
        Save augmented bounding box labels in YOLO format
		
        Parameters
		----------
        label_path: str
            path to label file
            
        image_aug: np.array
            augmented image
        
        class_name: str 
            class name of augmented image

		bbs_df: pd.DataFrame
			bounding boxes
		
        Returns
		-------
        """
        with open(label_path, 'w') as f:
            for index, row in bbs_df.iterrows():
                xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                w, h = image_aug.shape[1], image_aug.shape[0]
                b = (xmin, xmax, ymin, ymax)
                bb = self.convert((w, h), b)
                class_ind = self.classes.index(class_name)
                label_line = '{0} {1:.2f} {2:.2f} {3:.2f} {4:.2f}\n'.format(class_ind, *bb)
                f.write(label_line)
    
    def convert_to_df(self):
        """
        Converts bounding boxes for training images in YOLO format to DataFrame
		
        Parameters
		----------
        """
        for class_folder in self.get_class_folders(LABELS_DIR):
            print('class: ', class_folder)
            label_files = [f for f in os.listdir(os.path.join(LABELS_DIR, class_folder))
                            if f.endswith('txt') 
                            and os.path.getsize(os.path.join(LABELS_DIR, class_folder, f)) > 0]
            for label_file in label_files:
                img_name = label_file.split('.')[0] + '.jpeg'
                
                with open(os.path.join(LABELS_DIR, class_folder, label_file)) as f:
                    for line in f:
                        # print(line.split(' '))
                        class_ind = int(line.split(' ')[0])
                        # print('class_ind:', class_ind)

                        box = [float(x) for x in line.split(' ')[1:]] * np.array([IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH, IMG_HEIGHT])
                        (x_center, y_center, width, height) = box.astype("int")
                        
                        x_min = int(x_center - (width / 2))
                        y_min = int(y_center - (height / 2))
                        x_max = x_min + int(width)
                        y_max = y_min + int(height)

                        class_name = self.classes[class_ind]
                        value = (img_name, IMG_WIDTH, IMG_HEIGHT, class_name, x_min, y_min, x_max, y_max)
                        self.bb_list.append(value)
        
        column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        self.bb_list_df = pd.DataFrame(self.bb_list, columns=column_name)

        # save to csv
        self.bb_list_df.to_csv(('labels.csv'), index=None)

    def augment(self):
        print('Augmenting ...')
        aug_bbs_xy = pd.DataFrame(columns=
                                    ['filename','width','height','class', 'xmin', 'ymin', 'xmax', 'ymax']
        )
        
        class_grouped = self.bb_list_df.groupby('class')
        
        for class_name in self.bb_list_df['class'].unique():
            class_group_df = class_grouped.get_group(class_name)
            filename_grouped = class_group_df.groupby('filename')
            
            for filename in class_group_df['filename'].unique():
                name = filename.split('.')[0]
                # get separate data frame grouped by file name
                filename_group_df = filename_grouped.get_group(filename)
                filename_group_df = filename_group_df.reset_index()
                filename_group_df = filename_group_df.drop(['index'], axis=1)
                
                # read image and bbox labels
                image_path = '%s/%s/%s' % (IMAGES_DIR, class_name, filename) 
                image = imageio.imread(image_path)
                bb_array = filename_group_df.drop(['filename', 'width', 'height', 'class'], axis=1).values
                bbs = BoundingBoxesOnImage.from_xyxy_array(bb_array, shape=image.shape)
                
                #   apply augmentation on image and on the bounding boxes
                image_aug, bbs_aug = self.aug(image=image, bounding_boxes=bbs)
                #   disregard bounding boxes which have fallen out of image pane    
                bbs_aug = bbs_aug.remove_out_of_image()
                #   clip bounding boxes which are partially outside of image pane
                bbs_aug = bbs_aug.clip_out_of_image()
                #   don't perform any actions with the image without bounding boxes and for class storage
                if re.findall('Image...', str(bbs_aug)) == ['Image([]'] or class_name == 'storage':
                    pass
                else:
                    #   create a data frame with augmented values of image width and height
                    info_df = filename_group_df.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1)
                    for index, _ in info_df.iterrows():
                        info_df.at[index, 'width'] = image_aug.shape[1]
                        info_df.at[index, 'height'] = image_aug.shape[0]
                    #   rename filenames by adding the predifined prefix
                    info_df['filename'] = info_df['filename'].apply(lambda x: IMG_PREFIX + x)
                    #   create a data frame with augmented bounding boxes coordinates
                    bbs_df = self.bbs_obj_to_df(bbs_aug)
                    bbs_df[['xmin', 'ymin', 'xmax', 'ymax']] = bbs_df[['xmin', 'ymin', 'xmax', 'ymax']].astype(int)
                    
                    #   concat all new augmented info into new data frame
                    aug_df = pd.concat([info_df, bbs_df], axis=1)
                    if aug_df.isnull().sum().sum() == 0:
                        # draw bbox
                        if draw_bboxes: 
                            self.draw_bbox(image_aug, class_name, filename, bbs_df)
                        # save augmented image
                        cv2.imwrite('%s/%s/%s%s' % (AUG_IMAGES_DIR, class_name, IMG_PREFIX, filename), image_aug)
                        #   append rows to aug_bbs_xy data frame
                        aug_bbs_xy = pd.concat([aug_bbs_xy, aug_df])
                        
                        # write bboxes to label file
                        label_path = os.path.join(AUG_LABELS_DIR, class_name, name + '.txt')
                        self.save_labels(label_path, image_aug, class_name, bbs_df)
        
        # construct dataframe with updated images and bounding boxes annotations 
        aug_bbs_xy = aug_bbs_xy.reset_index()
        aug_bbs_xy = aug_bbs_xy.drop(['index'], axis=1)


    def process(self):
        self.setup()
        self.convert_to_df()
        self.augment()
        

def augment_data():
    augmenter = DataAugmenter()
    augmenter.process()

if __name__ == '__main__':
    augment_data()
