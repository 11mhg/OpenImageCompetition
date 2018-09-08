from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
from .bbox import Box
from tqdm import tqdm
from .tfrecord_utils import convert_to, input_fn, es_input_fn



class Data:
    def __init__(self, classes_text, image_size=(800,800,3),batch_size=32, shuffle_buffer_size=4, prefetch_buffer_size=1,num_parallel_calls=4 , num_parallel_readers=1):
        with open(classes_text) as f:
            self.class_names = []
            for lines in f.readlines():
                arr = lines.strip().split(',')
                self.class_names.append(arr[-1])
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.prefetch_buffer_size = prefetch_buffer_size
        self.num_parallel_calls = num_parallel_calls
        self.num_parallel_readers = num_parallel_readers
    
    def get_batch(self,filenames=None,es=False):
        if es:
            return es_input_fn(self,filenames)
        return input_fn(self, filenames)
        
        
        


class PreProcessData:
    def __init__(self,classes_text='./dummy_labels.txt',image_size=(800,800)):
        self.images = None
        self.labels = None
        self.image_size=image_size
        with open(classes_text) as f:
            self.class_names = []
            for lines in f.readlines():
                arr = lines.strip().split(',')
                self.class_names.append(arr[0])

    def load_mot(self,mot_dir):
        self.images = []
        self.labels = []
        self.name = 'MOT_Training'
        pbar = tqdm(os.listdir(mot_dir))
        max_boxes = 0
        for folder in pbar:
            if '.' in folder:
                continue
            height = 0
            width = 0
            with open(mot_dir+folder+'/seqinfo.ini','r') as info:
                for lines in info:
                    if 'imWidth' in lines:
                        lines = lines.split('=')
                        width = float(lines[1])
                    elif 'imHeight' in lines:
                        lines = lines.split('=')
                        height = float(lines[1])
                    else:
                        continue
            assert height!=0
            assert width!=0
            with open(mot_dir+folder+'/gt/gt.txt','r') as gt:
                dict_annot = {}
                dict_annot['frame'] = {}
                for index, lines in enumerate(gt):
                    splitline = [float(x.strip()) for x in lines.split(',')]
                    label = int(splitline[7])-1
                    x_val = splitline[2]
                    y_val = splitline[3] 
                    box_width = splitline[4]
                    box_height = splitline[5]

                    x_center = x_val +(box_width/2.)
                    y_center = y_val +(box_height/2.)
                    box = Box()
                    box.calculate_xyxy(x_center,y_center,box_width,box_height)
                    box.label = label
                    frame_id = int(splitline[0])
                    if frame_id not in dict_annot['frame']:
                        dict_annot['frame'][frame_id] = []
                    dict_annot['frame'][frame_id].append(box)
                for frame_id in sorted(dict_annot['frame'].keys()):
                    img = mot_dir+folder+'/img1/'+str(frame_id).zfill(6)+'.jpg'
                    boxes = dict_annot['frame'][frame_id]
                    boxes = np.array(boxes)
                    self.images.append((img,height,width,3))
                    if max_boxes < boxes.shape[0]:
                        max_boxes = boxes.shape[0]
                    self.labels.append(boxes)
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        self.num_examples = self.images.shape[0]
        self.max_boxes = max_boxes

    def get_open_images(self,filedir,data_type='train'):
        import csv
        self.images = []
        self.labels = []
        self.name='OpenImages'+'-'+data_type
        self.max_boxes = 0
        self.num_examples = 0
        annotations_file = filedir+'annotations/' + '{}-bbox.csv'.format(data_type)
        with open(annotations_file,'r') as csvfile:
            bbox_reader = csv.reader(csvfile,delimiter=',')
            print("Open Images contains a large number of files, do not be discourage if it takes a long time.")
            next(bbox_reader)
            dict_annot = {}
            pbar = tqdm(bbox_reader)
            pbar.set_description("Reading Annotations")
            for elem in pbar:
                filename = elem[0]
                label = elem[2]
                xmin = float(elem[4])
                xmax = float(elem[5])
                ymin = float(elem[6])
                ymax = float(elem[7])
                #convert label to int
                label = self.class_names.index(label) 
                box = Box(x0=xmin, y0 = ymin, x1=xmax, y1=ymax,label=label)
                if filename not in dict_annot.keys():
                    dict_annot[filename] = []
                    height=1
                    width=1
                #need to read the image height and width every time we run into a new filename
                dict_annot[filename].append(box)
            for filename in dict_annot.keys():
                image_name = filedir+data_type+'/'+filename+'.jpg'
                boxes = np.array(dict_annot[filename])
                self.images.append((image_name,height,width))
                if self.max_boxes < boxes.shape[0]:
                    self.max_boxes = boxes.shape[0]
                self.labels.append(boxes)
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        self.num_examples = self.images.shape[0]
                



    def write_tf(self,directory,num_shards = 1):
        convert_to(self,directory,self.name,num_shards=num_shards)

