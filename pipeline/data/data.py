from __future__ import print_function

import numpy as np
import tensorflow as tf
from .bbox import *
import os
from tqdm import tqdm
from .tfrecord_utils import input_fn, _identity, generator_masks, get_instance
import pickle
import time
from prefetch_generator import background

def pickle_load(filename):
    return pickle.load(open(filename,"rb"))

class Data:
    def __init__(self, classification=False, classes_text=None, batch_size=32, shuffle_buffer_size=4, prefetch_buffer_size=1, num_parallel_calls=4, num_parallel_readers=1):
        if classes_text:
            with open(classes_text) as f:
                self.class_names=[]
                self.ref_names=[]
                for lines in f.readlines():
                    arr = lines.strip().split(',')
                    self.ref_names.append(arr[0])
                    self.class_names.append(arr[-1])
        self.classification = classification
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.prefetch_buffer_size = prefetch_buffer_size
        self.num_parallel_calls = num_parallel_calls
        self.num_parallel_readers = num_parallel_readers

    def get_batch(self,filenames=None):
        return input_fn(self,filenames)

    def get_gen_batch(self,filedir,data_type='val'):
        self.get_oid(filedir,data_type)
        with tf.device('/cpu:0'):
            dataset = tf.data.Dataset.from_generator(self.get_generator(),
                    output_types=(
                        tf.float32,
                        tf.float32,
                        tf.int64,
                        tf.string,
                        tf.int64,
                        tf.int64),
                    output_shapes=(
                        tf.TensorShape([416,416,4]),
                        tf.TensorShape([4]),
                        tf.TensorShape([]),
                        tf.TensorShape([]),
                        tf.TensorShape([]),
                        tf.TensorShape([]))
                    )
            dataset = dataset.apply(tf.contrib.data.map_and_batch(
                map_func = _identity, batch_size=self.batch_size))
            dataset = dataset.prefetch(buffer_size=self.prefetch_buffer_size)
        return dataset

    def get_oid(self,filedir,data_type='train'):
        import csv
        self.images = []
        self.labels = []
        image_save_loc = filedir+'annotations/{}-images-saved.p'.format(data_type)
        labels_save_loc = filedir+'annotations/{}-labels-saved.p'.format(data_type)
        if os.path.exists(image_save_loc) and os.path.exists(labels_save_loc):
            self.images = pickle.load(open(image_save_loc,"rb"))
            self.labels = pickle.load(open(labels_save_loc,"rb"))
            self.num_images = self.images.shape[0]
            return 
        annotations_file = filedir+'annotations/{}-bbox.csv'.format(data_type)
        with open(annotations_file,'r') as csvfile:
            bbox_reader = csv.reader(csvfile,delimiter = ',')
            next(bbox_reader)
            num_lines = sum(1 for row in bbox_reader)
        with open(annotations_file,'r') as csvfile: 
            bbox_reader = csv.reader(csvfile,delimiter=',')
            next(bbox_reader)
            dict_annot={}
            pbar = tqdm(bbox_reader,total=num_lines)
            pbar.set_description('Reading Annotation')
            for elem in pbar:
                filename=elem[0]
                label = elem[2]
                xmin = float(elem[4])
                xmax = float(elem[5])
                ymin = float(elem[6])
                ymax = float(elem[7])
                label = self.ref_names.index(label)
                box = Box(x0=xmin,y0=ymin,x1=xmax,y1=ymax,label=label)
                if filename not in dict_annot.keys():
                    dict_annot[filename] = []
                    height=1
                    width=1
                dict_annot[filename].append([box.x0,box.y0,box.x1,box.y1,box.label])
            for filename in dict_annot.keys():
                image_name = filedir+data_type+'/'+filename+'.jpg'
                boxes = np.array(dict_annot[filename])
                self.images.append(image_name)
                self.labels.append(boxes)
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        pickle.dump(self.images, open(image_save_loc,"wb"))
        pickle.dump(self.labels, open(labels_save_loc,"wb"))
        self.num_images = self.images.shape[0]

    def get_generator(self):
        self.masked = np.array([None]*self.num_images)
        self.all_sorted_inds=np.array([None]*self.num_images)
        for i in range(self.masked.shape[0]):
            self.masked[i] = [False] * (self.labels[i].shape[0])
        image_index = np.arange(self.num_images)
        np.random.shuffle(image_index)
        def _generator():
            while True:
                ind = np.random.choice(image_index)
                img, box, c, img_name, random_ind = get_instance(self,ind)
                yield (np.array(img),np.array(box),c,img_name,ind, random_ind)
        return _generator



































