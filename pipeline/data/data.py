from __future__ import print_function

import numpy as np
import tensorflow as tf
from .bbox import *
import os
from tqdm import tqdm
from .tfrecord_utils import input_fn, _identity, generator_masks, get_instance

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
                        tf.int64),
                    output_shapes=(
                        tf.TensorShape([self.batch_size,416,416,4]),
                        tf.TensorShape([self.batch_size,4]),
                        tf.TensorShape([self.batch_size,1]),
                        tf.TensorShape([self.batch_size,1]),
                        tf.TensorShape([self.batch_size,1]))
                    )
            dataset = dataset.apply(tf.contrib.data.map_and_batch(
                map_func = _identity, batch_size=self.batch_size))
            dataset = dataset.prefetch(buffer_size=self.prefetch_buffer_size)
        return dataset


    def get_oid(self,filedir,data_type='train'):
        import csv
        self.images = []
        self.labels = []
        annotations_file = filedir+'annotations/{}-bbox.csv'.format(data_type)
        with open(annotations_file,'r') as csvfile:
            bbox_reader = csv.reader(csvfile,delimiter = ',')
            next(bbox_reader)
            dict_annot={}
            pbar = tqdm(bbox_reader)
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
                dict_annot[filename].append(box)
            for filename in dict_annot.keys():
                image_name = filedir+data_type+'/'+filename+'.jpg'
                boxes = np.array(dict_annot[filename])
                self.images.append(image_name)
                self.labels.append(boxes)
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        self.num_images = self.images.shape[0]

    def get_generator(self):
        self.masked = np.array([None]*self.num_images)
        for i in range(self.masked.shape[0]):
            self.masked[i] = [False] * (self.labels[i].shape[0])
        image_index = np.arange(self.num_images)
        np.random.shuffle(image_index)
        def _generator():
            while True:
                imgs, boxes, cs, img_names, indices = [],[],[],[],[]
                for _ in range(self.batch_size):
                    ind = np.random.choice(image_index)
                    img, box, c, img_name, index = get_instance(self,ind)
                    imgs.append(img)
                    boxes.append(box)
                    cs.append(c)
                    img_names.append(img_name)
                    indices.append(index)
                yield (np.array(imgs),np.array(boxes),np.array(cs),np.array(img_names),np.array(indices))
        return _generator
