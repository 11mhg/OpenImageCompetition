from __future__ import print_function

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
from augmenter import *
import os
import math
import threading
import cv2



def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _floats_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _process_image_files(images,labels,max_boxes,image_size,filename,shard_index):
    print("Starting",flush=True,end='\r')
    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in range(images.shape[0]):
            image_name = images[index][0]
            try:
                image = open(image_name,'rb').read()
            except:
                continue
            height = float(images[index][1])
            width = float(images[index][2])
            xmin = []
            xmax = []
            ymin = []
            ymax = []
            label = []
            #get the xyxy components of box
            boxes = labels[index]
            for i in range(max_boxes):
                if i < boxes.shape[0]:
                    box = boxes[i]
                    b = box.xyxy
                    xmin.append(b[0]/width)
                    ymin.append(b[1]/height)
                    xmax.append(b[2]/width)
                    ymax.append(b[3]/height)
                    label.append(box.label)
                else:
                    xmin.append(0)
                    ymin.append(0)
                    xmax.append(0)
                    ymax.append(0)
                    label.append(-1)
                    
            #Do your heavy image augmentation here (rotation, gaussian, etc)
            #image scale padding can be done later on
                
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image/object/bbox/xmin': _floats_feature(xmin),
                        'image/object/bbox/ymin': _floats_feature(ymin),
                        'image/object/bbox/xmax': _floats_feature(xmax),
                        'image/object/bbox/ymax': _floats_feature(ymax),
                        'image/object/bbox/labels': _floats_feature(label),
                        'image/object/number_of_boxes': _int64_feature(max_boxes),
                        'image/image_raw': _bytes_feature(tf.compat.as_bytes(image))
                    }))
            writer.write(example.SerializeToString())
            print(index/images.shape[0]*100,flush=True,end='\r')
        print("Done",flush=True,end='\r')

def convert_to(self,directory, name, image_size = (800,800), num_shards = 1):
    filenames = [os.path.join(directory, name+'_{}'.format(i)+'.tfrecords') for i in range(num_shards)]

    coord = tf.train.Coordinator()
    threads = []
    for i in range(num_shards):
        filename = filenames[i]
        elem_per_shard = int(math.ceil((self.num_examples/num_shards)))
        start_ndx = i * elem_per_shard
        end_ndx = min((i+1)*elem_per_shard,self.num_examples)
        images = self.images[start_ndx:end_ndx-1]
        labels = self.labels[start_ndx:end_ndx-1]
        args = (images, labels, self.max_boxes,image_size,filename,i)
        t = threading.Thread(target=_process_image_files,args=args)
        t.start()
        threads.append(t)
    coord.join(threads)
    print("Done")



def input_fn(self,filedir):
    files = tf.data.Dataset.list_files(filedir)
    #dataset = files.interleave(tf.data.TFRecordDataset,cycle_length=self.num_parallel_readers)
    #The following reads files in parallel and decrypts in parallel (may be useful)
    #It replaces the interleave above
    with tf.device('/cpu:0'):
        dataset = files.apply(tf.contrib.data.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=self.num_parallel_readers))

        dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size)
    #parallelize the parsing function 
    #dataset = dataset.map(map_func=parse_fn, num_parallel_calls=self.num_parallel_calls)
    #dataset = dataset.batch(batch_size=self.batch_size)

    #The following may be useful for large batch sizes also replaces map and batch above
        dataset = dataset.apply(tf.contrib.data.map_and_batch(
            map_func=parse_fn, batch_size=self.batch_size))
    
    #The following prefetches batches for the next batch and readies them
        dataset = dataset.prefetch(buffer_size=self.prefetch_buffer_size)
    return dataset

def _augment_helper(image):
    return image

def parse_fn(example):
    #Parse TFExample record and do data aug
    example_fmt = {
        'image/image_raw': tf.FixedLenFeature([],tf.string,""),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/number_of_boxes': tf.FixedLenFeature([],dtype=tf.int64),
        'image/object/bbox/labels': tf.VarLenFeature(dtype=tf.float32),
    }
    parsed = tf.parse_single_example(example, example_fmt)
    image = tf.image.decode_jpeg(parsed['image/image_raw'])
    xmin = tf.expand_dims(parsed['image/object/bbox/xmin'].values,0)
    ymin = tf.expand_dims(parsed['image/object/bbox/ymin'].values,0)
    xmax = tf.expand_dims(parsed['image/object/bbox/xmax'].values,0)
    ymax = tf.expand_dims(parsed['image/object/bbox/ymax'].values,0)
    labels = tf.expand_dims(parsed['image/object/bbox/labels'].values,0)
    n_boxes = parsed['image/object/number_of_boxes']
    
    bbox = tf.concat(axis=0, values = [xmin, ymin, xmax, ymax])
    bbox = tf.transpose(bbox,[1,0])
    
    labels = tf.cast(labels,dtype=tf.int32)
    labels = tf.transpose(labels,[1,0]) 
    
    bbox_shape = tf.stack([n_boxes, tf.constant(4,dtype=tf.int64)])
    bbox = tf.reshape(bbox,bbox_shape)
    
    labels_shape = tf.stack([n_boxes,tf.constant(1,dtype=tf.int64)])
    labels = tf.reshape(labels,labels_shape)
    
    #should already be at 800x800x3 
    image = tf.image.resize_bilinear(image,[800,800],align_corners=True)
    image = tf.image.convert_image_dtype(image,tf.float16)

    return image, labels, bbox
    
