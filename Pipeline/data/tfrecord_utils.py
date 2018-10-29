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
from elasticsearch import Elasticsearch, client
import base64
import random
import itertools

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
            xmax = np.array(xmax)
            xmin = np.array(xmin)
            ymax = np.array(ymax)
            ymin = np.array(ymin)

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image/object/bbox/xmin': _floats_feature(xmin.tolist()),
                        'image/object/bbox/ymin': _floats_feature(ymin.tolist()),
                        'image/object/bbox/xmax': _floats_feature(xmax.tolist()),
                        'image/object/bbox/ymax': _floats_feature(ymax.tolist()),
                        'image/object/bbox/labels': _floats_feature(label),
                        'image/object/number_of_boxes': _int64_feature(boxes.shape[0]),
                        'image/object/max_boxes': _int64_feature(max_boxes),
                        'image/image_raw': _bytes_feature(tf.compat.as_bytes(image)),
                    }))
            writer.write(example.SerializeToString())
            print(index/images.shape[0]*100,flush=True,end='\r')
        print("")
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
        'image/object/max_boxes': tf.FixedLenFeature([],dtype=tf.int64),
        'image/object/bbox/labels': tf.VarLenFeature(dtype=tf.float32),
    }
    parsed = tf.parse_single_example(example, example_fmt)

    x0 = tf.expand_dims(parsed['image/object/bbox/xmin'].values,0)
    y0 = tf.expand_dims(parsed['image/object/bbox/ymin'].values,0)
    x1 = tf.expand_dims(parsed['image/object/bbox/xmax'].values,0)
    y1 = tf.expand_dims(parsed['image/object/bbox/ymax'].values,0)
    labels = tf.expand_dims(parsed['image/object/bbox/labels'].values,0)
    


    n_boxes = tf.maximum(tf.squeeze(parsed['image/object/number_of_boxes']) - tf.constant(1,dtype=tf.int64),tf.constant(1,dtype=tf.int64))
    
    max_boxes = parsed['image/object/max_boxes']
    max_boxes = tf.squeeze(max_boxes)
    max_boxes = tf.reshape(max_boxes,[1])

    box_ind = tf.random_uniform([],minval=0,maxval=n_boxes,dtype=tf.int64)
    
    x0 = tf.reshape(x0,max_boxes)
    y0 = tf.reshape(y0,max_boxes)
    x1 = tf.reshape(x1,max_boxes)
    y1 = tf.reshape(y1,max_boxes)
    labels = tf.reshape(labels,max_boxes)

    box_ind = tf.random_uniform([],minval=0,maxval=n_boxes,dtype=tf.int64)
    slice_ind = tf.stack([box_ind,])
    xmin = tf.squeeze(tf.slice(x0,slice_ind,[1]))
    ymin = tf.squeeze(tf.slice(y0,slice_ind,[1]))
    xmax = tf.squeeze(tf.slice(x1,slice_ind,[1]))
    ymax = tf.squeeze(tf.slice(y1,slice_ind,[1]))
    label = tf.squeeze(tf.slice(labels,slice_ind,[1]))
    
    b_w = xmax - xmin
    b_h = ymax - ymin
    cx = xmin + tf.divide(b_w,tf.constant(2,tf.float32))
    cy = ymin + tf.divide(b_h,tf.constant(2,tf.float32))

    bbox = tf.stack([cx,cy,b_w,b_h])
    image = tf.image.decode_jpeg(parsed['image/image_raw'],channels=3)
    image_shape = tf.shape(image)
    image_h = image_shape[1]
    image_w = image_shape[2]
    image = tf.image.resize_images(image,[416,416],align_corners=True)
    image = tf.image.convert_image_dtype(image,tf.float16)

    #make mask   
    mask = tf.py_func(get_mask,[box_ind, x0,y0,x1,y1], tf.float16)
    image = image/tf.constant(255.0,tf.float16)
    image = tf.concat([image,mask],axis=-1)
    image = tf.reshape(image,(416,416,4)) 
    return image, label, bbox
   
def get_mask(box_ind, xmin,ymin,xmax,ymax):
    mask = np.ones((416,416,1),np.float16)
    areas = np.trim_zeros((xmax-xmin)*(ymax-ymin))
    sorted_inds = np.argsort(areas)
    for i in sorted_inds:
        if i==box_ind:
            break
        x0 = int(np.floor(xmin[i]*416))
        y0 = int(np.floor(ymin[i]*416))
        x1 = int(np.floor(xmax[i]*416))
        y1 = int(np.floor(ymax[i]*416))
        mask[y0:y1,x0:x1] = 0.0
    return mask


def _get_es_batch():
    global d_type
    global ind
    name = ind + '_'+d_type  
    page = es.search(
                index = name,
                doc_type = 'train',
                scroll = '3m',
                size = 1,
                body = {
                    'query':{'match_all':{}}
                })

    sid = page['_scroll_id']
    scroll_size = page['hits']['total']
    print(scroll_size)
    print(name)
    while(scroll_size > 0 ):
        page = es.scroll(scroll_id = sid, scroll='3m')
        sid = page['_scroll_id']
        results = page['hits']['hits']
        scroll_size = len(results)
        image = base64.b64decode(results[0]['_source']['image'])
        label = results[0]['_source']['label']
        masks = base64.b64decode(results[0]['_source']['masks'])
        masks = np.frombuffer(masks,dtype=float16)
        masks = np.reshape(masks,(800,800,1))


        xmin = results[0]['_source']['xmin']
        ymin = results[0]['_source']['ymin']
        xmax = results[0]['_source']['xmax']
        ymax = results[0]['_source']['ymax']
        
        random.seed(7)
        
        index_val = random.randint(0,len(xmin)-1)

        mask = masks[:,:,index_val]
        xmin = xmin[index_val]
        ymin = ymin[index_val]
        xmax = xmax[index_val]
        ymax = ymax[index_val]

        label = label[index_val]

        yield image, label, mask, xmin, ymin, xmax, ymax 

def _preprocess_es(image, label, mask, xmin, ymin, xmax, ymax): 
    image = tf.image.decode_jpeg(image,channels=3)
    mask = tf.reshape(mask,[800,800,1]) 
    image = tf.image.resize_images(image,[416,416],align_corners=True)
    mask = tf.image.resize_images(mask,[416,416],align_corners=True)
    inp = tf.concat([image,mask],axis=-1)
   
    b_w = xmax - xmin
    b_h = ymax - ymin
    cx = xmin + tf.divide(b_w,tf.constant(2,tf.float32))
    cy = ymin + tf.divide(b_h,tf.constant(2,tf.float32))

    bbox = tf.stack([cx,cy,b_w,b_h])
    inp = tf.cast(inp,tf.float16)

    return inp, label, bbox


es = Elasticsearch()
d_type = 'train'
ind = 'open_image'
def es_input_fn(self,filenames):
    global d_type
    d_type = filenames
    with tf.device('/cpu:0'):
        dataset = tf.data.Dataset.from_generator(_get_es_batch,
                output_types=(
                    tf.string,
                    tf.int64,
                    tf.float16,
                    tf.float32,
                    tf.float32,
                    tf.float32,
                    tf.float32),
                output_shapes=(
                    tf.TensorShape([]),
                    tf.TensorShape([]),
                    tf.TensorShape([800,800,1]),
                    tf.TensorShape([]),
                    tf.TensorShape([]),
                    tf.TensorShape([]),
                    tf.TensorShape([]))
                )
        dataset = dataset.apply(tf.contrib.data.map_and_batch(
            map_func=_preprocess_es, batch_size=self.batch_size))
        dataset = dataset.prefetch(buffer_size=self.prefetch_buffer_size)
    return dataset
