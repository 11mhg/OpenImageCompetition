import numpy as np
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
import threading
import os
import math
import cv2


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _process_image_files(images, labels, max_boxes, image_size, filename):
    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in range(images.shape[0]):
            print(index/images.shape[0]*100,flush=True,end='\r')
            image_name = images[index][0]
            try:
                image = open(image_name,'rb').read()
            except:
                continue
            height = float(images[index][1])
            width = float(images[index][2])         
            #get the xyxy components of box
            boxes = labels[index]
            for i in range(boxes.shape[0]):
                box = boxes[i]
                b = box
                    
                b.calculate_cxcy(b.x0,b.y0,b.x1,b.y1)
                cx = b.cx * width
                cy = b.cy * height
                w = b.w * width
                h = b.h * height

                size = h if h>w else w
                    
                    
                area = (int(cx - (size/2.)),int(cy - (size/2.)), int(cx+(size/2.)),int(cy+(size/2.)))
                label = int(box.label)
                #Do your heavy image augmentation here (rotation, gaussian, etc)
                #image scale padding can be done later on
            
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'image/label': _int64_feature(label),
                            'image/box' : _int64_feature(bbox),
                            'image/image_raw': _bytes_feature(tf.compat.as_bytes(image))
                    }))
                writer.write(example.SerializeToString())
           
def convert_to(self,directory, name, image_size = (224,224), num_shards = 1):
    filenames = [os.path.join(directory, name+'_{}'.format(i)+'.tfrecords') for i in range(num_shards)]

    #should paralellize, look at parallelize shard writing tfrecord on google
    coord = tf.train.Coordinator()
    threads=[]

    for i in range(num_shards):
        filename = filenames[i]
        elem_per_shard = int(math.ceil((self.num_examples/num_shards)))
        start_ndx = i*elem_per_shard
        end_ndx = min((i+1)*elem_per_shard,self.num_examples)
        images = self.images[start_ndx:end_ndx-1]
        labels = self.labels[start_ndx:end_ndx-1]
        args = (images,labels, self.max_boxes, image_size, filename)
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

    num_boxes = tf.maximum(tf.squeeze(parsed['image/object/number_of_boxes']) - tf.constant(1,dtype=tf.int64),tf.constant(1,dtype=tf.int64))

    max_boxes = tf.squeeze(parsed['image/object/max_boxes'])
    max_boxes = tf.reshape(max_boxes,[1])

    x0 = tf.reshape(x0,max_boxes)
    y0 = tf.reshape(y0,max_boxes)
    x1 = tf.reshape(x1,max_boxes)
    y1 = tf.reshape(y1,max_boxes)
    labels = tf.reshape(labels,max_boxes)

    box_ind = tf.random_uniform([],minval=0, maxval= num_boxes, dtype=tf.int64)
    slice_b = tf.stack([box_ind,])
    xmin = tf.squeeze(tf.slice(x0,slice_b,[1]))
    ymin = tf.squeeze(tf.slice(y0,slice_b,[1]))
    xmax = tf.squeeze(tf.slice(x1,slice_b,[1]))
    ymax = tf.squeeze(tf.slice(y1,slice_b,[1]))
    label = tf.squeeze(tf.slice(labels,slice_b,[1]))
   
    image = tf.image.decode_jpeg(parsed['image/image_raw'],
                                          channels=3)    
    
    
    initial_width = tf.to_float(tf.shape(image)[1])
    initial_height = tf.to_float(tf.shape(image)[0])
    
    #make mask
    mask = tf.py_func(get_mask,[box_ind,x0,y0,x1,y1,initial_width,initial_height],tf.float16)

    xmin = tf.multiply(xmin,initial_width)
    ymin = tf.multiply(ymin,initial_height)
    xmax = tf.multiply(xmax,initial_width)
    ymax = tf.multiply(ymax,initial_height)

    w = xmax - xmin
    h = ymax - ymin

    xmin = tf.cast(xmin,tf.int32)
    ymin = tf.cast(ymin,tf.int32)
    w = tf.maximum(tf.cast(w,tf.int32),tf.constant(1,dtype=tf.int32))
    h = tf.maximum(tf.cast(h,tf.int32),tf.constant(1,dtype=tf.int32))
    
    
    image = tf.image.crop_to_bounding_box(image,
                                          ymin,
                                          xmin,
                                          h,
                                          w)
    mask = tf.image.crop_to_bounding_box(mask,
                                         ymin,
                                         xmin,
                                         h,
                                         w)
    image = tf.image.resize_images(image,[200,200],align_corners=True,preserve_aspect_ratio=True)
    mask = tf.image.resize_images(mask,[200,200],align_corners=True,preserve_aspect_ratio=True)

    image = tf.image.convert_image_dtype(image,tf.float16)
    image = image/tf.constant(255.0,tf.float16)
    mask = tf.cast(mask,tf.float16)
    image = tf.concat([image,mask],axis=-1)
    label = tf.cast(label,tf.int32)

    image = tf.reshape(image,(200,200,4))
    
    
    return image, label

def get_mask(box_ind,xmin,ymin,xmax,ymax,initial_width,initial_height):
    height = int(initial_height)
    width = int(initial_width)
    mask = np.ones((height,width,1),np.float16)
    areas = np.trim_zeros((xmax-xmin)*(ymax-ymin))
    sorted_inds = np.argsort(areas)
    for i in sorted_inds:
        if i==box_ind:
            break
        x0 = int(np.floor(xmin[i]*width))
        y0 = int(np.floor(ymin[i]*height))
        x1 = int(np.floor(xmax[i]*width))
        y1 = int(np.floor(ymax[i]*height))
        mask[y0:y1,x0:x1] = 0.0
    return mask






