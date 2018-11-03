import numpy as np
import tensorflow as tf
from PIL import Image
import os
import math
import cv2
import random
from .bbox import *
import threading

def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _floats_feature(value):
    if not isinstance(value,list):
        value=[value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    if not isinstance(value,list):
        value=[value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def input_fn(self, filenames):
    files = tf.data.Dataset.list_files(filenames)
    #the following reads files in parallel and decrypts in parallel
    with tf.device('/cpu:0'):
        dataset = files.apply(tf.contrib.data.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=self.num_parallel_readers))

        dataset = dataset.shuffle(buffer_size = self.shuffle_buffer_size)

        #parallelize the parsing function and map and batch

        #dataset = dataset.map(map_func=parse_fn, num_parallel_calls=self.num_parallel_calls)
        #dataset = dataset.batch(batch_size=self.batch_size)
        if self.classification:
            dataset = dataset.apply(tf.contrib.data.map_and_batch(
                map_func=class_parse_fn,batch_size=self.batch_size))
        else:
            dataset = dataset.apply(tf.contrib.data.map_and_batch(
                map_func=box_parse_fn,batch_size=self.batch_size))
        dataset = dataset.prefetch(buffer_size=self.prefetch_buffer_size)
    return dataset

def box_parse_fn(example):
    #parse tfexample record
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

    # get number of boxes to use as random
    n_boxes = tf.maximum(tf.squeeze(parsed['image/object/number_of_boxes']) - tf.constant(1,dtype=tf.int64),tf.constant(1,dtype=tf.int64))

    #get max num boxes 
    max_boxes = parsed['image/object/max_boxes']
    max_boxes = tf.squeeze(max_boxes)
    max_boxes = tf.reshape(max_boxes,[1])

    #get random index
    box_ind = tf.random_uniform([], minval=0, maxval=n_boxes, dtype=tf.int64)

    #reshape
    x0 = tf.reshape(x0,max_boxes)
    y0 = tf.reshape(y0,max_boxes)
    x1 = tf.reshape(x1,max_boxes)
    y1 = tf.reshape(y1,max_boxes)
    labels = tf.reshape(labels,max_boxes)

    #slice index
    slice_ind = tf.stack([box_ind,])

    #slice the values for the box and the label
    xmin = tf.squeeze(tf.slice(x0,slice_ind,[1]))
    ymin = tf.squeeze(tf.slice(y0,slice_ind,[1]))
    xmax = tf.squeeze(tf.slice(x1,slice_ind,[1]))
    ymax = tf.squeeze(tf.slice(y1,slice_ind,[1]))
    label = tf.squeeze(tf.slice(labels,slice_ind,[1]))
    
    #get bbox values
    b_w = xmax - xmin
    b_h = ymax - ymin

    cx = xmin + tf.divide(b_w,tf.constant(2,tf.float32))
    cy = ymin + tf.divide(b_h,tf.constant(2,tf.float32))

    bbox = tf.stack([cx,cy,b_w,b_h])
    
    #decode image
    image = tf.image.decode_jpeg(parsed['image/image_raw'],channels=3)
    image_shape = tf.shape(image)
    image_h = image_shape[1]
    image_w = image_shape[2]
    #reshape image
    image = tf.image.resize_images(image,[416,416],align_corners=True)
    image = tf.image.convert_image_dtype(image,tf.float32)
    
    #get mask
    mask = tf.py_func(get_mask,[box_ind, x0,y0, x1, y1], tf.float32)
    image = image/tf.constant(255.0,tf.float32)
    image = tf.concat([image,mask],axis=-1)
    image = tf.reshape(image,(416,416,4))

    return image, label, bbox


def class_parse_fn(example):
    #parse tfexample record
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

    # get number of boxes to use as random
    n_boxes = tf.maximum(tf.squeeze(parsed['image/object/number_of_boxes']) - tf.constant(1,dtype=tf.int64),tf.constant(1,dtype=tf.int64))

    #get max num boxes 
    max_boxes = parsed['image/object/max_boxes']
    max_boxes = tf.squeeze(max_boxes)
    max_boxes = tf.reshape(max_boxes,[1])

    #get random index
    box_ind = tf.random_uniform([], minval=0, maxval=n_boxes, dtype=tf.int64)

    #reshape
    x0 = tf.reshape(x0,max_boxes)
    y0 = tf.reshape(y0,max_boxes)
    x1 = tf.reshape(x1,max_boxes)
    y1 = tf.reshape(y1,max_boxes)
    labels = tf.reshape(labels,max_boxes)

    #slice index
    slice_ind = tf.stack([box_ind,])

    #slice the values for the box and the label
    xmin = tf.squeeze(tf.slice(x0,slice_ind,[1]))
    ymin = tf.squeeze(tf.slice(y0,slice_ind,[1]))
    xmax = tf.squeeze(tf.slice(x1,slice_ind,[1]))
    ymax = tf.squeeze(tf.slice(y1,slice_ind,[1]))
    label = tf.squeeze(tf.slice(labels,slice_ind,[1]))

    image = tf.image.decode_jpeg(parsed['image/image_raw'],channels=3)
    image = tf.image.resize_images(image,[416,416],align_corners=True)
    
    initial_width = tf.to_float(tf.shape(image)[1])
    initial_height = tf.to_float(tf.shape(image)[0])

    #make mask
    mask = tf.py_func(get_mask,[box_ind,x0,y0,x1,y1],tf.float32)

    xmin = xmin * initial_width
    ymin = ymin * initial_height
    xmax = xmax * initial_width
    ymax = ymax * initial_height

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

    scale_factor_height = (tf.to_float(tf.constant(200)) / tf.to_float(h))
    scale_factor_width = (tf.to_float(tf.constant(200)) / tf.to_float(w))

    scale_factor = tf.minimum(scale_factor_height,scale_factor_width)
    scale_height_const = tf.to_int32(scale_factor * tf.to_float(h))
    scale_width_const = tf.to_int32(scale_factor * tf.to_float(w))

    image = tf.image.resize_images(image,[scale_height_const,scale_width_const],align_corners=True)
    mask = tf.image.resize_images(mask,[scale_height_const,scale_width_const],align_corners=True)


    image = tf.image.resize_image_with_pad(image,200,200)
    mask = tf.image.resize_image_with_pad(mask,200,200)

    image = tf.image.convert_image_dtype(image,tf.float32)
    image = image/tf.constant(255.0,tf.float32)

    mask = tf.cast(mask,tf.float32)
    image = tf.concat([image,mask],axis=-1)
    label = tf.cast(label,tf.int32)

    image = tf.reshape(image,(200,200,4))

    return image, label


def get_mask(box_ind, xmin,ymin,xmax,ymax):
    mask = np.ones((416,416,1),np.float32)
    areas = np.trim_zeros((xmax-xmin)*(ymax-ymin))
    sorted_inds = np.argsort(areas)

    for i in sorted_inds:
        if i==box_ind:
            break
        x0 = int(np.floor(xmin[i] * 416))
        y0 = int(np.floor(ymin[i] * 416))
        x1 = int(np.floor(xmax[i] * 416))
        y1 = int(np.floor(ymax[i] * 416))
        mask[y0:y1,x0:x1] = 0.0
    return mask


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



def get_instance(self,ind):
    img = self.images[ind]
    labels = self.labels[ind]
    boxes = np.zeros((labels.shape[0],4),np.float32)
    classes = []
    for e, box in enumerate(labels):
        boxes[e,0] = box.x0
        boxes[e,1] = box.y0
        boxes[e,2] = box.x1
        boxes[e,3] = box.y1
        classes.append(box.label)

    classes = np.array(classes)
    areas = np.trim_zeros((boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]))
    sorted_inds = np.argsort(areas)

    self.masked[ind][sorted_inds[0]] = True
    rand_int = random.randint(0,sorted_inds.shape[0]-1)
    while self.masked[ind][sorted_inds[rand_int]] != True:
        rand_int -= 1
    self.masked[ind][sorted_inds[rand_int+1]] = True
                
    img = Image.open(img).convert('RGB')
    img = img.resize((416,416),resample=Image.BILINEAR)
    img = np.array(img,dtype=np.float32)
    
    if img.max() > 1:
        img /= 255.0
    box = boxes[sorted_inds[rand_int]]
    box *= 416.0
    b_w = box[2] - box[0]
    b_h = box[3] - box[1]
    cx = box[0] + (b_w/2.)
    cy = box[1] + (b_h/2.)
    box = [cx,cy,b_w,b_h]
    c = classes[sorted_inds[rand_int]]
    mask = generator_masks(img,sorted_inds[rand_int])
    img = np.concatenate((img,mask),axis=-1)

    return img, box, c

def generator_masks(img,ind):
    if ind == 0:
        return np.zeros((416,416,1),dtype=np.float32)
    path = os.path.abspath(os.path.join(os.path.dirname(img),'..','masks'))
    path = os.path.join(path,os.path.splitext(os.path.basename(img))[0]+'_'+str(ind)+'.npy')
    if not os.path.exists(path):
        raise ValueError("Problem during reading of masks, cannot find mask at path:\n"+str(path))
    return np.load(path)

def generator(self): 
    self.masked = np.array([None]*self.num_images)
    for i in range(masked.shape[0]):
        self.masked[i] = [False] * (self.labels[i].shape[0])
    image_index = np.arange(self.num_images)
    np.random.shuffle(image_index) 
    def _generator():
        while True:
            imgs, boxes, cs = [],[],[]
            for _ in range(self.batch_size):
                ind = np.random.choice(image_index)
                img, box, c = get_instance(self,ind)
                imgs.append(img)
                boxes.append(box)
                cs.append(c)
            yield (np.array(imgs),np.array(boxes),np.array(cs))

    return _generator


