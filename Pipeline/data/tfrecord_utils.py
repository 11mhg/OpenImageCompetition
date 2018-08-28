import numpy as np
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
import threading
import os
import math
import cv2
import base64
import elasticsearch
from elasticsearch import Elasticsearch,client,helpers

es = Elasticsearch()



def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _floats_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _process_image_files(images, labels, max_boxes, image_size, filename):
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
            example = tf.train.Example(
                   features=tf.train.Features(
                       feature={
                           'image/object/bbox/xmin': _floats_feature(xmin),
                           'image/object/bbox/ymin': _floats_feature(ymin),
                           'image/object/bbox/xmax': _floats_feature(xmax),
                           'image/object/bbox/ymax': _floats_feature(ymax),
                           'image/object/bbox/labels': _floats_feature(label),
                           'image/object/max_boxes': _int64_feature(int(boxes.shape[0])),
                           'image/object/number_of_boxes': _int64_feature(max_boxes),
                           'image/image_raw': _bytes_feature(tf.compat.as_bytes(image))
                        }))
            writer.write(example.SerializeToString())
            print(index/images.shape[0]*100,flush=True,end='\r')
        print('')
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

def _es_data():
	global ind_name
	page = helpers.scan(es,
			index = ind_name,
			doc_type='train',
			scroll = '2m',
			size=100,
			query = {
				'query':{'match_all':{}}
			},
			sort = ['_doc'],
			request_timeout = 10000
	)
	np.random.seed(10)
	for i in page:
		xmin = i['_source']['xmin']
		ymin = i['_source']['ymin']
		xmax = i['_source']['xmax']
		ymax = i['_source']['ymax']
		image = i['_source']['image']
		mask = i['_source']['mask']
		labels = i['_source']['label']
		
		mask = np.frombuffer(base64.b64decode(mask),dtype=np.uint8).reshape(800,800,-1)

		rand_slice = np.random.randint(0,len(labels))
		xmin = xmin[rand_slice]
		ymin = ymin[rand_slice]
		xmax = xmax[rand_slice]
		ymin = ymax[rand_slice]
		label = labels[rand_slice]
		mask = mask[:,:,rand_slice]

		image = base64.b64decode(image)
		image = cv2.imdecode(np.fromstring(image,dtype=np.uint8),1)
		height, width, channels = image.shape
		
		xmin =int(xmin*width*width)
		ymin =int(ymin*height*height)
		xmax =int(xmax*width*width)
		ymax = int(ymax*height*height)
		w = float(xmax - xmin)
		h = float(ymax - ymin)
		
		max_wh = max(w,h)
		r_w = w/max_wh
		r_h = h/max_wh

		new_w = int(r_w * 200)
		new_h = int(r_h * 200)

		crop_img = image[ymin:ymax,xmin:xmax,:]
		crop_img = cv2.resize(crop_img, (new_w,new_h))

		yield crop_img, mask, label
		
def es_map(crop_img, mask, label):
	img = tf.image.convert_image_dtype(crop_img,tf.float32)
	mask = tf.image.convert_image_dtype(mask,tf.float32)
	
	img = tf.image.resize_image_with_crop_or_pad(img,200,200)
	mask = tf.image.resize_images(mask,[200,200],align_corners=True)

	image = tf.concat([img,mask],axis=-1)

	return image, label

ind_name = None

def es_input_fn(self,es_name='open_image_',data_type='train'):
	global ind_name
	ind_name = es_name+data_type
	output_types = (tf.uint8,tf.uint8, tf.int64)
	with tf.device('/cpu:0'):
		dataset = tf.data.Dataset.from_generator(
					_get_es_batch,
					output_types = output_types)
		dataset = dataset.apply(tf.contrib.data.map_and_batch(
					map_func = es_map, batch_size = self.batch_size))

		dataset = dataset.prefetch(buffer_size = self.prefetch_buffer_size)
	return dataset
					
		
	
	



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
        'image/object/bbox/labels': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmin':tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin':tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax':tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax':tf.VarLenFeature(dtype=tf.float32),
        'image/object/number_of_boxes':tf.FixedLenFeature([],dtype=tf.int64),
        'image/object/max_boxes':tf.FixedLenFeature([],tf.int64,1)
    }
    parsed = tf.parse_single_example(example, example_fmt)
    image = tf.image.decode_jpeg(parsed['image/image_raw'],channels=3)
    labels = parsed['image/object/bbox/labels'].values
    xmin = parsed['image/object/bbox/xmin'].values
    ymin = parsed['image/object/bbox/ymin'].values
    xmax = parsed['image/object/bbox/xmax'].values
    ymax = parsed['image/object/bbox/ymax'].values

    max_boxes = parsed['image/object/max_boxes']
 
    slice_ind = tf.random_uniform([],minval=0,maxval=max_boxes,dtype=tf.int64)  

    x0 = xmin[slice_ind]
    y0 = ymin[slice_ind]
    x1 = xmax[slice_ind]
    y1 = ymax[slice_ind]

    label = tf.stack([x0,y0,x1,y1])
    label = label * tf.constant(2,tf.float32) 
    label = label - tf.constant(1,tf.float32)


    mask = tf.py_func(get_mask,[slice_ind,xmin,ymin,xmax,ymax],tf.uint8)
    mask = tf.reshape(mask,(800,800,1))
    mask = tf.image.convert_image_dtype(mask,tf.float32)
    mask = tf.image.resize_images(mask,[416,416],align_corners=True)
    image = tf.image.resize_images(image,[416,416],align_corners=True)

    image = tf.concat([image,mask],axis=-1)

    return image, label 


def get_mask(slice_ind, xmin, ymin, xmax, ymax):
    mask = np.zeros((800,800,1),np.uint8)
    areas = np.trim_zeros((xmax-xmin)*(ymax-ymin))
    sorted_inds = np.argsort(areas)
    for i in sorted_inds:
        if i == slice_ind:
            break
        x0 = int(np.floor(xmin[i]*800))
        x1 = int(np.floor(xmax[i]*800))
        y0 = int(np.floor(ymin[i]*800))
        y1 = int(np.floor(ymax[i]*800))
        mask[y0:y1,x0:x1] = 255
    return mask
