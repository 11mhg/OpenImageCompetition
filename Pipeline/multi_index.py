import csv
import numpy as np
import tensorflow as tf
import os
import math
import sys
import cv2
import base64
import elasticsearch
import threading
import datetime

from multiprocessing import Process, Queue
from data.bbox import Box
from tqdm import tqdm
from data.tfrecord_utils import convert_to, input_fn
from collections import defaultdict
from collections import OrderedDict
from PIL import Image
from augmenter import *
from elasticsearch import Elasticsearch,client ,helpers
es = Elasticsearch()

proc_images = Queue(3)
conv_images = Queue(7)

print datetime.datetime.now()
def do_work(in_queue):
    while True:
        actions = in_queue.get()
        
        #print "INDEX QUEUE: ", in_queue.qsize()

def do_proc_image(in_queue):
    while True:
        images,labels,max_boxes = in_queue.get()
        for index in range(images.shape[0]):
            image_name = images[index][0]
            try:
                image = base64.b64encode(open(image_name,'rb').read())
            except Exception as e:
                print(e)
                input("Wait")
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

            sto = (image, image_name, xmin ,ymin ,xmax ,ymax ,label)
            conv_images.put(sto)
	    #print "PROC QUEUE: ",in_queue.qsize()
def do_covert(in_queue):
    actions = []
    while True:
        image, image_name, xmin ,ymin ,xmax ,ymax ,label = in_queue.get()
        image_id = os.path.splitext(os.path.basename(image_name))[0]
        
        hold_mask = np.zeros((800,800),dtype=np.uint8)
        m_xmin = (np.array(xmin)*800)
        m_xmax = (np.array(xmax)*800)
        m_ymin = (np.array(ymin)*800)
        m_ymax = (np.array(ymax)*800)
        areas = np.trim_zeros((m_xmax - m_xmin) * (m_ymax - m_ymin))
        sorted_inds = np.argsort(areas)

        masks = np.zeros((800,800,len(sorted_inds)),dtype=np.uint8)
        for i in sorted_inds:
            b = int(np.floor(m_xmin[i]))
            c = int(np.floor(m_xmax[i]))
            d = int(np.floor(m_ymin[i]))
            e = int(np.floor(m_ymax[i]))
            masks[:,:,i] = hold_mask[:,:]
            hold_mask[d:e,b:c] = 1

        actions.append({
            '_index':'open_image_val',
			'_type':'train',
            'image': str(image),
            'label' : label,
            'xmin':xmin,
            'ymin':ymin,
            'xmax':xmax,
            'ymax':ymax,
            'id ':image_id,
            'mask':str(base64.b64encode(masks.tobytes()))
        })
        if len(actions) == 500:
            try:
                helpers.bulk(es,actions,request_timeout=10000)
            except:
                print 'error'
            actions = []
        print "CONV QUEUE: ", in_queue.qsize()
class Data:
    def __init__(self, classes_text, image_size=(800,800,3)):
        with open(classes_text) as f:
            self.class_names = []
            for lines in f.readlines():
                arr = lines.strip().split(',')
                self.class_names.append(arr[-1])
        self.image_size = image_size
    

class PreProcessData:
    def __init__(self,classes_text='./dummy_labels.txt',image_size=(800,800)):
        self.image_size=image_size
        self.bulk_ind = 100
        with open(classes_text) as f:
            self.class_names = []
            for lines in f.readlines():
                arr = lines.strip().split(',')
                self.class_names.append(arr[0])
        for i in range(1):
            Process(target=do_proc_image,args=(proc_images,)).start()

        for i in range(7):
            Process(target=do_covert,args=(conv_images,)).start()
        #for i in range(1):
         #   Process(target=do_thread).start()
        #for i in xrange(10):
       #     threading.Thread(target=do_work,args=(work,)).start()

    def get_open_images(self,filedir,data_type='train',name="OpenImage"):
        self.ind_name = 'open_image_'+data_type
        try:
            create_index(name = self.ind_name)
        except:
            pass
        
        self.name='OpenImages'+'-'+data_type

        images = []
        labels = []
        max_boxes = 0
        annotations_file = 'missing_val.csv'
        #annotations_file = filedir+'annotations/' + '{}-bbox.csv'.format(data_type)
        with open(annotations_file,'r') as csvfile:
            bbox_reader = csv.reader(csvfile,delimiter=',')
            print("Open Images contains a large number of files, do not be discourage if it takes a long time.")
            next(bbox_reader)
            dict_annot = defaultdict(list)
            pbar = tqdm(bbox_reader)
            pbar.set_description("Reading Annotations")
            for j,elem in enumerate(pbar):
                filename = elem[0]
                label = elem[2]
                xmin = float(elem[4])
                xmax = float(elem[5])
                ymin = float(elem[6])
                ymax = float(elem[7])
                #convert label to int
                label = self.class_names.index(label) 
                box = Box(x0=xmin, y0 = ymin, x1=xmax, y1=ymax,label=label)

                dict_annot[filename].append(box)
                if len(dict_annot) == self.bulk_ind +1 :
                    # save the last entry since it doesnt have all the boxes yet
                    od = OrderedDict(dict_annot)
                    temp = od.popitem()
                    for filename in od.keys():
                        image_name = filedir+data_type+'/'+filename+'.jpg'
                        boxes = np.array(dict_annot[filename])
                        images.append((image_name,self.image_size[0],self.image_size[1]))
                        if max_boxes < boxes.shape[0]:
                            max_boxes = boxes.shape[0]
                        labels.append(boxes)
                    images = np.array(images)
                    labels = np.array(labels)
                    
                    sto = (images,labels,max_boxes)
                    proc_images.put(sto)
                    
                    max_boxes = 0
                    images = []
                    labels = []
                    dict_annot.clear()
                    dict_annot[temp[0]] = temp[1]


            for filename in dict_annot.keys():
                image_name = filedir+data_type+'/'+filename+'.jpg'
                boxes = np.array(dict_annot[filename])
                images.append((image_name,self.image_size[0],self.image_size[1]))
                if max_boxes < boxes.shape[0]:
                    max_boxes = boxes.shape[0]
                labels.append(boxes)
            images = np.array(images)
            labels = np.array(labels)
            sto = (images,labels,max_boxes)
            proc_images.put(sto)

            print "DONE"


    def write_tf(self,directory,num_shards = 1):
        convert_to(self,directory,self.name,num_shards=num_shards)
    
def create_index(name = "open_image"):
    request_body = {
            'settings' : {
                'number_of_shards': 3,
                'number_of_replicas': 0,
                'refresh_interval':'2s'
            },
            #define field properties
            'mappings': {
                "train": {
                    'properties': {
                        'image':{'type':'text'},
                        'label': {'type': 'integer'},
                        'xmin':{'type':'float'},
                        'ymin':{'type':'float'},
                        'xmax':{'type':'float'},
                        'ymax':{'type':'float'},
                        'id ':{'type':'keyword'},
                        'mask':{'type':'text'}
                    }}}}

    #create the index
    es.indices.create(index = name, body = request_body)

