import csv
import numpy as np
# import tensorflow as tf
from tqdm import tqdm
from PIL import Image
from augmenter import *
import os
import math
import threading
import cv2
import base64
import elasticsearch
from elasticsearch import Elasticsearch,client ,helpers
es = Elasticsearch()

def _process_image_files(images,labels,max_boxes,image_size,shard_index):
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
            else:
                xmin.append(0)
                ymin.append(0)
                xmax.append(0)
                ymax.append(0)
                label.append(-1)
                #string, 
        yield image ,image_name,xmin ,ymin ,xmax ,ymax ,label

def convert_to(self,directory, name, data_type = 'train',image_size = (800,800), num_shards = 1):
    doc_count=1
    actions=[]
    for i in range(num_shards):
        elem_per_shard = int(math.ceil((self.num_examples/num_shards)))
        start_ndx = i * elem_per_shard
        end_ndx = min((i+1)*elem_per_shard,self.num_examples)
        images = self.images[start_ndx:end_ndx-1]
        labels = self.labels[start_ndx:end_ndx-1]

        for temp in _process_image_files(images, labels, self.max_boxes,image_size,i):
            image,image_name ,xmin ,ymin ,xmax ,ymax ,label = temp
            image_id = os.path.splitext(os.path.basename(image_name))[0]
            actions.append({"_index":"open_image", "_type":'train',
                    'image': str(image),
                    'label' : label,
                    'xmin':xmin,
                    'ymin':ymin,
                    'xmax':xmax,
                    'ymax':ymax,
                    'id ':image_id
                })
            if (doc_count%1500==0):
                try:
                    helpers.bulk(es, actions,request_timeout=100000)
                except elasticsearch.ElasticsearchException as es1:
                    print "error"
                    print es1
                actions =[]
            doc_count+=1
        helpers.bulk(es, actions,request_timeout=100000)
    print "Index Information"
    print es.cat.indices(v='true')

def create_index():
    request_body = {
            'settings' : {
                'number_of_shards': 15,
                'number_of_replicas': 1,
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
                        'id':{'type':'text'}
                    }}}}

    #create the index
    es.indices.create(index = "open_image", body = request_body)
# create_index()
