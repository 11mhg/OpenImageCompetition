import csv
import numpy as np
import tensorflow as tf
import os
import math
import threading
import Queue
import sys
import cv2
import base64
import elasticsearch


from data.bbox import Box
from tqdm import tqdm
from data.tfrecord_utils import convert_to, input_fn
from collections import defaultdict
from collections import OrderedDict
from PIL import Image
from augmenter import *
from elasticsearch import Elasticsearch,client ,helpers
es = Elasticsearch()

work = Queue.Queue()
lock = threading.Lock()

def do_work(in_queue):
    while True:
        item = in_queue.get()
        try:
            helpers.bulk(es, item, request_timeout = 100000)
        except elasticsearch.ElasticsearchException as es1:
            lock.acquire()
            with open('logs_train.txt','a') as f:
                 for i in item:
                     f.write(i['id ']+'\n')
            lock.release()
	del item[:]
        in_queue.task_done()

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
    
    def get_batch(self,filenames):
        return input_fn(self, filenames)

class PreProcessData:
    def __init__(self,classes_text='./dummy_labels.txt',image_size=(800,800)):
        self.images = None
        self.labels = None
        self.image_size=image_size
        self.doc_count = 0
        self.bulk_ind = 1000
        with open(classes_text) as f:
            self.class_names = []
            for lines in f.readlines():
                arr = lines.strip().split(',')
                self.class_names.append(arr[0])

        for i in xrange(10):
            t = threading.Thread(target=do_work,args=(work,))
            t.daemon = True
            t.start()

    def get_open_images(self,filedir,data_type='train',name="OpenImage"):
	self.ind_name = 'open_image_'+data_type
        try:
            create_index(name = self.ind_name)
        except:
            pass
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
            dict_annot = defaultdict(list)
            pbar = tqdm(bbox_reader)
            pbar.set_description("Reading Annotations")
            for j,elem in enumerate(pbar):
                if j<1196752:
                    continue
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
                if len(dict_annot) == self.bulk_ind*10 +1 :
                    # save the last entry since it doesnt have all the boxes yet
                    od = OrderedDict(dict_annot)
                    temp = od.popitem()
                    for filename in od.keys():
                        image_name = filedir+data_type+'/'+filename+'.jpg'
                        boxes = np.array(dict_annot[filename])
                        self.images.append((image_name,self.image_size[0],self.image_size[1]))
                        if self.max_boxes < boxes.shape[0]:
                            self.max_boxes = boxes.shape[0]
                        self.labels.append(boxes)
                    self.images = np.array(self.images)
                    self.labels = np.array(self.labels)
                    self.num_examples = self.images.shape[0]
                    self.convert_to(filedir, name, data_type=data_type)
                    dict_annot.clear()
                    dict_annot[temp[0]] = temp[1]

            self.convert_to(filedir, name, data_type=data_type) 

    def convert_to(self,directory, name, data_type = 'train',image_size = (800,800), num_shards = 1):
        images = self.images[:]
        labels = self.labels[:]
        actions = []
        # get mask from es mask
        # p = np.frombuffer(base64.b64decode(res['_source']['maskInt']),dtype=np.uint8).reshape(800,800,-1)
        for temp in _process_image_files(images, labels, self.max_boxes,image_size,num_shards):
            image,image_name ,xmin ,ymin ,xmax ,ymax ,label = temp
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
                hold_mask[d:e,b:c] = 1
                masks[:,:,i] = hold_mask[:,:]
            self.doc_count+=1
            actions.append({"_index":self.ind_name, "_type":'train',
                    'image': str(image),
                    'label' : label,
                    'xmin':xmin,
                    'ymin':ymin,
                    'xmax':xmax,
                    'ymax':ymax,
                    'id ':image_id,
                    'mask':str(base64.b64encode(masks.tobytes()))
                })
            if self.doc_count%50 == 0:
                work.put(actions)
                actions = []
        
        work.join()
        self.images = []
        self.labels = []
        self.max_boxes = 0
        self.num_examples = 0

    def write_tf(self,directory,num_shards = 1):
        convert_to(self,directory,self.name,num_shards=num_shards)

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

        yield image ,image_name,xmin ,ymin ,xmax ,ymax ,label

def create_index(name = "open_image"):
    request_body = {
            'settings' : {
                'number_of_shards': 12,
                'number_of_replicas': 0,
                'refresh_interval':'-1'
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
                        'id':{'type':'text'},
                        'mask':{'type':'text'}
                    }}}}

    #create the index
    es.indices.create(index = name, body = request_body)
