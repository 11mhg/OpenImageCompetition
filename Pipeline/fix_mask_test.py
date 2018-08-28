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
from multiprocessing import Process, Queue
from data.bbox import Box
from tqdm import tqdm
#from data.tfrecord_utils import convert_to, input_fn
from collections import defaultdict
from collections import OrderedDict
from PIL import Image
from augmenter import *
from elasticsearch import Elasticsearch,client ,helpers

es = Elasticsearch()
work = Queue()

def do_work(in_queue):
    while True:
        iden,xmin,xmax,ymin,ymax,message = in_queue.get()
        if message == 'exit':
            print 'cleaning up worker ....'
            break
        hold_mask = np.zeros((800,800),dtype=np.uint8)
        m_xmin = (np.array(xmin)*800*800)
        m_xmax = (np.array(xmax)*800*800)
        m_ymin = (np.array(ymin)*800*800)
        m_ymax = (np.array(ymax)*800*800)
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
        try:
            es.update(index = 'open_image_val',doc_type = 'train',id=iden,
            body={'doc':{'mask':str(base64.b64encode(masks.tobytes()))}},retry_on_conflict = 100)
        except:
            with open('logs.txt','a') as f:
                print "index error"
                print iden
                writer = csv.writer(f)
                writer.writerow([iden])

processes = []
for i in range(2):
    p = Process(target=do_work,args=(work,))
    processes.append(p)
    p.start()


page = helpers.scan(es,
        index = 'open_image_val',
        doc_type = 'train',
        scroll = '2m',
        size = 2000,
        query = {
            'query':{'match_all':{}},
            '_source':['xmin','xmax','ymin','ymax']
        },
        sort = ['_doc'],
        request_timeout = 10000
    )

# Start scrolling
for j in tqdm(page):
    iden = j['_id']
    try:
        xmin = j['_source']['xmin']
        xmax = j['_source']['xmax']
        ymin = j['_source']['ymin']
        ymax = j['_source']['ymax']
        action = (iden,xmin,xmax,ymin,ymax,'')
        work.put(action)
    except:
        with open('failed.csv','w') as f:
            writer = csv.writer(f)
            writer.writerow([iden])
            print iden
action = (None,None,None,None,None,'exit')

for p in processes:
    work.put(action)
print("Waiting for process to finish")
print work.qsize()
for p in processes:
    p.join()
print("Finally Done")
