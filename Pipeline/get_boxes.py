import csv
import numpy as np
import os
import math
import threading
import Queue
import sys
import cv2
import base64
import elasticsearch

from tqdm import tqdm
from data.bbox import Box
from collections import defaultdict
from collections import OrderedDict
from augmenter import *
from elasticsearch import Elasticsearch,client ,helpers

es = Elasticsearch()
from collections import defaultdict,OrderedDict
from data.bbox import Box
dict_annot = defaultdict(list)
filedir = '../../Dataset/OpenImage/'
annotations_file = filedir+'annotations/' + '{}-bbox.csv'.format('train')

def _process_image_files(labels,names):
    for index in range(labels.shape[0]):
        height = float(800)
        width = float(800)
        xmin = []
        xmax = []
        ymin = []
        ymax = []
        label = []
        #get the xyxy components of box
        boxes = labels[index]
        for i in boxes:
            b = i.xyxy
            xmin.append(b[0]/width)
            ymin.append(b[1]/height)
            xmax.append(b[2]/width)
            ymax.append(b[3]/height)
            label.append(box.label)

        yield xmin ,ymin ,xmax ,ymax,names[index]

with open(annotations_file,'r') as csvfile:
    with open('boxes.csv','w') as f:
        writer = csv.writer(f,delimiter=',')
        bbox_reader = csv.reader(csvfile,delimiter=',')
        print("Open Images contains a large number of files, do not be discourage if it takes a long time.")
        next(bbox_reader)
        dict_annot = defaultdict(list)
        pbar = tqdm(bbox_reader)
        pbar.set_description("Reading Annotations")
        labels = []
        names = []
        for elem in pbar:
            filename = elem[0]
            label = elem[2]
            xmin = float(elem[4])
            xmax = float(elem[5])
            ymin = float(elem[6])
            ymax = float(elem[7])
            box = Box(x0=xmin, y0 = ymin, x1=xmax, y1=ymax,label=label)
            dict_annot[filename].append(box)
            if len(dict_annot) == 10000 +1 :

                
                od = OrderedDict(dict_annot)
                temp2 = od.popitem()
                for filename in od.keys():
                    boxes = np.array(dict_annot[filename])
                    labels.append(boxes)
                    names.append(filename)
                for temp in _process_image_files(np.array(labels),names):
                    xmin ,ymin ,xmax ,ymax ,names = temp
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
                    writer.writerow([names,str(base64.b64encode(masks.tobytes()))])
                labels = []
                names = []
                dict_annot.clear()
                dict_annot[temp2[0]] = temp2[1]
                


