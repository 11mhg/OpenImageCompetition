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

from data.bbox import Box
from collections import defaultdict
from collections import OrderedDict
from augmenter import *
from elasticsearch import Elasticsearch,client ,helpers
from elasticsearch_dsl import Search
es = Elasticsearch()


a = helpers.scan(es,index = 'open_image_train',
                doc_type = 'train',
                scroll = '2m',
                size = 10000,
                query = {
#                     'slice':{'id':slice_no,'max':SLICES},
                    'query':{'match_all':{}},
                    '_source':['id ']
#                     '_source':['xmin','ymin','xmax','ymax','mask']
                },
                sort = ['_doc'],request_timeout=10000)


from tqdm import tqdm
import csv
with open('image_ids_test.csv','w') as f:
    writer = csv.writer(f,delimiter = ',')
    for i in tqdm(a):
        stuff = [[j['_id'],j['_source']['id ]] for j in i]
        writer.writerows(stuff)

