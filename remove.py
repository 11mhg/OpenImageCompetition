import ast
import csv
import numpy as np
import tensorflow as tf
import os

import base64
import elasticsearch
from multiprocessing import Process, Queue
from tqdm import tqdm
from collections import defaultdict
from collections import OrderedDict
from PIL import Image
from elasticsearch import Elasticsearch,client ,helpers

es = Elasticsearch()
work = Queue()
def do_work(in_queue):
	while True:
		query = in_queue.get()
		try:
			es.delete_by_query(index='open_image_train',doc_type='train', \
                                body={"query":{"term": {"id ": query}}})
		except:
			print query
			print "error"
for i in range(2):
	Process(target=do_work, args=(work,)).start()

with open('logs2.txt','r') as f:
	reader = csv.reader(f,delimiter = ',')
	for i in tqdm(reader):
		work.put(i[0])

print "DONE"
