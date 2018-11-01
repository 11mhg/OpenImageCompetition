from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
from .tfrecord_utils import input_fn

class Data:
    def __init__(self, classification=False, classes_text=None, batch_size=32, shuffle_buffer_size=4, prefetch_buffer_size=1, num_parallel_calls=4, num_parallel_readers=1):
        if classes_text:
            with open(classes_text) as f:
                self.class_names=[]
                for lines in f.readlines():
                    arr = lines.strip().split(',')
                    self.class_names.append(arr[-1])
        self.classification = classification
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.prefetch_buffer_size = prefetch_buffer_size
        self.num_parallel_calls = num_parallel_calls
        self.num_parallel_readers = num_parallel_readers

    def get_batch(self,filenames=None):
        return input_fn(self,filenames)
