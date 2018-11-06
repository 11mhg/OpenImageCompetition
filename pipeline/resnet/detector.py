from .resnet import get_class_resnet
from .utils import get_flags
from data.data import Data
import numpy as np
import os, sys
import time
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.slim.python.slim.nets import resnet_v2

slim = tf.contrib.slim

class Detector():
    def __init__(self,FLAGS):
        self.flags = FLAGS
        self.num_epochs_before_decay = 2
        self.lr_decay=0.7
        self.lr = 0.0002
        self.checkpoint_file = self.flags.model_dir+'model.ckpt'
        if self.flags.dtype=='float16':
            self.dtype='float16'
        else:
            self.dtype='float32'


