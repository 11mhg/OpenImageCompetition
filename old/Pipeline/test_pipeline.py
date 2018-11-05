from data.data import *
import tensorflow as tf
import time
from tqdm import tqdm

d = Data('./dummy_labels.txt')
dataset = d.get_batch('../Dataset/MOT/*.tfrecords')
iterator = dataset.make_initializable_iterator()
Image, Labels, Boxes = iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer)
    start_samples =  time.time()
    for i in pbar:
        try:
            _,_,_ = sess.run([Image,Labels,Boxes])
        except tf.errors.OutOfRangeError:
            sess.run(iterator.initializer)
            _,_,_ = sess.run([Image,Labels,Boxes])

