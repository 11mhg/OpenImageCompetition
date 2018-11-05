import tensorflow as tf
import numpy as np
from data.data import *
import PIL
import time
from tqdm import tqdm

data = PreProcessData('./dummy_labels.txt')
data.load_mot('../Dataset/MOT/images/train/')

sess = tf.Session()

image = tf.placeholder(dtype=tf.float32,shape=[None,800,800,3])
labels = tf.placeholder(dtype=tf.int64, shape=[None,None,1])
boxes = tf.placeholder(dtype=tf.float32, shape=[None,None,4])
r = 0

def get_batch(size):
    global r
    global data
    if r*size+size > data.labels.shape[0]:
        r=0
    max_boxes = data.max_boxes
    batch_boxes = []
    batch_labels = []
    batch_images = []
    for ind in range(r*size, r*size+size):
        temp_boxes = data.labels[ind]
        bboxes = []
        labels = []
        for i in range(0,max_boxes):
            if i < temp_boxes.shape[0]:
                box = temp_boxes[i]
                bboxes.append(box.xyxy)
                labels.append(box.label)
            else:
                bboxes.append([0,0,0,0])
                labels.append(-1)
        bboxes = np.array(bboxes)
        labels = np.array(labels)
        labels = labels.reshape((-1,1))
        batch_boxes.append(bboxes)
        batch_labels.append(labels)
        image = data.images[ind][0]
        image = PIL.Image.open(image).resize((800,800),PIL.Image.BILINEAR)
        image = np.array(image).astype('float32')
        image /= 255.
        batch_images.append(image)
    batch_boxes = np.array(batch_boxes)
    batch_labels = np.array(batch_labels)
    batch_images = np.array(batch_images)
    return batch_images, batch_labels, batch_boxes

with tf.Session() as sess:
    batch_size = 32
    pbar = tqdm(range(200))
    start_time = time.time()
    for i in pbar:
        batch_images, batch_labels, batch_boxes = get_batch(batch_size)
        _, _, _ = sess.run([image,labels,boxes],feed_dict={image: batch_images, labels: batch_labels, boxes: batch_boxes})
    end_time = time.time()

    print("Batches completed in {} seconds.".format(end_time-start_time))
    print("{} samples per second.".format(200*32/(end_time-start_time)))
    print("{} seconds per sample.".format((end_time-start_time)/(200*32)))
