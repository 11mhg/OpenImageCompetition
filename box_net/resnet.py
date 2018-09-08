from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
from absl import app
import tensorflow as tf
import os
import sys
from utils import get_flags
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.slim.python.slim.nets import resnet_v2 
from data.data import *
slim = tf.contrib.slim
flags = get_flags()


def loss_fn(logits, labels):
    classification_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels= labels,logits=logits,name='classification_loss')

    reg_loss = tf.add_n(tf.losses.get_regularization_losses())
    total_loss = classification_loss+ reg_loss
    return total_loss


def get_resnet(inputs, is_training=False):
    with tf.variable_scope("box_net"):
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            out, end_points = resnet_v2.resnet_v2_50(inputs,None,reuse=tf.AUTO_REUSE, is_training=is_training)
            out = tf.layers.conv2d(out, 2048,[1,1],activation=tf.nn.leaky_relu,name='box',reuse=tf.AUTO_REUSE)
            box_out = tf.layers.conv2d(out, 4,[1,1], activation=tf.nn.tanh,name='final_box',reuse=tf.AUTO_REUSE)
            box_out = tf.squeeze(box_out,[1,2])
    return box_out, end_points


class Resnet_Classifier():
    def __init__(self, FLAGS):
        self.flags = FLAGS
        self.num_epochs_before_decay = 2
        self.lr_decay = 0.7
        self.lr = 0.0002
        self.checkpoint_file = self.flags.model_dir+'model.ckpt'

    def ready_dataset(self):
        data = Data(self.flags.labels)
        self.num_classes =len(data.class_names)
        self.train_dataset = data.get_batch(self.flags.data_dir)
        self.val_dataset = data.get_batch(self.flags.val_dir)
        self.train_iter = self.train_dataset.make_initializable_iterator()
        self.val_iter = self.val_dataset.make_initializable_iterator()
        self.input_tensor, self.label_tensor, self.box_tensor = self.train_iter.get_next()
        self.val_image, self.val_labels, self.val_box = self.val_iter.get_next()
        #put labels in -1 to 1 range
        self.box_tensor = tf.multiply(self.box_tensor,tf.constant(2,dtype=tf.float32)) - tf.constant(1,dtype=tf.float32)
        self.val_box = tf.multiply(self.val_box,tf.constant(2,dtype=tf.float32)) - tf.constant(1,dtype=tf.float32)



    def train_step(self,sess, train_op, metrics_op, global_step):
        start_time = time.time()
        total_loss, global_step_count, _ = sess.run([train_op, global_step, metrics_op])
        time_elapsed = time.time() - start_time
        logging.info('global_step %s: loss %.4f (%.2f sec/step)', global_step_count, total_loss, time_elapsed)
        return total_loss, global_step_count
    
    def val_step(self, sess, val_op, val_xy,val_wh):
        v_loss, v_xy, v_wh = sess.run([val_op, val_xy,val_wh])
        return v_loss, v_xy, v_wh

    def train(self):
        num_batches_per_epoch = int(np.floor(self.flags.steps_per_epoch / self.flags.batch_size))
        num_steps_per_epoch = num_batches_per_epoch 
        decay_steps = int(self.num_epochs_before_decay * num_steps_per_epoch)
        self.input_tensor = tf.cast(self.input_tensor,tf.float32)
        box_out, end_points = get_resnet(self.input_tensor,is_training=True)
        loss = tf.sqrt(tf.reduce_sum(tf.square(self.box_tensor - box_out)))
        tf.losses.add_loss(loss)
        total_loss = tf.losses.get_total_loss()

        global_step = tf.train.get_or_create_global_step()

        lr = tf.train.exponential_decay(
                learning_rate = self.lr,
                global_step = global_step,
                decay_steps = decay_steps,
                decay_rate = self.lr_decay,
                staircase=True)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = slim.learning.create_train_op(total_loss,optimizer)
        

        rmse, rmse_update = tf.metrics.root_mean_squared_error(self.box_tensor,box_out)
        metrics_op = tf.group(rmse_update)
        
        tf.summary.image('input_images',self.input_tensor[:5,:,:,:3])
        tf.summary.image('input_masks',self.input_tensor[:5,:,:,3:])
        tf.summary.scalar('losses/Total_loss', total_loss)
        tf.summary.scalar('rmse',rmse)
        tf.summary.scalar('learning_rate', lr)

        val_box, val_endpoints = get_resnet(tf.cast(self.val_image,tf.float32),is_training=False) 

        val_loss = tf.sqrt(tf.reduce_sum(tf.square(self.val_box - val_box))) 

        val_xy = tf.reduce_sum(tf.square(self.val_box[:,:2] - val_box[:,:2]))
        val_wh = tf.reduce_sum(tf.square(self.val_box[:,2:] - val_box[:,2:]))
    

        tf.summary.scalar('losses/val_loss',val_loss)
        tf.summary.scalar('val_xy',val_xy)
        tf.summary.scalar('val_wh',val_wh)

        summary_op = tf.summary.merge_all()

        saver = tf.train.Saver(max_to_keep=4)
       
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            #initialize both local and global variables for metrics and network
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            #if a checkpoint exists, load it in
            if tf.train.latest_checkpoint(self.flags.model_dir) is not None:
                saver.restore(sess,tf.train.latest_checkpoint(self.flags.model_dir))
            #define log writer
            summary_writer = tf.summary.FileWriter(self.flags.logs,graph=tf.get_default_graph())
            #intiialize the train and val dataset fetchers
            sess.run(self.train_iter.initializer)
            sess.run(self.val_iter.initializer)
            previous_val = 100000
            for epoch in range(self.flags.num_epochs): 
                for step in range(num_steps_per_epoch):
                    try:
                        if step % 10 == 0 :
                            loss, _ = self.train_step(sess,train_op, metrics_op, global_step)
                            summaries = sess.run(summary_op) 
                            summary_writer.add_summary(summaries, epoch+1)
                        else:
                            loss, _ = self.train_step(sess,train_op, metrics_op, global_step)
                    except tf.errors.OutOfRangeError:
                        logging.info("Train data sequence done, resetting initializer")
                        sess.run(self.train_iter.initializer)
                    except KeyboardInterrupt:
                        print("Interrupted")
                        sys.exit(0)

                #saver.save(sess,self.checkpoint_file,global_step = global_step)
                logging.info('Epoch %s/%s', epoch+1, self.flags.num_epochs)
                learning_rate_value, rmse_value = sess.run([lr,rmse])
                logging.info('Current Learning Rate: %s', learning_rate_value)
                logging.info('Current Streaming RMSE: %s', rmse_value)
                
                logging.info("Beginning Validation")
                v_loss = 0.0
                v_xy = 0.0
                v_wh = 0.0
                count = 1.0
                while True:
                    count += 1
                    try:
                        v_loss, xy_diff, wh_diff = self.val_step(sess, val_loss, val_xy, val_wh)
                        if count % 100 == 0:
                            logging.info('Processing validation sample # %s',count)
                        v_xy += xy_diff
                        v_wh += wh_diff
                    except tf.errors.OutOfRangeError:
                        count -= 1
                        logging.info("Validation data sequence done, resetting initializer")
                        sess.run(self.val_iter.initializer)
                        break
                    except KeyboardInterrupt:
                        print("Interrupted")
                        sys.exit(0)
                avg_xy = v_xy/count
                avg_wh = v_wh/count
                total = avg_xy+avg_wh
                if total <= previous_val:
                    logging.info('Better validation, saving now')
                    saver.save(sess,self.checkpoint_file,global_step=global_step)
                    previous_val = total
                logging.info('Num validation samples; %s', count)
                logging.info('Validation Loss: %s',v_loss)
                logging.info('Validation xy regression: %s', avg_xy)
                logging.info('Validation wh regression: %s', avg_wh)


                

def main(argv):
    global flags
    cr = Resnet_Classifier(flags.FLAGS)
    cr.ready_dataset()
    cr.train()


if __name__=='__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run(main)
