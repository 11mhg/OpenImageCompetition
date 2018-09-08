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
from classifier_data.classifier_data import *
slim = tf.contrib.slim
flags = get_flags()


def loss_fn(logits, labels):
    classification_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels= labels,logits=logits,name='classification_loss')

    reg_loss = tf.add_n(tf.losses.get_regularization_losses())
    total_loss = classification_loss+ reg_loss
    return total_loss


def get_resnet(inputs, num_classes, is_training=False):
    with tf.variable_scope("classifier"):
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, end_points = resnet_v2.resnet_v2_50(inputs,num_classes, reuse=tf.AUTO_REUSE, is_training=is_training)
    return logits, end_points




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
        self.input_tensor, self.label_tensor = self.train_iter.get_next()
        self.val_image, self.val_labels = self.val_iter.get_next()



    def train_step(self,sess, train_op, metrics_op, global_step):
        start_time = time.time()
        total_loss, global_step_count, _ = sess.run([train_op, global_step, metrics_op])
        time_elapsed = time.time() - start_time
        logging.info('global_step %s: loss %.4f (%.2f sec/step)', global_step_count, total_loss, time_elapsed)
        return total_loss, global_step_count
    
    def val_step(self, sess, val_op, val_metric):
        start_time = time.time()
        v_loss, v_acc = sess.run([val_op, val_metric])
        time_elapsed = time.time() - start_time
        return v_loss, v_acc

    def train(self):
        num_batches_per_epoch = int(np.floor(self.flags.steps_per_epoch / self.flags.batch_size))
        num_steps_per_epoch = num_batches_per_epoch 
        decay_steps = int(self.num_epochs_before_decay * num_steps_per_epoch)
        self.input_tensor = tf.cast(self.input_tensor,tf.float32)
        logits, end_points = get_resnet(self.input_tensor,self.num_classes,is_training=True)
        one_hot_labels = slim.one_hot_encoding(self.label_tensor,self.num_classes)
        logits = tf.squeeze(logits,[1,2])

        loss = tf.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, logits=logits)

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
        

        predictions = tf.squeeze(tf.argmax(end_points['predictions'],-1),[1])
        probabilities = end_points['predictions']

        accuracy, accuracy_update = tf.metrics.accuracy(self.label_tensor,predictions)
        metrics_op = tf.group(accuracy_update)
        tf.summary.image('input_images',self.input_tensor[:5,:,:,:3])
        tf.summary.image('input_masks', self.input_tensor[:5,:,:,3:])
        tf.summary.scalar('losses/Total_loss', total_loss)
        tf.summary.scalar('accuracy',accuracy)
        tf.summary.scalar('learning_rate', lr)

        val_logits, val_endpoints = get_resnet(tf.cast(self.val_image,tf.float32),self.num_classes,is_training=False) 
        val_one_hot_labels = slim.one_hot_encoding(self.val_labels,self.num_classes)
        val_logits = tf.squeeze(val_logits,[1,2])

        val_loss = tf.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, logits=logits)
         
        val_pred = tf.squeeze(tf.argmax(val_endpoints['predictions'],-1),[1])
        val_accuracy, val_accuracy_update = tf.metrics.accuracy(self.val_labels,val_pred)
        val_metrics_op = tf.group(val_accuracy_update)
        
        
        tf.summary.scalar('losses/val_loss',val_loss)
        tf.summary.scalar('val_accuracy',val_accuracy)

        summary_op = tf.summary.merge_all()

        saver = tf.train.Saver(max_to_keep=4)
        prev_val_accuracy = 0.0
        
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth=True

        with tf.Session(config=config) as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            if tf.train.latest_checkpoint(self.flags.model_dir) is not None:
                logging.info("Loading latest checkpoint found")
                saver.restore(sess,tf.train.latest_checkpoint(self.flags.model_dir))
            summary_writer = tf.summary.FileWriter(self.flags.logs,graph=tf.get_default_graph())
            sess.run(self.train_iter.initializer)
            sess.run(self.val_iter.initializer)
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
                    except:
                        logging.info("There was a problem this iteration, skipping.")
                        continue

                logging.info('Epoch %s/%s', epoch+1, self.flags.num_epochs)
                learning_rate_value, accuracy_value = sess.run([lr,accuracy])
                logging.info('Current Learning Rate: %s', learning_rate_value)
                logging.info('Current Streaming Accuracy: %s', accuracy_value)
                
                logging.info("Beginning Validation")
                v_loss = 0.0
                count = 0
                while True:
                    count += 1
                    try:
                        v_loss, _ = self.val_step(sess, val_loss, val_metrics_op)

                        if count % 100 == 0:
                            logging.info('Processing validation sample # %s',count)
                    except tf.errors.OutOfRangeError:
                        logging.info("Validation data sequence done, resetting initializer")
                        sess.run(self.val_iter.initializer)
                        break
                    except KeyboardInterrupt:
                        print("Interrupted")
                        sys.exit(0)
                    except:
                        logging.info("There was a problem this iteration, skipping.")
                
                v_accuracy = sess.run([val_accuracy])[0]
                
                logging.info('Num validation samples; %s', count)
                logging.info('Validation Loss: %s',v_loss)
                logging.info('Validation Accuracy: %s', v_accuracy)
                if v_accuracy >= prev_val_accuracy:
                    prev_val_accuracy = v_accuracy
                    saver.save(sess,self.checkpoint_file,global_step = global_step)


                

def main(argv):
    global flags
    cr = Resnet_Classifier(flags.FLAGS)
    cr.ready_dataset()
    cr.train()


if __name__=='__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run(main)
