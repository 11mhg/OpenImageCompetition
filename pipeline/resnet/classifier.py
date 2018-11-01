from .resnet import get_class_resnet
from .flags import get_flags
from ..data.data import Data
import numpy as np
import os, sys
import time
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.slim.python.slim.nets import resnet_v2

slim = tf.contrib.slim


class Classifier():

    def __init__(self,FLAGS):
        self.flags = FLAGS
        self.num_epochs_before_decay=2
        self.lr_decay = 0.7
        self.lr = 0.0002
        self.checkpoint_file = self.flags.model_dir+'model.ckpt'
        if self.flags.dtype == 'float32':
            self.dtype = 'float32'
        else:
            self.dtype = 'float16'

    def ready_dataset(self):
        data = Data(self.flags.labels, classification=True, batch_size = self.flags.batch_size)
        self.num_classes = len(data.class_names)
        self.train_dataset = data.get_batch(self.flags.data_dir)
        self.val_dataset = data.get_batch(self.flags.val_dir)

        self.train_iter = self.train_dataset.make_initializable_iterator()
        self.val_iter = self.val_dataset.make_initializable_iterator()
       
        self.train_image, self.train_label = self.train_iter.get_next()
        self.val_image, self.val_label = self.val_iter.get_next()

    
    def train(self):
        num_batches_per_epoch = int(np.floor(self.flags.steps_per_epoch / self.flags.batch_size))
        num_steps_per_epoch = num_batches_per_epoch
        decay_steps = int(self.num_epochs_before_decay * num_steps_per_epoch)

        if self.dtype == 'float32':
            self.train_image = tf.cast(self.train_image,tf.float32)
            self.val_image = tf.cast(self.val_image,tf.float32)
        else:
            self.train_image = tf.cast(self.train_image,tf.float16)
            self.val_image = tf.cast(self.val_image,tf.float16)

        train_logits, train_end_points = get_class_resnet(self.train_image,self.num_classes,is_training=True)

        one_hot_labels = tf.one_hot(self.label_tensor, self.num_classes)
        train_logits = tf.squeeze(train_logits,[1,2])

        if self.dtype == 'float16':
            train_logits = tf.cast(train_logits, tf.float32)

        loss = tf.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, logits = train_logits)
        
        loss = tf.losses.get_total_loss()

        global_step = tf.train.get_or_create_global_step()

        lr = tf.train.exponential_decay(
                learning_rate = self.lr,
                global_step = global_step,
                decay_steps = decay_steps,
                decay_rate = self.lr_decay,
                staircase=True)

        loss_scale = 128. if self.dtype=='float16' else 1.

        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        grad_mult = {variables[i]:1./loss_scale for i in range(0,len(variables))}

        optimizer = tf.trani.AdamOptimizer(learning_rate=lr)
        train_op = slim.learning.create_train_op(loss * loss_scale, optimizer, gradient_multipliers=grad_mult)

        predictions = tf.squeeze(tf.argmax(end_points['predictions'],-1),[1])
        probabilities = end_points['predictions']

        accuracy, accuracy_update = tf.metrics.accuracy(self.train_label, predictions)
        metrics_op = tf.group(accuracy_update)


        #train summaries
        train_image_summ = tf.summary.image('train/input_images',self.train_image[:5,:,:,:3])
        train_mask_summ = tf.summary.image('train/input_masks',self.train_image[:5,:,:,3:])
        train_loss_summ = tf.summary.scalar('train/train_loss',loss)
        train_acc_summ = tf.summary.scalar('train/accuracy',accuracy)
        train_lr_summ = tf.summary.scalar('train/learning_rate',lr)

        train_summary_op = tf.summary.merge([train_image_summ,
            train_mask_summ,train_loss_summ,train_acc_summ,train_lr_summ])

        val_logits, val_end_points = get_class_resnet(self.val_image,self.num_classes,is_training=False)

        val_one_hot_labels = tf.one_hot(self.val_labels,self.num_classes)
        val_logits = tf.squeeze(val_logits,[1,2])

        if self.dtype=='float16':
            val_logits = tf.cast(val_logits,tf.float32)

        val_loss = tf.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, logits=val_logits)

        val_pred = tf.squeeze(tf.argmax(val_end_points['predictions'],-1),[1])
        val_accuracy,val_accuracy_update = tf.metrics.accuracy(self.val_labels, val_pred)
        val_metrics_op = tf.group(val_accuracy_update)

        
        #summaries
        val_image_summ = tf.summary.image('val/input_images',self.val_image[:5,:,:,:3])
        val_mask_summ = tf.summary.image('val/input_masks',self.val_image[:5,:,:,3:])
        val_loss_summ = tf.summary.scalar('val/train_loss',val_loss)
        val_acc_summ = tf.summary.scalar('val/accuracy',val_accuracy)

        val_summary_op = tf.summary.merge([val_image_summ, val_mask_summ,
                                           val_loss_summ,val_acc_summ])

        saver = tf.train.Saver(max_to_keep=4)
        prev_val_accuracy=0.0

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth=True

        with tf.Session(config=config) as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            logging.info("Initial variables have been initialized")
            if tf.train.latest_checkpoint(self.flags.model_dir) is not None:
                logging.info("Loading latest checkpoint found")
                saver.restore(sess,tf.train.latest_checkpoint(self.flags.model_dir))

            summary_writer = tf.summary.FileWriter(self.flags.logs,graph=tf.get_default_graph())

            sess.run(self.train_iter.initializer)
            sess.run(self.val_iter.initializer)
            logging.info("TFData api has been initialized")

            val_counter = 0
            for epoch in range(self.flags.num_epochs):
                for step in range(num_steps_per_epoch):
                    try:
                        if step % 10 == 0:
                            loss, step_count = self.train_step(sess,train_op, metrics_op, global_step)
                            train_summaries = sess.run(train_summary_op)
                            summary_writer.add_summary(train_summaries,step_count)
                        else:
                            loss,_ = self.train_step(sess,train_op,metrics_op,global_step)
                    except tf.errors.OutOfRangeError:
                        logging.info("Train data sequence completed, resetting initializer")
                        sess.run(self.train_iter.initializer)
                    except KeyboardInterrupt:
                        logging.info("Keyboard Interrupt")
                        sys.exit(0)
                logging.info('Epoch %s/%s', epoch+1, self.flags.num_epochs)
                learning_rate_value, accuracy_value = sess.run([lr,accuracy])
                logging.info('Current lr: %s', learning_rate_value)
                logging.info('Current training accuracy: %s',accuracy_value)

                logging.info("Validation")
                v_loss = 0.0
                count = 0
                while True:
                    val_counter += 1
                    count+=1
                    try:
                        v_loss, _ = self.val_step(sess, val_loss, val_metrics_op)
                        if count % 10 == 0:
                            val_summaries = sess.run(val_summary_op)
                            summary_writer.add_summary(val_summaries,val_counter)
                        if count % 100 == 0:
                            logging.info('Processing validation sample # %s',count)
                    except tf.errors.OutOfRangeError:
                        logging.info("Validation data sequence done, resetting initializer")
                        sess.run(self.val_iter.initializer)
                        break
                    except KeyboardInterrupt:
                        logging.info("Keyboard Interrupt")
                        sys.exit(0)
                    except:
                        count -= 1
                        logging.info("There was a problem this iteration, skipping")

                v_accuracy = sess.run([val_accuracy])[0]
                logging.info('Num validation samples processed: %s',count)
                logging.info('Validation Loss: %s',v_loss)
                logging.info('Validation Accuracy: %s',v_accuracy)
                
                if v_accuracy >= prev_val_accuracy:
                    logging.info('Better validation accuracy, saving.')
                    prev_val_accuracy = v_accuracy
                    saver.save(sess,self.checkpoint_file, global_step = global_step)

                logging.info('*'*32)
            logging.info("Done Training")



































