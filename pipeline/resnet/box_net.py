from .resnet import get_box_resnet
from .utils import get_flags
from data.data import Data
import numpy as np
import os, sys
import time
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.slim.python.slim.nets import resnet_v2

slim = tf.contrib.slim


class Box_net():

    def __init__(self,FLAGS):
        self.flags = FLAGS
        self.num_epochs_before_decay=2
        self.lr_decay = 0.7
        self.lr = 0.0002
        self.checkpoint_file = self.flags.model_dir+'model.ckpt'
        if self.flags.dtype == 'float16':
            self.dtype = 'float16'
        else:
            self.dtype = 'float32'

    def ready_dataset(self):
        data = Data(classification=False, classes_text=self.flags.labels, batch_size = self.flags.batch_size)
        self.train_dataset = data.get_batch(self.flags.data_dir)
        self.val_dataset = data.get_batch(self.flags.val_dir)
        self.train_iter = self.train_dataset.make_initializable_iterator()
        self.val_iter = self.val_dataset.make_initializable_iterator()

        self.train_image, self.train_label, self.train_box = self.train_iter.get_next()
        self.val_image, self.val_label, self.val_box = self.val_iter.get_next()

        #put labels in -1 to 1 range
        self.train_box = tf.multiply(self.train_box,tf.constant(2,dtype=tf.float32)) - tf.constant(1,dtype=tf.float32)
        self.val_box = tf.multiply(self.val_box,tf.constant(2,dtype=tf.float32)) - tf.constant(1,dtype=tf.float32)

        t_xy = self.train_box[:,:2]
        t_wh = tf.log(self.train_box[:,2:])
        self.train_box = tf.concat([t_xy,t_wh],-1)

        v_xy = self.val_box[:,:2]
        v_wh = tf.log(self.val_box[:,2:])
        self.val_box = tf.concat([v_xy, v_wh],-1)

        print(self.train_box)
        input("Wait")

    def train_step(self, sess, train_op, metrics_op, global_step):
        start_time = time.time()
        total_loss, global_step_count, _ = sess.run([train_op, global_step, metrics_op])
        time_elapsed = time.time() - start_time
        logging.info('global_step %s: loss %.4f (%.2f sec/step)', global_step_count, total_loss, time_elapsed)
        return total_loss, global_step_count

    def val_step(self, sess, val_op, val_metric):
        v_loss, v_acc = sess.run([val_op, val_metric])
        return v_loss, v_acc

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

        train_out, train_attn = get_box_resnet(self.train_image,is_training=True)
        
        if self.dtype == 'float16':
            train_out = tf.cast(train_out, tf.float32)

        loss = tf.losses.mean_squared_error(self.train_box,train_out)
        total_loss = tf.losses.get_total_loss()
        
        global_step = tf.train.get_or_create_global_step()

        lr = tf.train.exponential_decay(
                learning_rate = self.lr,
                global_step = global_step,
                decay_steps = decay_steps,
                decay_rate = self.lr_decay,
                staircase=True)

#        loss_scale = 128. if self.dtype=='float16' else 1.
#
#        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
#        grad_mult = {variables[i]:1./loss_scale for i in range(0,len(variables))}

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
#        train_op = slim.learning.create_train_op(loss * loss_scale, optimizer, gradient_multipliers=grad_mult)
        train_op = slim.learning.create_train_op(total_loss, optimizer)


        rmse, rmse_update = tf.metrics.root_mean_squared_error(self.train_box,train_out) 
        metrics_op = tf.group(rmse_update)


        #train summaries
        train_image_summ = tf.summary.image('train/input_images',self.train_image[:5,:,:,:3])
        train_mask_summ = tf.summary.image('train/input_masks',self.train_image[:5,:,:,3:])
        train_attn_summ = tf.summary.image('train/attn_out',train_attn[:5,:,:,:])
        train_loss_summ = tf.summary.scalar('train/train_loss',total_loss)
        train_rmse_summ = tf.summary.scalar('train/rmse',rmse)
        train_lr_summ = tf.summary.scalar('train/learning_rate',lr)

        train_summary_op = tf.summary.merge([train_image_summ,train_attn_summ,
            train_mask_summ,train_loss_summ,train_rmse_summ,train_lr_summ])

        #validation
        val_out, val_attn = get_box_resnet(self.val_image,is_training=False)

        if self.dtype=='float16':
            val_out = tf.cast(val_out,tf.float32)

        val_loss = tf.sqrt(tf.reduce_sum(tf.square(self.val_box - val_out)))

        val_rmse, val_rmse_update = tf.metrics.root_mean_squared_error(self.val_box, val_out)

        val_metrics_op = tf.group(val_rmse_update)

        #summaries
        val_image_summ = tf.summary.image('val/input_images',self.val_image[:5,:,:,:3])
        val_mask_summ = tf.summary.image('val/input_masks',self.val_image[:5,:,:,3:])
        val_attn_summ = tf.summary.image('val/attn_out',val_attn[:5,:,:,:])
        val_loss_summ = tf.summary.scalar('val/train_loss',val_loss)
        val_acc_summ = tf.summary.scalar('val/rmse',val_rmse)

        val_summary_op = tf.summary.merge([val_image_summ, val_mask_summ,
            val_attn_summ,val_loss_summ,val_acc_summ])

        saver = tf.train.Saver(max_to_keep=4)
        
        
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
                learning_rate_value, rmse_value = sess.run([lr,rmse])
                logging.info('Current lr: %s', learning_rate_value)
                logging.info('Current training rmse: %s',rmse_value)

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

                v_rmse = sess.run([val_rmse])[0]
                logging.info('Num validation samples processed: %s',count)
                logging.info('Validation Loss: %s',v_loss)
                logging.info('Validation RMSE: %s',v_rmse)
                
                if v_rmse <= prev_val_rmse:
                    logging.info('Better validation RMSE, saving.')
                    prev_val_rmse = v_rmse
                    saver.save(sess,self.checkpoint_file, global_step = global_step)

                logging.info('*'*32)
            logging.info("Done Training")



































