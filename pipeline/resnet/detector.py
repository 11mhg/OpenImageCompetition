from .resnet import get_class_resnet, get_box_resnet
from .utils import get_flags
from data.data import Data
import numpy as np
import cv2
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

    def ready_dataset(self):
        self.train_data = Data(classification=False, classes_text=self.flags.labels, batch_size=self.flags.batch_size)
        self.num_classes = len(self.train_data.class_names)
        self.train_data.get_oid(self.flags.data_dir,data_type='train')
        self.train_gen = self.train_data.get_generator()()
        self.val_data = Data(classification=False,classes_text=self.flags.labels,batch_size=self.flags.batch_size)
        self.val_data.get_oid(self.flags.data_dir,data_type='val')
        self.val_gen = self.val_data.get_generator()()
        self.img = tf.placeholder(tf.float32,(self.flags.batch_size,416,416,4))
        self.box = tf.placeholder(tf.float32,(self.flags.batch_size,4))

        b_xy = (self.box[:,:2]*2.0)-1.0
        b_wh = tf.log(self.box[:,2:])
        self.box = tf.concat([b_xy,b_wh],axis=-1)


        self.label = tf.placeholder(tf.int64,(self.flags.batch_size,))
        self.class_img = tf.placeholder(tf.float32,(self.flags.batch_size,200,200,4))
        logging.info("Dataset is Ready!")

    def get_val_batch(self):
        images,boxes,labels,img_names,img_indices,rand_indices=[],[],[],[],[],[]
        for _ in range(self.flags.batch_size):
            img, box, l, img_name, img_ind, rand_ind = next(self.val_gen)
            images.append(img)
            boxes.append(box)
            labels.append(l)
            img_names.append(img_name)
            img_indices.append(img_ind)
            rand_indices.append(rand_ind)
        return np.array(images), np.array(boxes), np.array(labels), np.array(img_names), np.array(img_indices),np.array(rand_indices)

    def get_train_batch(self):
        images, boxes, labels, img_names, img_indices, rand_indices = [], [], [], [], [], []
        for _ in range(self.flags.batch_size):
            img, box, l, img_name, img_ind, rand_ind = next(self.train_gen)
            images.append(img)
            boxes.append(box)
            labels.append(l)
            img_names.append(img_name)
            img_indices.append(img_ind)
            rand_indices.append(rand_ind)
        return np.array(images), np.array(boxes), np.array(labels), np.array(img_names), np.array(img_indices),np.array(rand_indices)

    def train(self):
        num_batches_per_epoch = int(np.floor(self.flags.steps_per_epoch / self.flags.batch_size))
        num_steps_per_epoch = num_batches_per_epoch
        decay_steps = int(self.num_epochs_before_decay * num_steps_per_epoch)

        if self.dtype == 'float16':
            self.img = tf.cast(self.img, tf.float16)
        self.training=tf.placeholder(tf.bool,name="training")
        logging.info("Initializing box net")
        with tf.device('/gpu:0'):
            box_out, attn = get_box_resnet(self.img,is_training=self.training)
            logging.info("Done with box net")
            if self.dtype == 'float16':
                box_out = tf.cast(box_out,tf.float32)
            
            logging.info("Creating loss for box net")
            box_loss = tf.losses.mean_squared_error(self.box,box_out)
            
            box_loss = tf.losses.get_regularization_loss(scope = 'box_net',name='box_regularization_loss') + box_loss
            logging.info("Box net loss done")
            global_step = tf.train.get_or_create_global_step()

            lr = tf.train.exponential_decay(
                    learning_rate = self.lr,
                    global_step = global_step,
                    decay_steps = decay_steps,
                    decay_rate = self.lr_decay,
                    staircase=True)
            loss_scale = 128. if self.dtype=='float16' else 1

            box_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='box_net')
            box_grad_mult = {box_variables[i]: 1./loss_scale for i in range(0,len(box_variables))}

            box_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            box_train_op = slim.learning.create_train_op(box_loss * loss_scale, box_optimizer,
                                                         colocate_gradients_with_ops=True)
            logging.info("Box net optimizer initialized")
            rmse, rmse_update = tf.metrics.root_mean_squared_error(self.box,box_out)

            #box summaries
            train_image_summ = tf.summary.image('train/box/input_images',self.img[:5,:,:,:3])
            train_mask_summ = tf.summary.image('train/box/input_masks',self.img[:5,:,:,3:])
            train_attn_summ = tf.summary.image('train/attn',attn[:5,:,:,:])
            train_boxloss_summ = tf.summary.scalar('train/box/train_loss',box_loss)
            train_rmse_summ = tf.summary.scalar('train/rmse',rmse)
            train_lr_summ = tf.summary.scalar('train/learning_rate',lr)

            train_boxsumm_op = tf.summary.merge([train_image_summ,train_mask_summ,train_attn_summ,
                                             train_boxloss_summ, train_rmse_summ,train_lr_summ])

            val_image_summ = tf.summary.image('val/box/input_images',self.img[:5,:,:,:3])
            val_mask_summ = tf.summary.image('val/box/input_masks',self.img[:5,:,:,3:])
            val_attn_summ = tf.summary.image('val/attn',attn[:5,:,:,:])
            val_loss_summ = tf.summary.scalar('val/box/val_loss',box_loss)
            val_rmse_summ = tf.summary.scalar('val/rmse',rmse)

            val_box_op = tf.summary.merge([val_image_summ,val_mask_summ,val_attn_summ,
                                           val_loss_summ,val_rmse_summ])
        logging.info("Box net summaries merged")
        #classifier
        if self.dtype=='float16':
            self.class_img = tf.cast(self.class_img,tf.float16)
        logging.info("Beginning classifier initialization")
        with tf.device('/gpu:1'):
            class_logits, class_endpoints = get_class_resnet(self.class_img,self.num_classes,is_training=self.training)
            one_hot_labels = tf.one_hot(self.label,self.num_classes)
            class_logits = tf.squeeze(class_logits,[1,2])

            if self.dtype=='float16':
                class_logits = tf.cast(class_logits,tf.float32)
            logging.info("Classifier initialization done")
            
            class_loss = tf.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, logits=class_logits)
            class_loss = tf.losses.get_regularization_loss(scope='classifier',name='classifier_reg_loss') + class_loss
            
            class_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='classifier')
            class_grad_mult = {class_variables[i]: 1./loss_scale for i in range(0,len(class_variables))}

            class_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            class_train_op = slim.learning.create_train_op(class_loss * loss_scale, class_optimizer,
                                                           colocate_gradients_with_ops=True)
            logging.info("Classifier optimizer created")
            class_predictions = tf.squeeze(tf.argmax(class_endpoints['predictions'],-1),[1])
            probabilities = class_endpoints['predictions']

            accuracy, accuracy_update = tf.metrics.accuracy(self.label, class_predictions)

             #classifier summaries
            train_class_image_summ = tf.summary.image('train/class/input_images',self.class_img[:5,:,:,:3])
            train_class_mask_summ = tf.summary.image('train/class/input_masks',self.class_img[:5,:,:,3:])
            train_class_loss_summ = tf.summary.scalar('train/class/train_loss',class_loss)
            train_class_acc_summ = tf.summary.scalar('train/accuracy',accuracy)

            train_class_op = tf.summary.merge([train_class_image_summ, train_class_mask_summ,
                                             train_class_loss_summ, train_class_acc_summ])

            val_class_image_summ = tf.summary.image('val/class/input_images',self.class_img[:5,:,:,:3])
            val_class_mask_summ = tf.summary.image('val/class/input_masks',self.class_img[:5,:,:,3:])
            val_class_loss_summ = tf.summary.scalar('val/class/val_loss',class_loss)
            val_class_acc_summ = tf.summary.scalar('val/class/accuracy',accuracy)

            val_class_op = tf.summary.merge([val_class_image_summ, val_class_mask_summ,
                                           val_class_loss_summ, val_class_acc_summ])
        logging.info("Classifier summaries merged")
        saver = tf.train.Saver(max_to_keep=4)
        prev_val_accuracy= 0.0
        prev_val_rmse = np.inf

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


            val_counter = 0
            avg_time=0
            for epoch in range(self.flags.num_epochs):
                for step in range(num_steps_per_epoch):
                    try:
                        t1 = time.time()
                        if step % 10==0: 
                            images,boxes,labels,img_names,img_indices,rand_indices = self.get_train_batch()
                            loss, step_count, _, attention,train_summaries = sess.run([box_train_op,global_step,rmse_update,attn,train_boxsumm_op],
                                    feed_dict={self.img:images,self.box:boxes,self.training:True})
                            summary_writer.add_summary(train_summaries,step_count)
                        else:
                            images,boxes,labels,img_names,img_indices,rand_indices=self.get_train_batch()
                            loss, step_count, _, attention = sess.run([box_train_op,global_step,rmse_update,attn],
                                    feed_dict={self.img:images,self.box:boxes,self.training:True})
                        for i in range(images.shape[0]):
                            img_name = img_names[i]
                            img_index = img_indices[i]
                            rand_index = rand_indices[i]
                            if rand_index > 0:
                                logging.info("*"*128*128)
                            attention_map = attention[i,:,:,:]
                            attention_map = cv2.resize(attention_map,(416,416))
                            attention_map = np.array(attention_map,np.float32)
                            if rand_index+1 < self.train_data.all_sorted_inds[img_index].shape[0]:
                                filedir = self.flags.data_dir+'masks/'+os.path.splitext(os.path.basename(
                                    img_name))[0]+'_'+str(rand_index)+'.npy'
                                np.save(filedir,attention_map)
                                self.train_data.masked[img_index][self.train_data.all_sorted_inds[img_index][rand_index]] = True
                        time_elapsed = time.time() - t1
                        if step % 10 == 0:
                            logging.info('global_step %s: loss %.4f (%.2f sec/step)',step_count,loss,time_elapsed)


                    except KeyboardInterrupt:
                        logging.info("Keyboard interrupt")
                        sys.exit(0)

                logging.info('Epoch %s/%s', epoch+1, self.flags.num_epochs)

