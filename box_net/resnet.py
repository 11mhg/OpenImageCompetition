import tensorflow as tf
import time
import numpy as np
from classifier_data.classifier_data import * 
from tensorflow.contrib.slim.python.slim.nets import resnet_v2

slim = tf.contrib.slim
resnet_arg_scope = resnet_v2.resnet_arg_scope



def get_resnet50(inputs,reuse=False, is_training=False):
    with slim.arg_scope(resnet_arg_scope()):
        blocks = [
            resnet_v2.resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
            resnet_v2.resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
            resnet_v2.resnet_v2_block('block3', base_depth=256, num_units=6, stride=2),
            resnet_v2.resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
        ] 
        net, end_points = resnet_v2.resnet_v2(inputs,None, is_training, global_pool=False, reuse=reuse)
        print(net)
        input("Wait")
    return net, end_points

class Classifier():

    def __init__(self, label_file, dataset_dir, val_dir, es=False, log_dir='./logs/', num_epochs=1, num_iter = 10000, batch_size=32, lr = 0.0002, lr_decay=0.7):
        self.label_file = label_file
        self.save_dir = './model/'
        self.dataset_dir = dataset_dir
        self.val_dir = val_dir
        self.log_dir = log_dir
        self.num_epochs = num_epochs
        self.num_iter = num_iter
        self.batch_size = batch_size
        self.lr = lr
        self.lr_decay = lr_decay
        self.es = es

        with open(self.label_file) as f:
            self.class_names = f.readlines()
        self.ref_names = [s.strip('\n').split(',')[0] for s in self.class_names]
        self.class_names = [s.strip('\n').split(',')[-1] for s in self.class_names]

    def get_inputs(self):
        data = Data(self.label_file,image_size=(224,224,3), batch_size=self.batch_size)
        if self.es:
            train_data = data.get_batch(es=True)
            val_data = data.get_batch(es=True)
        else:
            train_data = data.get_batch(self.dataset_dir)
            val_data = data.get_batch(self.val_dir)

        self.train_iter = train_data.make_initializable_iterator()
        self.val_iter = val_data.make_initializable_iterator()

        self.train_image, self.train_label = self.train_iter.get_next()
        self.val_image, self.val_label = self.val_iter.get_next()
        self.train_image = tf.cast(self.train_image,tf.float32)
        self.val_image = tf.cast(self.val_image,tf.float32)


    def train(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        self.num_batches_per_epoch = int(self.num_iter / self.batch_size)
        self.decay_steps = int(2*self.num_batches_per_epoch)

        self.get_inputs()

        self.logits, self.end_points = get_resnet50(self.train_image, len(self.class_names), is_training=True)
        self.val_logits, self.val_end_points = get_resnet50(self.val_image,len(self.class_names),is_training=False)

        self.train_one_hot = tf.one_hot(self.train_label,len(self.class_names))
        self.val_one_hot = tf.one_hot(self.val_label, len(self.class_names))
        
        self.logits = tf.squeeze(self.logits)
        self.val_logits = tf.squeeze(self.val_logits)

        train_loss = tf.losses.softmax_cross_entropy(onehot_labels = self.train_one_hot,
                                                     logits = self.logits)
        total_train_loss = tf.losses.get_total_loss()

        val_loss = tf.losses.softmax_cross_entropy(onehot_labels = self.val_one_hot,
                                                   logits = self.val_logits)




        global_step = tf.train.get_or_create_global_step()
        lr = tf.train.exponential_decay(
                learning_rate = self.lr,
                global_step = global_step,
                decay_steps = self.decay_steps,
                decay_rate = self.lr_decay,
                staircase = True)

        optimizer = tf.train.AdamOptimizer(learning_rate = lr)
        train_op = optimizer.minimize(
                             loss=total_train_loss,
                             global_step=global_step)


        #metrics
        predictions = tf.squeeze(tf.argmax(self.end_points['predictions'],3),1)
        probabilities = self.end_points['predictions']

        accuracy, accuracy_update = tf.metrics.accuracy(predictions,self.train_label)
        train_metrics_op = tf.group(accuracy_update)

        val_predictions = tf.squeeze(tf.argmax(self.val_end_points['predictions'],3),1)
        val_prob = self.val_end_points['predictions']
        val_accuracy, val_accuracy_update = tf.metrics.accuracy(val_predictions, self.val_label)
        val_metrics_op = tf.group(val_accuracy_update)


        # summaries
        tf.summary.image('Input_image',self.train_image[:,:,:,:3])
        tf.summary.scalar('losses/Total_Loss',total_train_loss)
        tf.summary.scalar('losses/Val_loss',val_loss)
        tf.summary.scalar('accuracy',accuracy)
        tf.summary.scalar('val_accuracy',val_accuracy)
        tf.summary.scalar('learning_rate',lr)
        
        summary_op = tf.summary.merge_all()
        
        self.saver = tf.train.Saver(filename=self.save_dir,max_to_keep=2)
        
        #set config stuff here
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)


        if tf.train.checkpoint_exists(self.save_dir):
            self.saver.restore(sess,tf.train.latest_checkpoint(self.save_dir))
        
        writer = tf.summary.FileWriter(self.log_dir,self.sess.graph)
        
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        self.sess.run(init_op)
        self.sess.run(self.train_iter.initializer)
        self.sess.run(self.val_iter.initializer)
        v_accuracy = 0.0

        for e in range(self.num_epochs):
            for i in range(self.num_batches_per_epoch):
                if i % 10 == 0:
                    loss, _ = self.train_step(train_op,total_train_loss, train_metrics_op, global_step)
                    summaries = self.sess.run(summary_op)
                else:
                    loss, _ = self.train_step(train_op,total_train_loss, train_metrics_op, global_step)
            
            accuracy_value = self.sess.run(accuracy)
            tf.logging.info('Epoch %s/%s', e+1, self.num_epochs)
            tf.logging.info('Training Accuracy: %s', accuracy_value)
            

            tf.logging.info('Beginning Validation')
            for i in range(self.num_batches_per_epoch):
                v_loss = self.val_step(val_loss, val_metrics_op)
            temp_acc = self.sess.run(val_accuracy)
            if temp_acc > v_accuracy:
                v_accuracy = temp_acc
                self.saver.save(self.sess,save_path = self.save_dir,global_step=global_step)
                tf.logging.info('Better Network saved: %s accuracy in validation',v_accuracy)
            tf.logging.info('Epoch finished')
        tf.logging.info('Done Training')


    def train_step(self,train_op,loss,metrics_op,global_step):
        start_time = time.time()
        try:
            _, total_loss, global_step_count, _ = self.sess.run([train_op,loss,global_step,metrics_op])
        except:
            self.sess.run(self.train_iter.initializer)
            _, total_loss, global_step_count, _ = self.sess.run([train_op,loss,global_step,metrics_op])

        time_elapsed = time.time() - start_time

        tf.logging.info('global step %s: loss: %.4f (%.2f sec/step)', global_step_count, total_loss, time_elapsed)
        return total_loss, global_step_count

    def val_step(self,val_loss,metrics_op):
        start_time = time.time()
        try:
            val_loss,_ = self.sess.run([val_loss,metrics_op])
        except:
            self.sess.run(self.val_iter.initializer)
            val_loss,_ = self.sess.run([val_loss,metrics_op])

        time_elapsed = time.time() - start_time

        tf.logging.info('val loss: %.4f (%.2f sec/step)', val_loss, time_elapsed)
        return val_loss
