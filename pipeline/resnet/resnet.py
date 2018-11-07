from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
from absl import app
import tensorflow as tf
import os
import sys
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.slim.python.slim.nets import resnet_v2

slim = tf.contrib.slim


def float32_variable_storage_getter(getter, name, shape=None, dtype=None,
                                    initializer=None, regularizer=None,
                                    trainable=True,*args,**kwargs):
    """Custom variable getter that forces trainable variables to be stored in 
    float 32 precision and casts them to the training precision.
    This particular variable getter comes from the nvidia docs, all credit 
    to them"""
    storage_dtype = tf.float32 if trainable else dtype

    variable = getter(name, shape, dtype=storage_dtype,
                      initializer=initializer, regularizer=regularizer,
                      trainable=trainable,
                      *args, **kwargs)
    if trainable and dtype != tf.float32:
        variable = tf.cast(variable, dtype)
    return variable


def gradients_with_loss_scaling(loss, variables, loss_scale=128):
    return [grad / loss_scale for grad in tf.gradients(loss*loss_scale, variables)]


def get_class_resnet(inputs,num_classes,is_training=False):
    with tf.variable_scope("classifier",custom_getter=float32_variable_storage_getter):
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, end_points = resnet_v2.resnet_v2_50(inputs,num_classes,reuse=tf.AUTO_REUSE,is_training=is_training)
    return logits, end_points


def get_box_resnet(inputs, is_training=False):
    with tf.variable_scope("box_net",custom_getter=float32_variable_storage_getter):
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            out, end_points = resnet_v2.resnet_v2_50(inputs, num_classes=None, global_pool = False, reuse=tf.AUTO_REUSE, is_training=is_training)
            l2_reg = tf.contrib.layers.l2_regularizer(scale=0.1)
            attn = tf.layers.conv2d(out,2048,[1,1],activation=None,name='attn',kernel_regularizer=l2_reg,reuse=tf.AUTO_REUSE)
            attn = tf.reduce_mean(attn,[3],name='attn_pool',keepdims=True)

#            attn = tf.layers.conv2d(out, 64, [1,1], padding='same',activation=tf.nn.leaky_relu,name='attn1',reuse=tf.AUTO_REUSE)
#            attn = tf.layers.conv2d(attn, 32, [1,1], padding='same',activation=tf.nn.leaky_relu,name='attn2',reuse=tf.AUTO_REUSE)
#            attn = tf.layers.conv2d(attn, 1,[1,1],padding='valid', activation=tf.nn.sigmoid,name='attn3',reuse=tf.AUTO_REUSE)
#            attn = tf.layers.conv2d(attn, 2048,[1,1],padding='same',activation=None,use_bias=False,kernel_initializer=tf.initializers.ones,name='attn4',trainable=False,reuse=tf.AUTO_REUSE)
            out = tf.multiply(attn,out)
#            out = tf.reduce_mean(out,[1,2],name='pool6',keepdims=True)
            out = tf.layers.conv2d(out, 512,[3,3],padding='same',activation=None,name='box',reuse=tf.AUTO_REUSE)
            out = tf.layers.flatten(out,name='box_flatten')
            box_out = tf.layers.dense(out, 4, activation=None, name='box_out',reuse=tf.AUTO_REUSE)

#            box_out = tf.squeeze(box_out,[1,2])
    return box_out, attn

