#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Author:Uzw
Date: 
"""
#tensorflow 1.10
'''
tf2中的tf.contrib.layers.xavier_initializer变为tf.keras.initializers.glorot_normal
xavier的可变参数：uniform: 使用uniform或者normal分布来随机初始化。
                seed: 可以认为是用来生成随机数的seed
                dtype: 只支持浮点数
'''

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

tf.set_random_seed(666)

# Xavier初始化器
v2_cons = tf.get_variable('Xavier', shape=[2, 2], initializer=tf.contrib.layers.xavier_initializer(seed=0))
# 正太分布初始化器
v1_nor = tf.get_variable('v1_nor', shape=[2, 2], initializer=tf.random_normal_initializer())
v2_nor = tf.get_variable('v2_nor', shape=[2, 2],
                         initializer=tf.random_normal_initializer(mean=0, stddev=5, seed=0))  # 均值、方差、种子值
# 截断正态分布初始化器
v1_trun = tf.get_variable('v1_trun', shape=[2, 2], initializer=tf.truncated_normal_initializer())
v2_trun = tf.get_variable('v2_trun', shape=[2, 2],
                          initializer=tf.truncated_normal_initializer(mean=0, stddev=5, seed=0))  # 均值、方差、种子值
# 均匀分布初始化器
v1_uni = tf.get_variable('v1_uni', shape=[2, 2], initializer=tf.random_uniform_initializer())
v2_uni = tf.get_variable('v2_uni', shape=[2, 2],
                         initializer=tf.random_uniform_initializer(maxval=-1., minval=1., seed=0))  # 最大值、最小值、种子值

b = tf.get_variable("b",  initializer=tf.constant(0.001, shape=[17]))
tf.reshape(pred, [-1, self.num_steps, self.num_tags])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # print("常量初始化器v1_cons:", sess.run(v1_cons))
    print("Xavier初始化器", sess.run(v2_cons))
    print(b.shape)
    print("b: ", sess.run(b))
    print("正太分布初始化器v2_nor:", sess.run(v2_nor))
    print("截断正态分布初始化器v1_trun:", sess.run(v1_trun))
    print("截断正态分布初始化器v2_trun:", sess.run(v2_trun))
    print("均匀分布初始化器v1_uni:", sess.run(v1_uni))
    print("均匀分布初始化器v2_uni:", sess.run(v2_uni))




# W1 = tf.get_variable("W1", [2, 2], initializer=tf.contrib.layers.xavier_initializer(seed=0))
# W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
#
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print("dic:", sess.run(W1))