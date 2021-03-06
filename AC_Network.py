# -*- coding: utf-8 -*-
"""
Created on Fri May 18 19:26:11 2018

@author: Aditya
"""
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from helper import *

class AC_Network():
    
    def __init__(self,s_size,a_size,dropout,scope,trainer):
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=[None,s_size+s_size],dtype=tf.float32)
            self.focus = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
            self.reshapedInput = tf.reshape(self.inputs,shape=[-1,6,6,2])
            self.trainLength = tf.placeholder(dtype=tf.int32)

            self.conv = slim.convolution2d(inputs=self.reshapedInput, num_outputs=1, 
                                            kernel_size=[1,1], stride=[1,1],
                                            activation_fn=tf.nn.relu, padding='SAME')           

            hidden_merge = tf.concat([slim.flatten(self.conv), self.focus], 1)
            
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(s_size+s_size,state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            
            
            rnn_in = tf.expand_dims(hidden_merge, [0])

            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in,sequence_length=self.trainLength,time_major=False)
            
            lstm_c, lstm_h = lstm_state
            
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            
            rnn_out = tf.reshape(lstm_outputs, [-1, s_size+s_size])
            #rnn_out_drop = tf.nn.dropout(rnn_out,dropout)

            #Output layers for policy and value estimations
            self.value = slim.fully_connected(rnn_out,1,
                weights_initializer=normalized_columns_initializer(0.01),
                activation_fn=None,
                biases_initializer=None)
            self.policy = slim.fully_connected(rnn_out,a_size,
                activation_fn=tf.nn.softmax,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=None)
            
            #Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                #Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy + 1e-6))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs + 1e-6)*self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
                
                #Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))
            