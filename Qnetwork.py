import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from helper import *
from GlobalConstants import MAX_ACTIONS

class Qnetwork(object):

	def __init__(self, h_size, rnn_cell, myScope):
		#recive 2 input feature observations and then process it through a convoluted neural network.
		self.scalarInput = tf.placeholder(shape=[None,72], dtype=tf.float32)
		#self.scalarShapeInput = tf.placeholder(shape=[None,36], dtype=tf.float32)

		self.Input = tf.reshape(self.scalarInput,shape=[-1,6,6,2])
		#self.shapeIn = tf.reshape(self.scalarShapeInput,shape=[-1,6,6,1])  
		
		#First CNN layer
		self.conv1 = slim.convolution2d(inputs=self.Input, num_outputs=1, kernel_size=[1,1], stride=[1,1], biases_initializer=None, activation_fn=tf.nn.relu, padding='SAME', scope=myScope+'_convC_1')
		#self.convS_1 = slim.convolution2d(inputs=self.shapeIn, num_outputs=1, kernel_size=[3,3], stride=[1,1], padding='SAME', scope=myScope+'_convS_1') 

		#Second CNN layer
		#self.conv = slim.convolution2d(inputs=self.Input, num_outputs=1,  kernel_size=[3,3], stride=[1,1], padding='SAME', biases_initializer=None, activation_fn=tf.nn.sigmoid, scope=myScope+'_conv')
                #self.convS_2 = slim.convolution2d(inputs=self.convS_1, num_outputs=1, kernel_size=[3,3], stride=[1,1], padding='SAME', scope=myScope+'_convS_2')

		#Thrid CNN layer
		#self.convC = slim.convolution2d(inputs=self.convC_2, num_outputs=1, biases_initializer=None,  kernel_size=[3,3], stride=[1,1], padding='SAME', scope=myScope+'_convC')
		#self.convS = slim.convolution2d(inputs=self.convS_2, num_outputs=1, biases_initializer=None,  kernel_size=[3,3], stride=[1,1], padding='SAME', scope=myScope+'_convS')	
		
		self.trainLength = tf.placeholder(dtype=tf.int32)
                self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])

		#self.stream = slim.fully_connected(slim.flatten(self.conv1), h_size, activation_fn=tf.nn.sigmoid, scope=myScope+'_hidden_init')		

		self.stream = tf.reshape(slim.flatten(self.conv1),[self.batch_size,self.trainLength,h_size])

		#Shape Branch.
		#self.drop = tf.nn.dropout(self.stream, 0.8)        
        
                self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)

                self.rnn_out, self.rnn_state = tf.nn.dynamic_rnn(inputs=self.stream, cell=rnn_cell, 
								dtype=tf.float32,sequence_length=self.trainLength,
								initial_state=self.state_in, scope=myScope+'_rnn')

                self.rnn_out = tf.reshape(self.rnn_out,shape=[-1,h_size])

		self.hidden = slim.fully_connected(self.rnn_out, h_size, activation_fn=tf.nn.sigmoid, scope=myScope+'_hidden')
		
		#Dueling Architecture.
		self.AW = tf.Variable(tf.random_normal([h_size,MAX_ACTIONS]))
        	self.VW = tf.Variable(tf.random_normal([h_size,1]))
        	self.Advantage = tf.matmul(self.hidden,self.AW)
        	self.Value = tf.matmul(self.hidden,self.VW)
        
        	#Then combine them together to get our final Q-values.
        	self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,reduction_indices=1,keep_dims=True))		
		#self.hidden = slim.fully_connected(self.rnn, h_size, scope=myScope+'_hidden')		

		#self.hidden2 = slim.fully_connected(self.rnn_out, h_size, activation_fn=tf.nn.sigmoid, scope=myScope+'_hidden2')
		
		#self.Qout = slim.fully_connected(self.rnn_out, MAX_ACTIONS, activation_fn=None, scope=myScope+'_Qout')

				

		self.max_val = tf.reduce_max(self.Qout)	
		self.predict = tf.argmax(self.Qout,1)

		#Calculate Loss.
		self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
		self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
		self.actions_onehot = tf.one_hot(self.actions,MAX_ACTIONS,dtype=tf.float32)	
		
		self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), reduction_indices=1)
		
		#self.td_error = tf.square(tf.stop_gradient(self.targetQ) - self.Q)
		
		self.clipped_error = huber_loss(self.targetQ, self.Q, 1.0)
		
		#mask = tf.one_hot(self.trainLength-1,self.trainLength,dtype=tf.float32)
		self.maskA = tf.zeros([self.batch_size,self.trainLength-1])
        	self.maskB = tf.ones([self.batch_size,1])
        	self.mask = tf.concat([self.maskA,self.maskB],1)
        	self.mask = tf.reshape(self.mask,[-1])	

        	self.loss = tf.reduce_mean(self.clipped_error * self.mask)
		
	
		self.trainer = tf.train.RMSPropOptimizer(0.001)
		self.updateModel = self.trainer.minimize(self.loss)

