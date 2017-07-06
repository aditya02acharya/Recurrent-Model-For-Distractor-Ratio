import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from GlobalConstants import MAX_ACTIONS

class Qnetwork(object):

	def __init__(self, h_size, rnn_cell, myScope):
		#recive 2 input feature observations and then process it through a convoluted neural network.
		self.scalarColorInput = tf.placeholder(shape=[None,36], dtype=tf.float32)
		self.scalarShapeInput = tf.placeholder(shape=[None,36], dtype=tf.float32)

		self.colorIn = tf.reshape(self.scalarColorInput,shape=[-1,6,6,1])
		self.shapeIn = tf.reshape(self.scalarShapeInput,shape=[-1,6,6,1])  
		
		#First CNN layer
		self.convC_1 = slim.convolution2d(inputs=self.colorIn, num_outputs=1, kernel_size=[3,3], stride=[1,1], padding='SAME', scope=myScope+'_convC_1')
		self.convS_1 = slim.convolution2d(inputs=self.shapeIn, num_outputs=1, kernel_size=[3,3], stride=[1,1], padding='SAME', scope=myScope+'_convS_1') 

		#Second CNN layer
		self.convC_2 = slim.convolution2d(inputs=self.convC_1, num_outputs=1,  kernel_size=[3,3], stride=[1,1], padding='SAME', scope=myScope+'_convC_2')
                self.convS_2 = slim.convolution2d(inputs=self.convS_1, num_outputs=1, kernel_size=[3,3], stride=[1,1], padding='SAME', scope=myScope+'_convS_2')

		#Thrid CNN layer
		self.convC = slim.convolution2d(inputs=self.convC_2, num_outputs=1, biases_initializer=None,  kernel_size=[3,3], stride=[1,1], padding='SAME', scope=myScope+'_convC')
		self.convS = slim.convolution2d(inputs=self.convS_2, num_outputs=1, biases_initializer=None,  kernel_size=[3,3], stride=[1,1], padding='SAME', scope=myScope+'_convS')	
		

		self.streamC = slim.flatten(self.convC)
		self.streamS = slim.flatten(self.convS)

		self.trainLength = tf.placeholder(dtype=tf.int32)
                self.batch_size = tf.placeholder(dtype=tf.int32)

		#Concat the colour and shape branch	

		self.merge = tf.concat([self.streamC, self.streamS], 1)		

		#Shape Branch.
		#self.drop_shape = tf.nn.dropout(self.merge, 0.8)        
                
                self.obsIn = tf.reshape(self.convC, [self.batch_size, self.trainLength, h_size+h_size])
        
                self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)

                self.rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs=self.obsIn, cell=rnn_cell, 
								dtype=tf.float32, sequence_length=self.trainLength , 
								initial_state=self.state_in, scope=myScope+'_rnn')

                self.rnn = tf.reshape(self.rnn,shape=[-1,h_size+h_size])
		
				
		self.hidden = slim.fully_connected(self.rnn, h_size, scope=myScope+'_hidden')		

		self.hidden2 = slim.fully_connected(self.hidden, h_size, activation_fn=tf.nn.sigmoid, scope=myScope+'_hidden2')
		
		self.Qout = slim.fully_connected(self.hidden2, MAX_ACTIONS, activation_fn=None, scope=myScope+'_Qout')
		
		
		self.predict = tf.argmax(self.Qout,1)

		#Calculate Loss.
		self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
		self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
		self.actions_onehot = tf.one_hot(self.actions,MAX_ACTIONS,dtype=tf.float32)	
		
		self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), reduction_indices=1)
		
		self.td_error = (self.targetQ - self.Q)

		self.clipped_error = tf.where(tf.abs(self.td_error) < 8.0, 0.5 * tf.square(self.td_error), 8.0 * (tf.abs(self.td_error) - 0.5 * 8.0))
		self.loss = tf.reduce_mean(self.clipped_error)
		
	
		self.trainer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
		self.updateModel = self.trainer.minimize(self.loss)

