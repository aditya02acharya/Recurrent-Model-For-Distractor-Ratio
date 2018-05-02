import numpy as np
import random
import tensorflow as tf
import itertools
import csv
import tensorflow.contrib.slim as slim

from Qnetwork import Qnetwork
from Environment import Environment
from GlobalConstants import N_ROWS, MAX_STEPS, MAX_ACTIONS, N_DR_ELEMENTS
from helper import *
from Experience_Buffer import *


num_episodes = 10 #How many episodes of game environment to train network with.
load_model = True #Whether to load a saved model.
path = "./Model" #The path to save/load our model to/from.
h_size = 36 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
h_size = 36 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
max_epLength = 8 #The max allowed length of our episode.
summaryLength = 100

tf.reset_default_graph()
cell = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
cellT = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)

mainQN = Qnetwork(h_size,cell,'main')
targetQN = Qnetwork(h_size,cellT,'target')

init = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep=2)

#create lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0
env = Environment()

with tf.Session() as sess:
	if load_model == True:
		print ('Loading Model...')
        	ckpt = tf.train.get_checkpoint_state(path)
        	saver.restore(sess,ckpt.model_checkpoint_path)

	else:
        	sess.run(init)
	
	#for op in tf.get_default_graph().get_operations():
    	#	print str(op.name) 
	#with tf.variable_scope('main_conv', reuse=True) as scope_conv:
    	#	W_conv1 = tf.get_variable('weights')
    	#	weights = W_conv1.eval()
	#	print weights

	fixFile = open("fixationhistory.csv",'w')
	metricFile = open("testMetricsLog.csv",'w')
	actionFile = open("actionhistory.csv",'w')
	

	for i in range(num_episodes):
		print i
		s  = env.reset()
		#print "Observation:"
		#print s
		#print env.get_display().get_colour()
		#print '============================'
		#print env.get_display().get_shape()
		d = False
		rAll = 0
		j = 0
		action_history = []
		fixation_history = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		#state = (np.zeros([1,h_size]),np.zeros([1,h_size]))
		#state = np.zeros((1, h_size), np.float32)
		state = (np.zeros([1,h_size]),np.zeros([1,h_size]))                                                       
		#print state
		
		step = 0
		while j < max_epLength:
			
			a, state_next, predict, stream = sess.run([mainQN.predict,mainQN.rnn_state, mainQN.Qout, mainQN.stream],
                                                        feed_dict={mainQN.scalarInput:[s],                                
                                                        #mainQN.scalarShapeInput:[s_s],                                           
                                                        mainQN.trainLength:1,                                                    
                                                        mainQN.state_in:state,                                     
                                                        mainQN.batch_size:1})  
			
			a = a[0]
			#print "Observation After CNN: "
			#print stream
			#print "Action"
			#print a
			action_history.append(a)
			s1, r, d = env.step(a)
			#print s1_c
			#print s1_s	
			if a < N_DR_ELEMENTS and step > 0:
				fix_x = a / N_ROWS
				fix_y = a % N_ROWS
				fixation_history[step] = env.get_display().get_colour()[fix_x][fix_y] * 2 + env.get_display().get_shape()[fix_x][fix_y]
			
			if a < N_DR_ELEMENTS:
				step += 1
			j += 1
			rAll += r
			s = s1
			
			state = state_next
			#print state
			total_steps += 1

			if d == True:
				break
		
		jList.append(j)
        	rList.append(rAll)

		if len(rList) % summaryLength == 0 and len(rList) != 0:
			print (total_steps,j,np.mean(rList[-summaryLength:]))

		#if step < MAX_STEPS and step > 1:
		print "action history: "
		print action_history
		print "fixation history: "
		print fixation_history 
		for item in fixation_history:
			fixFile.write("%s," % (str(item)))
		fixFile.write("%s" % str(-5))
		fixFile.write("\n")
		metricFile.write("%s,%s,%s,%s,%s" % (str(step-1), str(rAll), str(env.get_display().get_target_status()), str(a), str(env.get_display().get_same_colour_ratio())))

		metricFile.write("\n")
		for item in action_history:
			actionFile.write("%s," % str(item))
		actionFile.write("\n")

	fixFile.close()
	metricFile.close()
	actionFile.close()
print ("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")
