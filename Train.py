import numpy as np
import random
import tensorflow as tf
import itertools
import csv
import tensorflow.contrib.slim as slim

from Qnetwork import Qnetwork
from Environment import Environment
from GlobalConstants import MAX_ACTIONS, N_DR_ELEMENTS
from helper import *
from Experience_Buffer import *

batch_size = 10
trace_length = 1
update_freq = 10000
y = .99
startE = 1.0
endE = 0.01
anneling_steps = 250
num_episodes = 25000000
pre_train_steps = 1000
load_model = False
h_size = 36
max_epLength = 8
summaryLength = 100000
tau = 0.001
env = Environment()

tf.reset_default_graph()

cell = tf.contrib.rnn.BasicLSTMCell(num_units=h_size+h_size, state_is_tuple=True)
cellT = tf.contrib.rnn.BasicLSTMCell(num_units=h_size+h_size, state_is_tuple=True)

mainQN = Qnetwork(h_size,cell,'main')
targetQN = Qnetwork(h_size,cellT,'target')

init = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep=5)

trainables = tf.trainable_variables()

targetOps = updateTargetGraph(trainables,tau)

myBuffer = Experience_Buffer()

e = startE
stepDrop = (startE - endE)/anneling_steps

#qList = [0] * 100
rList = [0] * 10000
lList = [0] * 10000
qList = [0] * 10000
total_steps = 0

#with open('./log.csv', 'w') as myfile:
    #wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    #wr.writerow(['Episode','Length','Reward','LOG','SAL'])

with tf.Session() as sess:
	if load_model == True:
		print ('Loading Model...')
		ckpt = tf.train.get_checkpoint_state("./Model")
		saver.restore(sess,ckpt.model_checkpoint_path)
	sess.run(init)

	updateTarget(targetOps,sess)

	for i in range(num_episodes):
		episodeBuffer = []
		s_c, s_s = env.reset()
		d = False
		rAll = 0
		j = 0
		step = 0
		loss = 0
		q_val = 0
	
		#reset recurrent hidden state.
		state = (np.zeros([1,h_size]),np.zeros([1,h_size]))
		#state = np.zeros([1,h_size])
		#Q-Learning.
		while j < max_epLength:
			j+=1
		
			#Choose action greedily with epsilon chance of random action.
			if np.random.rand(1) < e or total_steps < pre_train_steps:
				state_next, predict = sess.run([mainQN.rnn_state, mainQN.Qout], 
										feed_dict={mainQN.scalarColorInput:[s_c], 
										#mainQN.scalarShapeInput:[s_s], 
										#mainQN.scalarFocusInput:focus, 
										mainQN.trainLength:1, 
										mainQN.state_in:state, 
										mainQN.batch_size:1})
				a = np.random.randint(0, MAX_ACTIONS)
			else:
				a, state_next, predict = sess.run([mainQN.predict,mainQN.rnn_state, mainQN.Qout], 
							feed_dict={mainQN.scalarColorInput:[s_c], 
							#mainQN.scalarShapeInput:[s_s], 
							#mainQN.scalarFocusInput:focus, 
							mainQN.trainLength:1, 
							mainQN.state_in:state,  
							mainQN.batch_size:1})
				a = a[0]

			s1_c, s1_s, r, d = env.step(a)

			if a < N_DR_ELEMENTS:
				step += 1			

			total_steps += 1
			episodeBuffer.append(np.reshape(np.array([s_c, s_s, a, r, s1_c, s1_s, d]),[1,7]))  
			if total_steps > pre_train_steps:
				if e > endE:
					e -= stepDrop

				if total_steps % (update_freq) == 0:
					updateTarget(targetOps,sess)
				#Reset the recurrent layer's hidden state
				#state_train = (np.zeros([batch_size,h_size]),np.zeros([batch_size,h_size]))		
				if total_steps % 4 == 0:
					trainBatch  = myBuffer.sample(batch_size, trace_length)
					#state_train = (np.vstack(trainBatch[:,8]), np.vstack(trainBatch[:,9]))
					state_train = (np.zeros([batch_size,h_size]),np.zeros([batch_size,h_size]))
					#Perform the Double-DQN update.
					Q1 = sess.run(mainQN.predict,feed_dict={\
                        			mainQN.scalarColorInput:np.vstack(trainBatch[:,4]), 
						#mainQN.scalarShapeInput:np.vstack(trainBatch[:,5]), 
                        			mainQN.trainLength:trace_length,mainQN.state_in:state_train,\
						mainQN.batch_size:batch_size})	

					Q2 = sess.run(targetQN.Qout,feed_dict={\
                        			targetQN.scalarColorInput:np.vstack(trainBatch[:,4]), 
						#targetQN.scalarShapeInput:np.vstack(trainBatch[:,5]),
                        			targetQN.trainLength:trace_length,targetQN.state_in:state_train,\
						targetQN.batch_size:batch_size})
					#flip the booleans
					end_mul = -(trainBatch[:,6] - 1)
				
					doubleQ = Q2[range(batch_size*trace_length), Q1]

					targetQ = trainBatch[:,3] + (y*doubleQ * end_mul)
	
					#Update network with target values.
					
					m , l = sess.run([mainQN.updateModel, mainQN.loss], feed_dict={mainQN.scalarColorInput:np.vstack(trainBatch[:,0]),
						#mainQN.scalarShapeInput:np.vstack(trainBatch[:,1]),
						mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,2], mainQN.trainLength:trace_length, mainQN.state_in:state_train,\
						mainQN.batch_size:batch_size})
					lList[total_steps%10000] = l
			rAll += r
			q_val += predict[0, a]
			s_c = s1_c
			s_s = s1_s
			#focus = np.copy(focus_next)	
			state = state_next
			if d == True:
				break

		bufferArray = np.array(episodeBuffer)
		episodeBuffer = zip(bufferArray)
		myBuffer.add(episodeBuffer)
		#lList[i%10000] = loss
		rList[i%10000] = rAll
		qList[i%10000] = q_val

		#Periodically save model.
		if i % summaryLength  == 0 and i != 0:
			saver.save(sess, './Model/model-'+str(i)+'.cptk')
			print ("Saved Model")

		if i % 10000  == 0 and i != 0:
			print (i, step, np.round(np.mean(rList), decimals=2), np.round(np.mean(qList), decimals=2), np.round(np.mean(lList), decimals=2), np.round(e, decimals=2))
			#print myBuffer.sample(1, 1)
			#print ("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")
			#saveToCenter(i, rList, jList, np.reshape(np.array(episodeBuffer), [len(episodeBuffer),7]), summaryLength,h_size,sess,mainQN)
	saver.save(sess, './Model/model-'+str(i)+'.cptk')
	
	 	