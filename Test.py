# -*- coding: utf-8 -*-
"""
Created on Sat May 19 09:25:48 2018

@author: Aditya
"""

from GlobalConstants import MAX_ACTIONS, N_DR_ELEMENTS, N_ROWS
import tensorflow as tf
import os
import numpy as np
from time import sleep

from AC_Network import AC_Network
from Worker import Worker
from Environment import Environment
from helper import *

max_episode_length = 36
num_episodes = 1
gamma = .99 # discount rate for advantage estimation and reward discounting
s_size = N_DR_ELEMENTS # Observations are 6 * 6 matrix
a_size = MAX_ACTIONS # Agent can fixate on 36 possible locations and one stop action.
dropout=0.8
load_model = True
model_path = './Model'
episode_rewards = []
episode_lengths = []
episode_durations = []
episode_features = []
env = Environment()

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

with tf.device("/cpu:0"): 
    master_network = AC_Network(s_size,a_size,dropout,'global',None) # Generate global network
    saver = tf.train.Saver(max_to_keep=5)
    
with tf.Session() as sess:

    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    fixFile = open("fixationhistory.csv",'w')
    metricFile = open("testMetricsLog.csv",'w')
    actionFile = open("actionhistory.csv",'w')
    
    for i in range(num_episodes):
        print(i)
        episode_buffer = []
        episode_values = []
        action_history = []
        fixation_history = [0]*10
        episode_reward = 0
        previous_action = -1
        episode_step_count = 0
        d = False     
        s,f = env.reset()
        #print(env.get_display().get_colour())
        #print(env.get_display().get_shape()) 
        rnn_state = master_network.state_init
        batch_rnn_state = rnn_state
        ecc = []
                
        while d == False:
            #Take an action using probabilities from policy network output.
            a_dist,v,rnn_state,conv = sess.run([master_network.policy,master_network.value,master_network.state_out,master_network.conv],
                                           feed_dict={master_network.inputs:[s],
                                           master_network.focus:[f],
                                           master_network.trainLength:1,
                                           master_network.state_in[0]:rnn_state[0],
                                           master_network.state_in[1]:rnn_state[1]})
                    
            a = np.random.choice(len(a_dist[0]),p=a_dist[0])
            #a = np.argmax(a_dist == a)
            print(conv)
            print("Taking action " + str(a))
            action_history.append(a)                    
            s1, f1, r, d = env.step(a)

            if a < N_DR_ELEMENTS:
                        fix_x = int(a / N_ROWS)
                        fix_y = int(a % N_ROWS)
                        fixation_history[episode_step_count] = env.get_display().get_colour()[fix_x][fix_y] * 2 + env.get_display().get_shape()[fix_x][fix_y]

            episode_reward += r
            s = s1
            f = f1       
            if a < N_DR_ELEMENTS:
            	episode_step_count += 1             
                    
            if d == True:
                break

        for item in fixation_history:
            fixFile.write("%s," % (str(item)))
        fixFile.write("%s" % str(-5))
        fixFile.write("\n")

        metricFile.write("%s,%s,%s,%s,%s" % (str(episode_step_count), str(episode_reward), str(env.get_display().get_target_status()), str(a), str(env.get_display().get_same_colour_ratio())))
        metricFile.write("\n")
        
        for item in action_history:
               actionFile.write("%s," % str(item))
        actionFile.write("\n")


    fixFile.close()
    metricFile.close()
    actionFile.close()    