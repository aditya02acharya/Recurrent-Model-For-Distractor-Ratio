# -*- coding: utf-8 -*-
"""
Created on Fri May 18 23:29:47 2018

@author: Aditya
"""
from GlobalConstants import MAX_ACTIONS, N_DR_ELEMENTS
import tensorflow as tf
import os
import multiprocessing
import threading
from time import sleep

from AC_Network import AC_Network
from Worker import Worker
from Environment import Environment

max_episode_length = 9
gamma = .99 # discount rate for advantage estimation and reward discounting
s_size = N_DR_ELEMENTS # Observations are 6 * 6 matrix
a_size = MAX_ACTIONS # Agent can fixate on 36 possible locations and two stop action.
dropout=0.8
load_model = False
model_path = './Model'

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

with tf.device("/cpu:0"): 
    global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
    master_network = AC_Network(s_size,a_size,dropout,'global',None) # Generate global network
    num_workers = multiprocessing.cpu_count() # Set workers to number of available CPU threads
    
    workers = []
    # Create worker classes
    for i in range(int(num_workers/2)):
        workers.append(Worker(Environment(),i,s_size,a_size,dropout,trainer,model_path,global_episodes))
    saver = tf.train.Saver(max_to_keep=1)
    
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
        
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)

