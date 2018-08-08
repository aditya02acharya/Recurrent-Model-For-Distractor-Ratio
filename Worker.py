# -*- coding: utf-8 -*-
"""
Created on Fri May 18 22:32:55 2018

@author: Aditya
"""
import tensorflow as tf
import numpy as np
from AC_Network import AC_Network
from helper import *

class Worker():

    def __init__(self,game,name,s_size,a_size,dropout,trainer,model_path,global_episodes):
        self.name = "worker_" + str(name)
        self.number = name        
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.episode_features = []
        self.summary_writer = tf.summary.FileWriter("train_"+str(self.number))
        
        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(s_size,a_size,dropout,self.name,trainer)
        self.update_local_ops = update_target_graph('global',self.name)
        
        self.env = game

    def train(self,rollout,episode_step_count,sess,gamma,bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:,0]
        focus = rollout[:,1]
        actions = rollout[:,2]
        rewards = rollout[:,3]
        values = rollout[:,7]

        # Here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns. 
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])

        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]

        advantages = discount(advantages,gamma)
        

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v:discounted_rewards,
            self.local_AC.inputs:np.vstack(observations),
            self.local_AC.focus:np.vstack(focus),
            self.local_AC.trainLength:episode_step_count,
            self.local_AC.actions:actions,
            self.local_AC.advantages:advantages,
            self.local_AC.state_in[0]:self.batch_rnn_state[0],
            self.local_AC.state_in[1]:self.batch_rnn_state[1]}
        
        v_l,p_l,e_l,g_n,v_n, self.batch_rnn_state,_ = sess.run([self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.state_out,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)
        
        return v_l / len(rollout),p_l / len(rollout),e_l / len(rollout), g_n,v_n
    
    
    def work(self,max_episode_length,gamma,sess,coord,saver):
        
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print ("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_reward = 0
                episode_step_count = 0
                d = False
                
                s,f = self.env.reset()
                rnn_state = self.local_AC.state_init
                self.batch_rnn_state = rnn_state
                
                while d == False:
                    #Take an action using probabilities from policy network output.
                    a_dist,v,rnn_state = sess.run([self.local_AC.policy,self.local_AC.value,self.local_AC.state_out],
                                                  feed_dict={self.local_AC.inputs:[s],
                                                  self.local_AC.focus:[f],
                                                  self.local_AC.trainLength:1,
                                                  self.local_AC.state_in[0]:rnn_state[0],
                                                  self.local_AC.state_in[1]:rnn_state[1]})
                    
                    try:
                        a = np.random.choice(len(a_dist[0]),p=a_dist[0])
                        #a = np.argmax(a_dist == a)
                    except:
                        print("Error in action space with nan.")
                        print(a_dist[0])
                    
                    s1, f1, r, d = self.env.step(a)
                    
                    episode_buffer.append([s,f,a,r,s1,f1,d,v[0,0]])
                    episode_values.append(v[0,0])

                    episode_reward += r
                    s = s1
                    f = f1                    
                    total_steps += 1
                    episode_step_count += 1
                    
                    if d == True:
                        break
                  
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_features.append(self.env.correct)
                self.episode_mean_values.append(np.mean(episode_values))
                
                # Update the network using the episode buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,episode_step_count,sess,gamma,0.0)                    
                    
                # Periodically save model parameters, and summary statistics.
                if episode_count % 10 == 0 and episode_count != 0:
                    if self.name == 'worker_0' and episode_count % 10000 == 0:
                        saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                        print ("Saved Model")
                    mean_reward = np.mean(self.episode_rewards[-10:])
                    self.episode_rewards = []
                    mean_length = np.mean(self.episode_lengths[-10:])
                    self.episode_lengths = []
                    mean_value = np.mean(self.episode_mean_values[-10:])
                    self.episode_mean_values = []
                    mean_feature = np.mean(self.episode_features[-10:])
                    self.episode_features = []
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Perf/Accuracy', simple_value=float(mean_feature))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()
                        
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1
                                        
                    

        