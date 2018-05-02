import numpy as np
import random

class Experience_Buffer(object):

	def __init__(self, buffer_size=100000):
		self.buffer = []
		self.buffer_size = buffer_size

	def add(self, experience):
		if len(self.buffer)  >= self.buffer_size:
			self.buffer[random.randint(0,self.buffer_size-1)] = experience
		else:
			self.buffer.append(experience)


	def sample(self,batch_size,trace_length):
		sampled_episodes = random.sample(self.buffer,batch_size)
		sampledTraces = []
		for episode in sampled_episodes:
			point = np.random.randint(0,len(episode)+1-trace_length)
			sampledTraces.append(episode[point:point+trace_length])
		sampledTraces = np.array(sampledTraces)
		return np.reshape(sampledTraces,[batch_size*trace_length,9])
			
	def seq_sample(self, batch_size, trace_length):
		sampled_episodes = random.sample(self.buffer,batch_size)
		sampledTraces = []
		for episode in sampled_episodes:
			trace_length = len(episode)
			sampledTraces.append(episode)
		sampledTraces = np.array(sampledTraces)
		return np.reshape(sampledTraces,[batch_size*trace_length,5]), trace_length		
