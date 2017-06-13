import numpy as np
import random
from neuron import Neuron

from numpy.linalg import inv

class ELM(object):
	"""docstring for ELM"""
	def __init__(self, input_size, output_layer_size):
		super(ELM, self).__init__()
		self.H = None
		self.bias = []
		self.input_size = input_size
		self.hidden_layer = []
		self.input_weights = []
		self.beta = np.zeros((input_size, 1))
		self.output_weights = np.random.rand(len(self.hidden_layer), output_layer_size)
		self.outputs = np.zeros((output_layer_size, 1))


	def train(self, input, **kwargs):
		outputs = []
		
		for i in range(len(self.hidden_layer)):
			outputs.append(self.hidden_layer[i].activate( self.input_weights[i].dot(input) +  self.bias[i]))
		#print(outputs)
		
		self.H = np.vstack([self.H, np.array(outputs)]) #add row to H matrix
		
	
		#print(self.H.shape)


	def update_beta(self, output):
		output = np.array(output)
		alpha = 50*0.001
		self.output_weights = inv((self.H.T.dot(self.H)) + np.identity(self.H.shape[1])*alpha).dot(self.H.T).dot(output.T)
		
	def predict(self, input):
		pass


	def add_neuron(self, amount, kind = 'linear'):
		for i in range(amount):
			self.hidden_layer.append(Neuron(kind))
			self.set_correspondent_weight_and_bias(kind)
		self.H = np.empty((0, len(self.hidden_layer)))

	def set_correspondent_weight_and_bias(self, kind):
		if kind == 'linear':
			values = []
			for i in range(self.input_size):
				values.append(1)
			self.input_weights.append(np.array(values).T) #weights are set to 1
			self.bias.append(0) #bias are set to 0
		else:
			values = []
			for i in range(self.input_size):
				values.append(random.random())
			self.input_weights.append(np.array(values).T)
			self.bias.append(random.random())