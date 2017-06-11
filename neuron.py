import numpy as np
class Neuron():
	def __init__(self, activation_kind):
		self.activation_function = self.get_activation_function(activation_kind)

	def activate(self, input):
		#print("activate value is: ", input)
		return self.activation_function(input)

	def get_activation_function(self, kind):
		if kind == "linear":
			function = lambda x: x
		elif kind == "sigmoid":
			function = lambda x: 1/(1 + np.e**((-1)*x) )
		return function

