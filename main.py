import pandas as pd
from elm import ELM
from helper import Helper

datasets = ['datasets/wisconsin_transformed.csv', '']
debug = False
h = Helper()
h.get_dataset(datasets[0])

train, test = h.split_dataset()

neural_network = ELM(input_size = 9, output_layer_size = 2)

neural_network.add_neuron(9, "linear")
neural_network.add_neuron(2, "sigmoid")

output_classes = []
print(len(train))
for item in train.values:
	#item[:len(item)-1]
	neural_network.train(item[:len(item)-1])
	
	output_classes.append(item[len(item)-1])
neural_network.update_beta(output_classes) #create output_weights


if debug:
	print("checking correctness of shapes:")
	print(neural_network.input_weights[0].shape)
	print(len(neural_network.input_weights))
	print(neural_network.H.shape)
	print(len(neural_network.bias))