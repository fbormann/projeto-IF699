import pandas as pd
from elm import ELM
from helper import Helper

datasets = ['datasets/wisconsin_transformed.csv', 'datasets/abalone.csv', 'datasets/computer_revised.csv']
debug = False
h = Helper()
h.get_dataset(datasets[2])

train, test = h.split_dataset()

neural_network = ELM(input_size = 7, output_layer_size = 1)

#neural_network.add_neuron(9, "linear")
neural_network.add_neuron(100, "sigmoid")

output_classes = []
print(len(train))
for item in train.values:
    #item[:len(item)-1]
    neural_network.train(item[:len(item)-1])
    
    
    output_classes.append(item[len(item)-1])
neural_network.update_beta(output_classes) #create output_weights


for item in h.test_dataset.values:
    print(neural_network.predict(item[:len(item)-1]))
    print(item[len(item)-1])


if debug:
    print("checking correctness of shapes:")
    print(neural_network.input_weights[0].shape)
    print(len(neural_network.input_weights))
    print(neural_network.H.shape)
    print(len(neural_network.bias))