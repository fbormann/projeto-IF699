import pandas as pd
from elm import ELM
from helper import Helper
from statistics import mean
import datetime

datasets = ['datasets/wisconsin_transformed.csv', 'datasets/abalone.csv', 'datasets/computer_revised.csv',
    'datasets/servo_revised.csv']
debug = False
h = Helper()
h.get_dataset(datasets[3])

train, test = h.split_dataset()

neural_network = ELM(input_size = 13, output_layer_size = 1)

#neural_network.add_neuron(9, "linear")
neural_network.add_neuron(100, "sigmoid")

output_classes = []
print(len(train))
print(datetime.datetime.now())
for item in train.values:
    #item[:len(item)-1]
    neural_network.train(item[:len(item)-1])
    
    
    output_classes.append(item[len(item)-1])
neural_network.update_beta(output_classes) #create output_weights
print(datetime.datetime.now())

error_values = []
for item in h.test_dataset.values:
    predicted = neural_network.predict(item[:len(item)-1])
    print(predicted)
    actual_value = item[len(item)-1]
    print(actual_value)
    error_values.append((actual_value - predicted)**2) #square the error 

print("MSE (Mean Squared Error): ",mean(error_values))

if debug:
    print("checking correctness of shapes:")
    print(neural_network.input_weights[0].shape)
    print(len(neural_network.input_weights))
    print(neural_network.H.shape)
    print(len(neural_network.bias))