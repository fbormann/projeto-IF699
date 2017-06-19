#the only intention here is to create dummy variables out of 
import pandas as pd

dataset = pd.read_csv("datasets/servo.csv", header=None)

dataset = pd.get_dummies(dataset)

targets = dataset.iloc[:, 2]
dataset.drop(dataset.columns[2], axis=1)
dataset = pd.concat([dataset, targets], axis=1)



dataset.to_csv("datasets/servo_revised.csv", index=False, header=False)

