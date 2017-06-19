import pandas as pd

dataset = pd.read_csv("datasets/computer.csv", header=None)

dataset = dataset.iloc[:, 2:len(dataset)-1]
print(dataset)

dataset.to_csv("datasets/computer_revised.csv", index=False, header=False)

