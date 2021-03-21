import sklearn as sk
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA, TruncatedSVD
from scipy.stats import zscore
import numpy as np
import csv

csv_file = open('C:/Users/julia/Downloads/7 semestre/IA/Projeto/MLP/diamonds_test.csv') #need to change the path of the file csv
test = csv.reader(csv_file, delimiter=';')

csv_file = open('C:/Users/julia/Downloads/7 semestre/IA/Projeto/MLP/diamonds_training.csv') #need to change the path of the file csv
training = csv.reader(csv_file, delimiter=';')

diamonds_training = list(training) #Creating a list with the data and calling it as training
diamonds_test = list(test) #Creating a list with the data and calling it as test

#Taking out the last column (the target)
atributes_training = [x[:-1] for x in diamonds_training][1:]    #Taking all the columns, except the last one (price), without the header
atributes_test = [x[:-1] for x in diamonds_test][1:]    #Taking all the columns, except the last one (price), without the header

#Converting from string to float (the atributes)
for x in range (0, len(atributes_training)): #Transform all the strings of the list in float 
    for y in range (0, len(atributes_training[x])):
        v = atributes_training[x][y]
        atributes_training[x][y]=float(v)

for x in range (0, len(atributes_test)): #Transform all the strings of the list in float 
    for y in range (0, len(atributes_test[x])):
        v = atributes_test[x][y]
        atributes_test[x][y]=float(v)

#Taking the last column (the target)
targets_training = [x[-1] for x in diamonds_training][1:] #Taking just the price column, without the header
targets_test = [x[-1] for x in diamonds_test][1:] #Taking just the price column, without the header

#Converting from string to float (the target)
for x in range (0, len(targets_training)): #Transform all the strings of the list in float 
     v = targets_training[x]
     targets_training[x]=float(v)

for x in range (0, len(targets_test)): #Transform all the strings of the list in float 
     v = targets_test[x]
     targets_test[x]=float(v)

#Normalizing the atributes
atributes_training_normalized = zscore(atributes_training, axis = 1)
atributes_test_normalized = zscore(atributes_test, axis = 1)

###### NOT IMPORTANT####### #How I 
#Creating training datasets
#training_datasets_atributes = dataset_atributes_normalized[:int (0.5*len(dataset_atributes_normalized))]
#training_datasets_targets = dataset_targets[:int (0.5*len(dataset_atributes_normalized))]
#####################################

###### NOT IMPORTANT####### 
#Creating a test dataset
#test_datasets_atributes = dataset_atributes_normalized[int (0.5*len(dataset_atributes_normalized)):]
#test_datasets_targets = dataset_targets[int (0.5*len(dataset_targets)):]
###########################


rede = MLPClassifier(hidden_layer_sizes = 5 , solver='sgd', batch_size = 100, max_iter=1000, tol=1e-4, activation='tanh')
#rede.fit(atributes_training_normalized, targets_training)
rede.fit(atributes_training, targets_training)
print(rede.score(atributes_test, targets_test))
#print(rede.score(atributes_test_normalized, targets_test))