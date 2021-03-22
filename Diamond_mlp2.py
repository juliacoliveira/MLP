import sklearn as sk
from sklearn.neural_network import MLPClassifier
from scipy.stats import zscore
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

##############################Training part############################
# Opening the file
csv_file = 'C:/Users/julia/Downloads/7 semestre/IA/Projeto/MLP/diamonds_training.csv' #need to change the path of the file csv
df = pd.read_csv(csv_file, delimiter=';')
print (df.head(3))

indexNames = df[ df['price'] == 4].index
df = df.drop(indexNames)
indexNames = df[ df['price'] == 3].index
df = df.drop(indexNames)

# Getting the columns names
linear_vars = df.select_dtypes(include = [np.number]).columns
linear_vars =  list(linear_vars)

# Verifying and transforming the lines with zero, EXCEPT the 'color' column
colorless_vars = linear_vars.copy()
colorless_vars.remove('color')
colorless_vars.remove('price')
print (linear_vars)
print (colorless_vars)

# Normalizing the atributes, EXCEPT the 'color'
def removeoutliers(df, colorless_vars, z):
    for var in colorless_vars:
        df1 = df[np.abs(zscore(df[var])) < z]
    return df1

df = removeoutliers(df, colorless_vars, 2)

# converting to log scale
def convertToLog(df, colorless_vars):
    for var in colorless_vars:
        df[var] = np.log(df[var])

convertToLog(df, colorless_vars)
#print (df.head(3))

# Divide the normalized data into targets and attributes
attr_df = df.drop(['price'], axis = 1).values.tolist()

targets_df = np.ravel(df.drop(['carat', 'color', 'table', 'x', 'y', 'z'], axis = 1).values.tolist())

###################################Test part##########################################

# Opening the file
csv_file = 'C:/Users/julia/Downloads/7 semestre/IA/Projeto/MLP/diamonds_test.csv' #need to change the path of the file csv
df = pd.read_csv(csv_file, delimiter=';')
df_visual = df.copy()
print (df.head(3))

indexNames = df[ df['price'] == 4].index
df = df.drop(indexNames)
indexNames = df[ df['price'] == 3].index
df = df.drop(indexNames)

# Getting the columns names
linear_vars = df.select_dtypes(include = [np.number]).columns
linear_vars =  list(linear_vars)

df = removeoutliers(df, colorless_vars, 2)
convertToLog(df, colorless_vars)

attr_test = df.drop(['price'], axis = 1).values.tolist()

targets_test = np.ravel(df.drop(['carat', 'color', 'table', 'x', 'y', 'z'], axis = 1).values.tolist())
print (targets_test)

########################################################################################

rede = MLPClassifier(hidden_layer_sizes = 15, solver='sgd', batch_size = 100, max_iter=1000, tol=1e-4, activation='tanh')
rede.fit(attr_df, targets_df)
print(rede.score(attr_test, targets_test))

predictions = rede.predict(attr_test)
print("Matriz de confusão =\n{}\n".format(confusion_matrix(targets_test, predictions)))