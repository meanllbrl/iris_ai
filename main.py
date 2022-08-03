


import numpy as np
import random
import pandas as pd
from pandas import plotting
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

#accuracy and value accuracy lists for plot
acc = []
val_acc=[]

def prnt(content):
    print("\n")
    print(content)
    print("\n")

#visualize with lienar regression
def visualizedLinearRegression(dataset):
   sns.lmplot(x="SepalWidthCm", y="SepalLengthCm", hue="Species", data=dataset)
   
#visualize with andrew line method    
def visualizeAndrewLine(dataset):
    plt.subplots(figsize = (10,8))
    plotting.andrews_curves(dataset.drop("Id", axis=1), "Species")

#this method returns the dataset
def getDataSet():
    #read the dataset using pandas
    df = pd.read_csv('../project-2/dataset/Iris.csv')
    #shuffle it 
    df= df.sample(frac=1).reset_index(drop=True)
    #print dataset information
    prnt(df.info())
    return df


#main fucntion of the neural network which will train the network for the specified number of epochs     
def NeuralNetwork(X_train, Y_train, X_val=None, Y_val=None, epochs=10, nodes=[], lr=0.15):
    hidden_layers = len(nodes) - 1
    #the weights of the network will get randomly initialized
    weights = InitializeWeights(nodes)
    #in each epoch, the weights will be updated
    for epoch in range(1, epochs+1):
        weights = Train(X_train, Y_train, lr, weights)
        if(epoch % 5 == 0):
            #every 20 epochs accuracy both for the training and validation sets will be printe
            print("------------------------")
            print("Epoch {}".format(epoch))
            training_accuracy=Accuracy(X_train, Y_train, weights)
            validation_accuracy=Accuracy(X_val, Y_val, weights)
            acc.append(training_accuracy)
            print("Training Accuracy:{}".format(training_accuracy))
            if X_val.any():
                print("Validation Accuracy:{}".format(validation_accuracy)) 
                val_acc.append(validation_accuracy)
            print("------------------------")
    return weights

def InitializeWeights(nodes):
    """Initialize weights with random values in [-1, 1] (including bias)"""
    layers, weights = len(nodes), []
    
    for i in range(1, layers):
        w = [[np.random.uniform(-1, 1) for k in range(nodes[i-1] + 1)]
              for j in range(nodes[i])]
        weights.append(np.matrix(w))
    
    return weights

def ForwardPropagation(x, weights, layers):
    activations, layer_input = [x], x
    for j in range(layers):
        activation = Sigmoid(np.dot(layer_input, weights[j].T))
        activations.append(activation)
        layer_input = np.append(1, activation) # Augment with bias
    
    return activations

def BackPropagation(y, activations, weights, layers ,lr=0.15):
    outputFinal = activations[-1]
    error = np.matrix(y - outputFinal) # Error at output
    
    for j in range(layers, 0, -1):
        currActivation = activations[j]
        
        if(j > 1):
            # Augment previous activation
            prevActivation = np.append(1, activations[j-1])
        else:
            # First hidden layer, prevActivation is input (without bias)
            prevActivation = activations[0]
        
        delta = np.multiply(error, SigmoidDerivative(currActivation))
        weights[j-1] += lr * np.multiply(delta.T, prevActivation)

        w = np.delete(weights[j-1], [0], axis=1) # Remove bias from weights
        error = np.dot(delta, w) # Calculate error for current layer
    
    return weights

def Train(X, Y, lr, weights):
    layers = len(weights)
    for i in range(len(X)):
        x, y = X[i], Y[i]
        x = np.matrix(np.append(1, x)) # Augment feature vector
        
        activations = ForwardPropagation(x, weights, layers)
        weights = BackPropagation(y, activations, weights, layers)

    return weights

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def SigmoidDerivative(x):
    return np.multiply(x, 1-x)

def Predict(item, weights):
    layers = len(weights)
    item = np.append(1, item) # Augment feature vector
    
    ##_Forward Propagation_##
    activations = ForwardPropagation(item, weights, layers)
    
    outputFinal = activations[-1].A1
    index = FindMaxActivation(outputFinal)

    # Initialize prediction vector to zeros
    y = [0 for i in range(len(outputFinal))]
    y[index] = 1  # Set guessed class to 1

    return y # Return prediction vector

def FindMaxActivation(output):
    """Find max activation in output"""
    m, index = output[0], 0
    for i in range(1, len(output)):
        if(output[i] > m):
            m, index = output[i], i
    
    return index

def Accuracy(X, Y, weights):
    """Run set through network, find overall accuracy"""
    correct = 0

    for i in range(len(X)):
        x, y = X[i], list(Y[i])
        guess = Predict(x, weights)

        if(y == guess):
            # Guessed correctly
            correct += 1

    return correct / len(X)


def accuracyPlot():
    plt.clf()
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title("Accuracy")
    plt.legend(['train', 'test'])
    plt.show()

def main():
    #getting the dataset
    df = getDataSet()
    #visualization of the dataset with andrew line visualization
    visualizeAndrewLine(df)
    #visualization of the dataset with linear regression visualization 
    visualizedLinearRegression(df)
    #need to grab the data (the information on each sample) from the pandas array and put it into a nice numpy one.
    X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    X = np.array(X)
    #encoding method
    one_hot_encoder = OneHotEncoder(sparse=False)
    Y = df.Species
    #need to convert classes from categorical ('Setosa', 'Versicolor', 'Virginica') to numerical (0, 1, 2) and then to one-hot encoded ([1, 0, 0], [0, 1, 0], [0, 0, 1]).
    Y = one_hot_encoder.fit_transform(np.array(Y).reshape(-1, 1))
    #split our dataset into train/validation/test using sklearn
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1)

    f = len(X[0]) # Number of features
    o = len(Y[0]) # Number of outputs / classes
    
    layers = [f, 5, 10, o] # Number of nodes in layers
    lr, epochs = 0.15, 160
    
    weights = NeuralNetwork(X_train, Y_train, X_val, Y_val, epochs=epochs, nodes=layers, lr=lr);
    accuracyPlot()
    prnt("Testing Accuracy: {}".format(Accuracy(X_test, Y_test, weights)))


#refference the main method
main()
