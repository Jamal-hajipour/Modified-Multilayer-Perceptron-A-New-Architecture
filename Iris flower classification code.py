"""
Modified Multilayer Perceptron: A New Architecture
This code is based on VÍTOR GAMA LEMOS code "Multilayer Perceptron from scratch"
https://www.kaggle.com/code/vitorgamalemos/multilayer-perceptron-from-scratch?scriptVersionId=79241565
which we modify it for our new architecture. You can also visit link below for
more information about Lemos code:
https://medium.com/computing-science/using-multilayer-perceptron-in-classification-problems-iris-flower-6fc9fbf36040    
"""
import random
import seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

seaborn.set(style='whitegrid'); seaborn.set_context('talk')

from sklearn.datasets import load_iris
iris_data = load_iris()

n_samples, n_features = iris_data.data.shape


random.seed(123)

x = 0 
ativation = {(lambda x: 1/(1 + np.exp(-x)))}
deriv = {(lambda x: x*(1-x))}

activation_tang = {(lambda x: np.tanh(x))}
deriv_tang = {(lambda x: 1-x**2)}

activation_ReLU = {(lambda x: x*(x > 0))}
deriv_ReLU = {(lambda x: 1 * (x>0))}

def separate_data():
    A = iris_dataset[0:40]
    tA = iris_dataset[40:50]
    B = iris_dataset[50:90]
    tB = iris_dataset[90:100]
    C = iris_dataset[100:140]
    tC = iris_dataset[140:150]
    train = np.concatenate((A,B,C))
    test =  np.concatenate((tA,tB,tC))
    return train,test
train_porcent = 80 # Porcent Training 
test_porcent = 20 # Porcent Test
iris_dataset = np.column_stack((iris_data.data,iris_data.target.T)) #Join X and Y
iris_dataset = list(iris_dataset)
random.shuffle(iris_dataset)

Filetrain, Filetest = separate_data()

train_X = np.array([i[:4] for i in Filetrain])
train_y = np.array([i[4] for i in Filetrain])
test_X = np.array([i[:4] for i in Filetest])
test_y = np.array([i[4] for i in Filetest])

class MultiLayerPerceptron(BaseEstimator, ClassifierMixin): 
    def __init__(self, params=None):     
        if (params == None):
            self.inputLayer = 4                        # Input Layer
            self.outputLayer = 3                       # Outpuy Layer
            self.learningRate = 0.005                  # Learning rate
            self.max_epochs = 600                      # Epochs
            self.activation = self.ativacao['sigmoid'] # Activation function
            self.deriv = self.derivada['sigmoid']
        else:
            self.inputLayer = params['InputLayer']
            self.OutputLayer = params['OutputLayer']
            self.learningRate = params['LearningRate']
            self.max_epochs = params['Epocas']
            self.activation = self.ativacao[params['ActivationFunction']]
            self.deriv = self.derivada[params['ActivationFunction']]
        'Starting Bias and Weights and l'
        self.WEIGHT_output = self.starting_weights(self.OutputLayer, self.inputLayer)
        self.ALPHA_output = self.starting_alphas(self.OutputLayer, 3)
        self.classes_number = 3 
        
    pass
    def starting_weights(self, x, y):
        return np.array([[2  * random.random() - 1 for i in range(x)] for j in range(y)])

    def starting_alphas(self, x, y):
        return np.array([[2  * random.random() - 1 for i in range(x)] for j in range(y)])
    
    
    ativacao = {
         'sigmoid': (lambda x: 1/(1 + np.exp(-x))),
            'tanh': (lambda x: np.tanh(x)),
            'Relu': (lambda x: x*(x > 0)),
               }
    derivada = {
         'sigmoid': (lambda x: x*(1-x)),
            'tanh': (lambda x: 1-x**2),
            'Relu': (lambda x: 1 * (x>0))
               }
    def Backpropagation_Algorithm(self, x):
       DELTA_output = []
       'Stage 1 - Error: OutputLayer'
       ERROR_output = self.output - self.OUTPUT
       DELTA_output = ((-1)*(ERROR_output) * self.deriv(self.OUTPUT))
        
       'Stage 2 - Update weights OutputLayer'
       L_output = np.matmul(x,self.WEIGHT_output)
       A_output = np.multiply(self.ALPHA_output[2 , :] , L_output) + self.ALPHA_output[1 , :]
       for j in range(self.OutputLayer):
            for i in range(self.inputLayer):
                self.WEIGHT_output[i][j] -= (self.learningRate * (DELTA_output[j] * x[i]) * A_output[j])
            self.ALPHA_output[2][j] -= (self.learningRate * DELTA_output[j] * (1/2) * L_output[j] * L_output[j])
            self.ALPHA_output[1][j] -= (self.learningRate * DELTA_output[j] * L_output[j])
            self.ALPHA_output[0][j] -= (self.learningRate * DELTA_output[j])
                
    def show_err_graphic(self,v_erro,v_epoca):
        plt.figure(figsize=(9,4))
        plt.plot(v_epoca, v_erro, "m-",color="b", marker=11)
        plt.xlabel("Number of Epochs")
        plt.ylabel("Squared error (MSE) ");
        plt.title("Error Minimization")
        plt.show()
 
    def predict(self, X, y):
        'Returns the predictions for every element of X'
        print("The Weight Matrix is: ",self.WEIGHT_output)
        print('The Alpha Matrix is: ',self.ALPHA_output)
        my_predictions = []
        my_realprediction = []
        'Forward Propagation'
        for idx,inputs in enumerate(X): 
            L_output = np.matmul(inputs,self.WEIGHT_output)
            self.OUTPUT = np.zeros(self.classes_number)
            'Stage 1 - (Forward Propagation)'
            for i in range (self.OutputLayer):
                self.OUTPUT[i] = self.activation(self.ALPHA_output[2][i] *(L_output[i])**2 + self.ALPHA_output[1][i] * (L_output[i]) + self.ALPHA_output[0][i])
            'Stage 2 - One-Hot-Encoding'
            if(np.argmax(self.OUTPUT) == 0): 
                my_predictions.append(0)
                my_realprediction.append(self.OUTPUT)
            elif(np.argmax(self.OUTPUT) == 1):
                my_predictions.append(1)
                my_realprediction.append(self.OUTPUT)
            elif(np.argmax(self.OUTPUT) == 2):
                my_predictions.append(2)
                my_realprediction.append(self.OUTPUT)
            
        array_score = []
        for i in range(len(my_predictions)):
            if my_predictions[i] == 0: 
                array_score.append([i, 'Iris-setosa', my_predictions[i], y[i],my_realprediction[i]])
            elif my_predictions[i] == 1:
                 array_score.append([i, 'Iris-versicolour', my_predictions[i], y[i],my_realprediction[i]])
            elif my_predictions[i] == 2:
                 array_score.append([i, 'Iris-virginica', my_predictions[i], y[i],my_realprediction[i]])
                    
        dataframe = pd.DataFrame(array_score, columns=['_id', 'class', 'output', 'hoped_output' , 'Real Output'])
        return my_predictions, dataframe

    def fit(self, X, y):  
        count_epoch = 1
        total_error = 0
        n = len(X); 
        epoch_array = []
        error_array = []
        W1 = []
        A1 = []
        while(count_epoch <= self.max_epochs):
            for idx,inputs in enumerate(X): 
                self.output = np.zeros(self.classes_number)
                self.OUTPUT = np.zeros(self.classes_number)
                'Stage 1 - (Forward Propagation)'
                for i in range (self.OutputLayer):
                    self.OUTPUT[i] = self.activation(self.ALPHA_output[2][i] *(np.dot(inputs, self.WEIGHT_output[: ,i])**2 + self.ALPHA_output[1][i] * (np.dot(inputs, self.WEIGHT_output[: ,i])) + self.ALPHA_output[0][i]))
                'Stage 2 - One-Hot-Encoding'
                if(y[idx] == 0): 
                    self.output = np.array([1,0,0]) #Class1 {1,0,0}
                elif(y[idx] == 1):
                    self.output = np.array([0,1,0]) #Class2 {0,1,0}
                elif(y[idx] == 2):
                    self.output = np.array([0,0,1]) #Class3 {0,0,1}
                
                square_error = 0
                for i in range(self.OutputLayer):
                    erro = 1/2 * (self.output[i] - self.OUTPUT[i])**2
                    square_error = (square_error + ( 0.05 * erro))
                    total_error = total_error + square_error
         
                'Backpropagation : Update Weights'
                self.Backpropagation_Algorithm(inputs)
                
            total_error = (total_error / n)
            if((count_epoch % 50 == 0)or(count_epoch == 1)):
                print("Epoch ", count_epoch, "- Total Error: ",total_error)
                error_array.append(total_error)
                epoch_array.append(count_epoch)
                
            W1.append(self.WEIGHT_output)
            A1.append(self.ALPHA_output)
             
                
            count_epoch += 1
        self.show_err_graphic(error_array,epoch_array)
        
        
        plt.plot(W1[0])
        plt.title('Weight Output update during training')
        plt.legend(['neuron1', 'neuron2', 'neuron3'])
        plt.ylabel('Value Weight')
        plt.show()
        
        plt.plot(A1[0])
        plt.title('Alpha Output update during training')
        plt.legend(['neuron1', 'neuron2', 'neuron3'])
        plt.ylabel('Value Alpha')
        plt.show()

        return self


dictionary = {'InputLayer':4, 'OutputLayer':3,
              'Epocas':700, 'LearningRate':0.005, 'ActivationFunction':'sigmoid'}

Perceptron = MultiLayerPerceptron(dictionary)
Perceptron.fit(train_X,train_y)

pred, dataframe = Perceptron.predict(test_X, test_y)
hits = n_set = n_vers = n_virg = 0
score_set = score_vers = score_virg = 0
for j in range(len(test_y)):
    if(test_y[j] == 0): n_set += 1
    elif(test_y[j] == 1): n_vers += 1
    elif(test_y[j] == 2): n_virg += 1
        
for i in range(len(test_y)):
    if test_y[i] == pred[i]: 
        hits += 1
    if test_y[i] == pred[i] and test_y[i] == 0:
        score_set += 1
    elif test_y[i] == pred[i] and test_y[i] == 1:
        score_vers += 1
    elif test_y[i] == pred[i] and test_y[i] == 2:
        score_virg += 1    
         
hits = (hits / len(test_y)) * 100
faults = 100 - hits
print(dataframe)

graph_hits = []
print("Porcents :","%.2f"%(hits),"% hits","and","%.2f"%(faults),"% faults")
print("Total samples of test",n_samples)
print("*Iris-Setosa:",n_set,"samples")
print("*Iris-Versicolour:",n_vers,"samples")
print("*Iris-Virginica:",n_virg,"samples")

graph_hits.append(hits)
graph_hits.append(faults)
labels = 'Hits', 'Faults';
sizes = [96.5, 3.3]
explode = (0, 0.14)

fig1, ax1 = plt.subplots();
ax1.pie(graph_hits, explode=explode,colors=['green','red'],labels=labels, autopct='%1.1f%%',
shadow=True, startangle=90)
ax1.axis('equal')
plt.show()

acc_set = (score_set/n_set)*100
acc_vers = (score_vers/n_vers)*100
acc_virg = (score_virg/n_virg)*100
print("- Acurracy Iris-Setosa:","%.2f"%acc_set, "%")
print("- Acurracy Iris-Versicolour:","%.2f"%acc_vers, "%")
print("- Acurracy Iris-Virginica:","%.2f"%acc_virg, "%")
names = ["Setosa","Versicolour","Virginica"]
x1 = [2.0,4.0,6.0]
fig, ax = plt.subplots()
r1 = plt.bar(x1[0], acc_set,color='orange',label='Iris-Setosa')
r2 = plt.bar(x1[1], acc_vers,color='green',label='Iris-Versicolour')
r3 = plt.bar(x1[2], acc_virg,color='purple',label='Iris-Virginica')
plt.ylabel('Scores %')
plt.xticks(x1, names);plt.title('Scores by iris flowers - Multilayer Perceptron')
plt.show()