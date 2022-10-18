#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#======================================================Multivariate=================================================================================
#reading Multivariate data

multi_data= pd.read_csv('multivariateData.csv')
#print(multi_data)
# adding headers 
# headers= ['feature_1','feature_2','output']
# multi_data.to_csv("multivariateData_2.csv", header=headers, index=False)
# multi_data= pd.read_csv("multivariateData_2.csv")
#Extracting input and output variables.

features=multi_data.values[:, 0:2] 
features = (features-np.mean(features))/np.std(features) 
output= multi_data.values[:, 2] 

#splitting the data

X_train,X_test,y_train,y_test=train_test_split(features,output,test_size=.2,random_state=1)

feature_1_train= X_train[:, 0] 
feature_2_train= X_train[:, 1] 
feature_1_test= X_test[:, 0] 
feature_2_test= X_test[:, 1] 

LR=0.001     #learning rate
N=len(feature_1_train)


def fit_multi(x1,x2):                 # method to train the linear regression model
    # intializing the parameters
    a=0
    w1=0
    w2=0
    for i in range(1000):
        a,w1,w2=Gradient_Descent_multi(x1,x2,a,w1,w2)
    return a ,w1,w2

def Gradient_Descent_multi(x1,x2,a,w1,w2):
      #hypothis function
      predicted_output= a + w1*x1+w2*x2
      difference= predicted_output-y_train
      #partial derivative of parameters of cost function
      da= (1/N) * sum(difference)  
      dw1 = (1/N) * sum(feature_1_train * difference)
      dw2 = (1/N) * sum(feature_2_train * difference) 
      # updating parameters
      a = a - LR * da
      w1 =w1 - LR * dw1
      w2 =w2 - LR * dw2
      return a ,w1,w2

def predict_multi(x1,x2):                   #method to use the trained linear regression model for prediction
     y_predicted = x1*w1+x2*w2+a
     return y_predicted


def Compute_Cost_multi(x1,x2,y,a,w1,w2):               #compute the value of the objective function eithr using train or test
    #hypothis function
    evaluated_y= x1*w1+x2*w2+a
    #cost function (mean square error)
    cost=(1/N) * sum((y-evaluated_y)**2)  
    mse= cost
    return mse, evaluated_y

def Evaluate_Performance_multi(x1,x2,y,a,w1,w2):        #calculate the accuracy of the prediction for the test data.
     mse,evaluated_y=Compute_Cost_multi(x1,x2,y,a,w1,w2)
     RMSE=np.sqrt(mse)
     return RMSE

 # testing functions
      
print("testing functions for multivariate...")
a,w1,w2=fit_multi(feature_1_train,feature_2_train)
train_prediction=predict_multi(feature_1_train,feature_2_train)
print("train_prediction",train_prediction)
mse,y_evaluated=Compute_Cost_multi(feature_1_test,feature_2_test,y_test,a,w1,w2)
print("mse :",mse,"..","y_evaluated :",y_evaluated)
accuracy=Evaluate_Performance_multi(feature_1_test,feature_2_test,y_test,a,w1,w2)
print("accuracy of RMSE :",accuracy)
