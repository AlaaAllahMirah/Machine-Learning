#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#===============================================univariate==========================================================================================
#reading univariate data

uni_data= pd.read_csv('univariateData.csv')
# adding headers 
# headers= ['feature', 'output']
# uni_data.to_csv("univariateData_2.csv", header=headers, index=False)
# uni_data= pd.read_csv("univariateData_2.csv")
# # visualizing data in scatter plot and notice linearity
# uni_data.plot(kind='scatter',x='feature',y='output')
# plt.show()




#Extracting input and output variables.

feature= uni_data.values[:, 0]  
output= uni_data.values[:, 1] 

#splitting the data

X_train,X_test,y_train,y_test=train_test_split(feature,output,test_size=.2,random_state=1)


LR=0.001     #learning rate
N=len(feature)



def fit(X_train,y_train):                 # method to train the linear regression model
    # intializing the parameters
    a=0
    w=0
    for i in range(1000):
        a,w=Gradient_Descent(a,w)
    return a ,w

def Gradient_Descent(a,w):
      #hypothis function
      predicted_output= a + w* X_train 
      difference= predicted_output-y_train
      #partial derivative of parameters of cost function
      da= (1/N) * sum(difference)  
      dw = (1/N) * sum(X_train * difference) 
      # updating parameters
      a = a - LR * da
      w =w - LR * dw
      return a ,w

def predict(X_train):                   #method to use the trained linear regression model for prediction
     y_predicted = X_train*w+a
     return y_predicted


def Compute_Cost(x,y,a,w):               #compute the value of the objective function eithr using train or test
    #hypothis function
    evaluated_y= x*w + a
    #cost function (mean square error)
    cost=(1/N) * sum((y-evaluated_y)**2)  
    mse= cost
    return mse,evaluated_y

def Evaluate_Performance(x,y,a,w):        #calculate the accuracy of the prediction for the test data.
     mse,evaluated_y=Compute_Cost(x,y,a,w)
     RMSE=np.sqrt(mse)
     return RMSE

      
# testing functions
a,w=fit(X_train,y_train)
train_prediction=predict(X_train)
print("train_prediction",train_prediction)
mse,y_evaluated=Compute_Cost(X_test,y_test,a,w)
print("mse :",mse,"..","y_evaluated :",y_evaluated)
accurcay=Evaluate_Performance(X_test,y_test,a,w)
print("accuracy of RMSE :",accurcay)


