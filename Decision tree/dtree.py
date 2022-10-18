import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split




# cost functions in this algorithm is Gini index and entropy each one has a different formula

def Gini(x):
    unique, counts = np.unique(x, return_counts=True)
    p = counts/x.shape[0]
    gini_coeff= 1-np.sum(p*p)    # p is the probability of having a specific value.
    return(gini_coeff)


def entropy(x):
 
    elements,counts = np.unique(x,return_counts = True)     # returns the number of times each unique item appears in col 
     #print(counts)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))]) 
     #print(entropy)
    return entropy

# root and different nodes are chosen based on the  higher information gain

# information gain will be calculated for all attributes in all features 

def IG(attribute,y):
 
  c1= sum(attribute)
  c2= attribute.shape[0] - c1
  return entropy(y)-c1/(c1+c2)*entropy(y[attribute])-c2/(c1+c2)*entropy(y[-attribute])
  

def categorical_options(x):
  
  x = x.unique()

  combinations = []
  for i in range(0, len(x)+1):
      for subset in itertools.combinations(x, i):
          subset = list(subset)
          combinations.append(subset)

  return combinations[1:-1]

def split(x, y):
 

  split_value = []
  ig = [] 
  options = categorical_options(x)

  for val in options:
    attribute =   x < val  
    val_ig = IG(attribute, y)
    ig.append(val_ig)
    split_value.append(val)
    max_ig = max(ig)
    ig_index = ig.index(max_ig)
    best_split = split_value[ig_index]
    return(max_ig,best_split)




file = open("cardio_train.csv")
data= np.loadtxt(file, delimiter=";",skiprows=1,dtype=int)
#print(data.shape)       # (70000, 13)    
x=data[:,1:-1]
#print(x.shape)          # (70000, 11) without id & target columns
y=data[:,-1]
#print(y.shape)          # (70000,) last column target column
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=1)
#print(X_train.shape)  #(63000, 11)
#print(X_test.shape)   #(7000, 11)
#print(y_train.shape)  #(63000,)
#print(y_test.shape)    #(7000,)



