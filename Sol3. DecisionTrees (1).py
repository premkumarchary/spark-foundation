#!/usr/bin/env python
# coding: utf-8

# ### Workshop - Decision Trees
# 
# This workshop deals with understanding the working of decision trees.

# In[124]:


# Importing libraries in Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import  accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier


# In[125]:


from sklearn.datasets import load_iris
# Loading the iris dataset
iris=load_iris()

data=pd.DataFrame(iris.data, columns=iris.feature_names)

y =iris.target


# In[126]:


#Explore the data
data.head()


# In[127]:


data.info()


# In[128]:


#shape of target data
y.shape


# In[129]:


data.describe()


# In[130]:


data.columns


# In[131]:


data.isnull().sum()


# In[132]:


data.dtypes


# In[133]:


# Data visulization
sns.distplot(data["sepal length (cm)" ])


# In[134]:


sns.distplot(data["sepal width (cm)" ])


# In[135]:


sns.distplot(data["petal length (cm)" ])


# In[136]:


sns.distplot(data["petal width (cm)" ])


# In[137]:


sns.pairplot(data)


# In[138]:


x= data 
y=y
from sklearn .model_selection import train_test_split  
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)
print("shape of features tarining data :",x_train.shape)
print("shape of features tarining data :",x_train.shape)
print("shape of features test data :",x_test.shape)
print("shape of features test data :",x_test.shape)


# ### define the Decision Tree Algorithm

# In[139]:


# Defining the decision tree algorithm
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(x_train,y_train)

print('Decision Tree Classifer Created')


# # accuracy for training data

# In[147]:


predict=dtree.predict(x_train)
print("Accuracy of training data",accuracy_score(predict,y_train)*100,"%")
print("confusin martix of training data :'\n' ",confusion_matrix(predict,y_train))
sns.heatmap(confusion_matrix(predict,y_train),annot=True,cmap='BuGn')


# In[148]:


predict=dtree.predict(x_test)
print("Accuracy of testing data",accuracy_score(predict,y_test)*100,"%")
print("confusin martix of testing data :'\n' ",confusion_matrix(predict,y_test))
sns.heatmap(confusion_matrix(predict,y_test),annot=True,cmap='BuGn')


# In[150]:


plt.figure(figsize =(19,19))
tree.plot_tree(dtree,filled=True,rounded=True,proportion= True,node_ids=True, feature_names=iris.feature_names)
plt.show()

