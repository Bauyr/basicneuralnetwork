import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import load_iris
from lightgbm import LGBMClassifier

iris = load_iris()


# In[3]:


list (iris.target_names)


# In[4]:


iris['feature_names']


# In[5]:


X = iris.data
Y = iris.target


# In[6]:


print(Y)


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)


# In[14]:


model = LGBMClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)
print('Accuracy: %.2f%%' % (accuracy * 100.0))
print(report)


# In[18]:


import matplotlib.pyplot as plt
from sklearn import datasets


# In[21]:


fig, axes = plt.subplots(nrows= 2, ncols=2)
colors= ['green', 'blue', 'red']

for i, ax in enumerate(axes.flat):
    for label, color in zip(range(len(iris.target_names)), colors):
        ax.hist(iris.data[iris.target==label, i], label=             
                            iris.target_names[label], color=color)
        ax.set_xlabel(iris.feature_names[i])  
        ax.legend(loc='upper right')


plt.show()







