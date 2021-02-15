import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from pandas.plotting import scatter_matrix
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
from plotly.graph_objs import *
from sklearn import preprocessing


dataset = pd.read_csv(r"C:\Users\aste\Desktop\konstantinos_ml\codes\parkinsondataset.csv") #read the dataset

dataset.info() # get info of the dataset
dataset.describe()

plt.figure(figsize=(20,15))
sns.heatmap(dataset.corr(), annot=True)
plt.show()
correlations = dataset.corr()# visualize heatmap of the correlations
scatter_matrix(dataset, alpha = 0.2, figsize=(60,60), diagonal = 'kde') # Visualize all correlations with plots
print(sns.distplot(dataset))# Visualize all distributions with plot

#check distrinutions of some important features (age and rpde)
dataset.info()
hist_data = [dataset['age']]
group_labels = ['age'] 
fig = ff.create_distplot(hist_data, group_labels)
#plot(fig)
hist_data = [dataset['RPDE']]
group_labels = ['RPDE'] 
fig = ff.create_distplot(hist_data, group_labels)
#plot(fig)


#Check the balance
dependent_analysis = dataset['result'].value_counts(normalize=True) * 100
print(dependent_analysis)

class_count_01, class_count_02 = dataset['result'].value_counts()
dataset['result'].value_counts().plot(kind='bar', title='count (target)')

print('class 0:', class_count_01)
print('class 1:', class_count_02)



y = dataset['result'].values
X = dataset.drop(['result'], axis='columns').values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)  #split data
sc = StandardScaler() #scale data with mean value and standard deviation
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#SVM
classifier = SVC(kernel = 'rbf', probability=True)
classifier.fit(X_train, y_train) #train model
y_pred = classifier.predict(X_test) #predict
cm1 = confusion_matrix(y_test, y_pred)
print(cm1)
print(accuracy_score(y_test, y_pred))
report = classification_report(y_test, y_pred) #evaluate
print(report)
print("Training set score: {:.3f}".format(classifier.score(X_train,y_train)))
print("Test set score: {:.3f}".format(classifier.score(X_test,y_test)))


#Decision Tree
classifier2 = DecisionTreeClassifier(criterion = 'entropy')
classifier2.fit(X_train, y_train) #train model
y_pred = classifier2.predict(X_test)#predict
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
report = classification_report(y_test, y_pred)#evaluate
print(report)
print("Training set score: {:.3f}".format(classifier2.score(X_train,y_train)))
print("Test set score: {:.3f}".format(classifier2.score(X_test,y_test)))
#Decision tree has overfitting 

#kNN
classifier4 = KNeighborsClassifier(n_neighbors = 2, metric = 'minkowski', p = 2)
classifier4.fit(X_train, y_train) #train model
y_pred = classifier4.predict(X_test)#predict
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
report = classification_report(y_test, y_pred)#evaluate
print(report)
print("Training set score: {:.3f}".format(classifier4.score(X_train,y_train)))
print("Test set score: {:.3f}".format(classifier4.score(X_test,y_test)))

#kNN has better f1 score, so we choose him




















