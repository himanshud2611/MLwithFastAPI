# training a simple random forest classifier on the dataset
import pandas as pd
import pickle

from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X,y = fetch_openml('mnist_784', version=1, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


clf = RandomForestClassifier(n_jobs=-1) #It tells the classifier to use all available CPU cores on your machine, "use as many parallel processes as possible".

#clf.fit() is used to train the model on your training data. It typically takes two arguments: the feature matrix X and the target vector y.
clf.fit(X_train, y_train)

#clf.score() is used to calculate the accuracy of the model on a given dataset. It also takes two arguments: X and y, and returns a single float value (the accuracy score).
print(clf.score(X_test, y_test))

#save a trained machine learning model to a file using the pickle library 
with open('mnist_model.pkl', 'wb') as f:
    pickle.dump(clf, f) #serialize the classifier object (clf) and save it to the file.