#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
# features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###

#########################################################


print("Preprocessing...")
features_train, features_test, labels_train, labels_test = preprocess()

# Use smaller training set
shrink_factor = 100
features_train = features_train[:len(features_train)/shrink_factor] 
labels_train = labels_train[:len(labels_train)/shrink_factor] 

print("\nTesting kernels with {percent:0.2%} of training data\n").format(percent = 1.0 / shrink_factor)
print("Kernel\t\tAccuracy\tTrain Time\tPred Time\n")

kernels = ["linear", "poly", "rbf", "sigmoid"]

for kernel in kernels:
	clf = SVC( kernel=kernel )

	t0 = time()
	clf.fit(features_train, labels_train)
	training_time = round(time() - t0, 3)

	t0 = time()
	predictions = clf.predict(features_test)
	pred_time = round(time() - t0, 3)

	accuracy = accuracy_score(predictions, labels_test)
	
	print("{kernel}\t\t{acc:0.4%}\t{train:.3f} s\t\t{pred:.3f} s").format(
		kernel=kernel, acc=accuracy, train=training_time, pred=pred_time)
