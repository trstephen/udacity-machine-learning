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
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###

#########################################################

# Use smaller training set
shrink_factor = 1
print("Using {percent:0.2%} of training data\n").format(percent = 1.0 / shrink_factor)
features_train = features_train[:len(features_train)/shrink_factor] 
labels_train = labels_train[:len(labels_train)/shrink_factor] 

def SVCTimer(kernel, C):
	print("Testing {0} with C = {1}").format(kernel, C)

	clf = SVC(kernel=kernel, C=C)

	t0 = time()
	clf.fit(features_train, labels_train)
	training_time = round(time() - t0, 3)
	print("Training time: {time}s").format(time = training_time)

	t0 = time()
	predictions = clf.predict(features_test)
	pred_time = round(time() - t0, 3)
	print("Prediction time: {time}s").format(time = pred_time)

	accuracy = accuracy_score(predictions, labels_test)
	print("Accuracy: {acc}\n").format(acc=accuracy)

	# predictions is a numpy.ndarray so .count() doesn't work
	# Sara = 0; Chris = 1
	emails = (predictions == 1).sum()
	print("Chris emails: {emails}").format(emails = emails)


C_vals = [10000]

for C in C_vals:
	SVCTimer(kernel="rbf", C=C)
