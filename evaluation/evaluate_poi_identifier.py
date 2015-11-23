#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn import cross_validation, tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
	features, labels, test_size=0.3, random_state=42)

### it's all yours from here forward!
# Lesson 13:
#	Overfit score:		0.989473684211
# 	30% split score:	0.724137931034
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

print("Accuracy (30% split): {acc}").format(acc = accuracy_score(pred, labels_test) )

# Lesson 14
# Exploit the fact that labels are: Non-POI=0, POI=1
print("Number of POIs identified in test set: {0}").format(int(sum(pred))) # 4

print("Total people in test set: {0}").format(len(features_test)) # 29

# pull the precision and recall scores from here
print(classification_report(labels_test, pred)) # 0, 0

### notes
# true pos: 6
# true neg: 9
# false pos: 3
# false neg: 2
# precision: 0.67 (true pos / (true pos + false pos))
# recall: 0.75 (true pos / (true pos + false neg))
