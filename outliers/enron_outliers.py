#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop("TOTAL") 
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below

### find first outlier: TOTAL
# print(sorted(data, key=lambda x:x[0], reverse=True)[:3])
# for person in data_dict.keys():
# 	if data_dict[person]["salary"] == 26704229:
# 		print person

### find secondary outliers: SKILLING & LAY
for person in data_dict.keys():
	if type(data_dict[person]["bonus"]) is int and \
		data_dict[person]["salary"] >= 1000000 and \
		data_dict[person]["bonus"] >= 5000000 :
		print person, data_dict[person]["bonus"]


for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()