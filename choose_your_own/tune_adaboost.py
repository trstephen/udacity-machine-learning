#!/usr/bin/python
"""
	Iterate over a bunch of DecisionTree and AdaBoost params
	because I have no intiution about how to tune these
	and might as well just brute force it.
"""

# import sys
# from time import time
# sys.path.append("../tools/")
from prep_terrain_data import makeTerrainData
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# make terrain data
features_train, labels_train, features_test, labels_test = makeTerrainData()

# DT params
min_samples_splits = range(105, 276, 10)

# adaboost params
n_estimators = range (30, 181, 10)
algos = ["SAMME", "SAMME.R"]

scores = []

print("Accuracy\tn_est\tmss\talgo")

for n_estimator in n_estimators:
	for algo in algos:
		for min_samples_split in min_samples_splits:
			dt_clf = DecisionTreeClassifier(min_samples_split=min_samples_split)
			clf = AdaBoostClassifier(dt_clf, n_estimators=n_estimator, algorithm=algo)

			clf.fit(features_train, labels_train)
			pred = clf.predict(features_test)
			accuracy = accuracy_score(pred, labels_test)

			if accuracy >= 0.93:
				print("{acc:0.4%}\t{est}\t{mss}\t{algo}").format(
					acc=accuracy, est=n_estimator, algo=algo, mss=min_samples_split)

			scores.append( (accuracy, n_estimator, algo, min_samples_split) )

print("\nSorting results, showing top 10\n")

scores.sort(key=lambda tup: tup[0], reverse=True)

print("Accuracy\tn_est\tmss\talgo")
for score in scores[:10]:
	print("{acc:0.4%}\t{est}\t{mss}\t{algo}").format(
				acc=score[0], est=score[1], algo=score[2], mss=score[3])
