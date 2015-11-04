#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print("Number of persons: {0}").format( len(enron_data) )
print("Number of features: {0}").format( len(enron_data["SKILLING JEFFREY K"].values()) )
print("Number of POIs: {0}").format( 
	sum(1 for name in enron_data.values() if name["poi"] == True) )

print("James Prentice's total stock value: ${0:,d}").format(
	enron_data["PRENTICE JAMES"]["total_stock_value"] )
print("Wesley Colwell to POIs: {0}").format(
	enron_data["COLWELL WESLEY"]["from_this_person_to_poi"] )
print("Skilling exercised options: ${0:,d}").format(
	enron_data["SKILLING JEFFREY K"]["exercised_stock_options"] )

for poi in ["SKILLING JEFFREY K", "LAY KENNETH L", "FASTOW ANDREW S"]:
	print("{poi} total payments: ${total:,d}").format(
		poi=poi.title(), total=enron_data[poi]["total_payments"])

print("Number of people with salaries: {0}").format(
	sum(1 for name in enron_data.values() if name["salary"] != "NaN") )
print("Number of people with email addresses: {0}").format(
	sum(1 for name in enron_data.values() if name["email_address"] != "NaN") )

countMissingTotalPayments = sum( 
	1 for name in enron_data.values() 
	if name["total_payments"] == "NaN")
print("Percentage without salary: {0:0.2%}").format(
	float(countMissingTotalPayments) / len(enron_data) )

countPOIsMissingTotalPayments = sum( 
	1 for name in enron_data.values() 
	if name["total_payments"] == "NaN" 
	and  name["poi"] == True )
print("Percentage of POIs without salary: {0:0.2%}").format(
	float(countPOIsMissingTotalPayments) / len(enron_data) )