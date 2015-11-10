#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    ### your code goes here
    errors = abs(predictions - net_worths)
    cleaned_data = zip(ages.tolist(), net_worths.tolist(), errors.tolist())

    cleaned_data = sorted(cleaned_data, key=lambda x: x[2][0]) # sorted smallest -> lagest

    return cleaned_data[:len(cleaned_data) - (len(cleaned_data) / 10)]

