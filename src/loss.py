import numpy as np

def gini(y):
    # compute 0 and 1 proportions
    proportions = np.unique(y, return_counts=True)[1] / len(y)
    return 1 - np.sum(proportions**2)

def entropy(y):
    proportions = np.unique(y, return_counts=True)[1] / len(y)

    return -np.sum(proportions*np.log2(proportions))

# def split_score(parent_x, left_x, right_x):
#     return