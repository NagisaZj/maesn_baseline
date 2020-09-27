import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
import csv
import pickle
import os
#import colour
import torch

def load_pkl(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

data = load_pkl('/home/zj/Desktop/maesn_baseline/itr_0.pkl')
print(data)