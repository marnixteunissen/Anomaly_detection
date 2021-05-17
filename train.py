import tensorflow as tf
import numpy as np
import pandas as pd
import models
import data_processing
import os
import itertools
from sacred import Experiment

# train models with varying nr of layers and filters:
# logging done with sacred: https://github.com/sakoarts/sacred_presentation/blob/master/sacred_presentation.ipynb

data_dir = os.getcwd() + r'\data\data-set'
train_set, val_set = data_processing.create_data_sets(data_dir, 'TOP', 'train')
num_classes = len(train_set.class_names)

layers = [5, 10, 15]
filters = [32, 64, 128]
experiments = list(itertools.product(layers, filters))

for l, f in experiments:
    []
