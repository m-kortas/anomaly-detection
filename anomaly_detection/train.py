"""Training file for neural nets"""

import numpy as np
import tensorflow as tf
from tensorflow import random

import utils
# !/usr/bin/env python
# coding: utf-8
from models import Conv

tf.config.list_physical_devices()

RANDOM_SEED = 42
PATH = "data/processed/processed_NE016_10_data.csv"

np.random.seed(RANDOM_SEED)
random.set_seed(RANDOM_SEED)

df_train, df_val, df_test = utils.read_split_data(PATH)

raw_data = df_test.copy()[utils.T:].reset_index(drop=True)

X_train, y_train, X_test, y_test, X_val, y_val = utils.prepare_dataset(
    df_test, df_train, df_val
)

history, model = Conv.create_train_model(X_train, y_train, X_val, y_val)

# utils.evaluate_training(history)
utils.show_results(model, X_test, raw_data)
