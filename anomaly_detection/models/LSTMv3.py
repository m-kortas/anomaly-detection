""" LSTM Model Architecture """
# !/usr/bin/env python
# coding: utf-8
from time import time

import numpy as np
import utils
from tensorflow import random
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

EPOCHS = 20
RANDOM_SEED = 42
CLASS_WEIGHT = {0: 1, 1: 2}

np.random.seed(RANDOM_SEED)
random.set_seed(RANDOM_SEED)

checkpoint_path = "models/vel-only_v3/cp.ckpt"
model_path = "models/vel-only_v3/model"


def create_train_model(X_train, y_train, X_val, y_val):
    """ Create and train model """
    callbacks, weights, val_weights, class_weight, metrics = utils.set_model_weights(
        y_train, y_val, CLASS_WEIGHT, checkpoint_path
    )

    model = Sequential()
    model.add(LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1, activation='sigmoid'))

    model.load_weights(checkpoint_path)

    model.compile(loss="binary_crossentropy",
                  metrics=metrics, optimizer="adam")

    start = time()
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=round(X_train.shape[0] / 20),
        validation_data=(X_val, y_val, val_weights),
        sample_weight=weights,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    model.save(model_path)

    print("-" * 10)
    print(f"Training was completed in {time() - start:.2f} secs")
    print(model.summary())
    return history, model

# Prediction binary_accuracy (mapped) = 93.96089561652475%
# Prediction F-score (mapped) = 0.702485758674262
