""" Conv Model Architecture """

# !/usr/bin/env python
# coding: utf-8

from time import time

import numpy as np
import utils
from tensorflow import keras, random

EPOCHS = 20
RANDOM_SEED = 42
CLASS_WEIGHT = {0: 1, 1: 3}

np.random.seed(RANDOM_SEED)
random.set_seed(RANDOM_SEED)

checkpoint_path = "models/vel-only_conv2/cp.ckpt"
model_path = "models/vel-only_conv2/model"


def make_model(input_shape):
    """ Create architecture """
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(
        filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(
        filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(
        filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(1, activation="sigmoid")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


def create_train_model(X_train, y_train, X_val, y_val):
    """ Create and train model """
    callbacks, weights, val_weights, class_weight, metrics = utils.set_model_weights(
        y_train, y_val, CLASS_WEIGHT, checkpoint_path
    )

    model = make_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    # model.load_weights(checkpoint_path)

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
        class_weight=class_weight
    )

    model.save(model_path)

    print("-" * 10)
    print(f"Training was completed in {time() - start:.2f} secs")
    print(model.summary())
    return history, model

# Prediction binary_accuracy (mapped) = 86.06906338694418%
# Prediction F-score (mapped) = 0.5219586978086391
