""" Pump maintanence demo results """
#!/usr/bin/env python
# coding: utf-8

from pickle import load

from tensorflow import keras
from tensorflow.keras import models

import utils

path = "data/ex"
df = utils.prepare_raw_data(path, date1="2021-02-01", date2="2021-06-01")

scaler = load(open("models/pump/scaler.pkl", "rb"))


def prepare_X(data):
    """ Transform features """
    data = scaler.transform(data)
    data = data.reshape((data.shape[0], 1, data.shape[1]))
    return data


X = df[utils.FEATURE_LIST].fillna(0)
Y = df["targets"].values
X = prepare_X(X)

model = models.load_model("models/pump/model.h5")

checkpoint_path = "models/pump/cp.ckpt"
# model.load_weights(checkpoint_path)

model.compile(
    loss="mean_squared_error", optimizer=keras.optimizers.Adam())

yhat2 = model.predict(X)
yhat2 = yhat2 / yhat2.max()
utils.plot_signal_hat(yhat2, Y)
yhat2[yhat2 <= 0.6] = 0
yhat2[yhat2 > 0.6] = 1

utils.plot_signal_hat(yhat2, Y)

df["pred_temp"] = yhat2
utils.show_metrics(df["pred_temp"], df["targets"])


df_abi = utils.prepare_raw_data(path, date1="2021-11-26", date2="2021-11-29")
abi_X = df_abi[utils.FEATURE_LIST].fillna(0)
abi_Y = df_abi["targets"].values
abi_X = prepare_X(abi_X)

yhat2 = model.predict(abi_X)
yhat2 = yhat2 / yhat2.max()
yhat2[yhat2 <= 0.6] = 0  # thresholds
yhat2[yhat2 > 0.6] = 1
df_abi["pred_temp"] = yhat2
utils.plot_signal_hat(
    df_abi["pred_temp"][200:
                        1800], df_abi["Raw Average Velocity (m/s)"][200:1800]
)
