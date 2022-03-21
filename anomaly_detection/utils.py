"""Utils for anomaly detection and time series analysis"""

#!/usr/bin/env python
# coding: utf-8

import random
from datetime import timedelta
from os import walk
from pickle import dump, load

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_sample_weight
from tensorflow import keras, random
from tensorflow.keras import backend as K

START_DATE = "2019-01-01"
END_DATE = "2021-06-01"

SITES = [
    "NE001CALa_17_data",
    "NE003CAL_17_data",
    "NE004CAL_17_data",
    "NE006a_10_data",
    "NE009_10_data",
    "NE010_10_data",
    "NE011_10_data",
    "NE012_11_data",
    "NE012_20_data",
    "NE016_10_data",
    "NE017_10_data",
]

FEATURE_LIST = [
    "Daily Cumulative Rainfall (mm)",
    "Final Rainfall (mm)",
    "MP1 FLOW1 (l/s)",
    "MP1 PDEPTH_1 (mm)",
    "MP1 UNIDEPTH (mm)",
    "MP1 UpDEPTH_1 (mm)",
    "MP1 WATERTEMP_1 (°C)",
    "Raw Average Velocity (m/s)",
    "hour",
    "day",
    "month",
]

COLS = FEATURE_LIST + ["date", "targets"]

TARGET_COL = "targets"

SCALER_PATH = "models/vel-only_v1/scaler.pkl"
T = 50
STEP = 1

rcParams["figure.figsize"] = 22, 10
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.set_seed(RANDOM_SEED)


# ----- Data Engineering and feature extraction --------------------------------


def label(row, col1, col2):
    """ Label anomaly  """
    if pd.isnull(row[col2]):
        return 1
    if pd.isnull(row[col1]):
        return 1
    elif row[col1] == 0:
        return 1
    elif round(row[col1]) != round(row[col2]):
        return 1
    else:
        return 0


def label_anomaly(row):
    """ Create target column """
    if row["vel_flag"] == 1:
        return 1
    elif row["level_flag"] == 1:
        return 1
    elif row["flow_flag"] == 1:
        return 1
    elif row["Data Quality Flag (none)"] == 1:
        return 1
    elif row["Data Quality Flag (none)"] == 2:
        return 1
    else:
        return 0


def add_anomalies(row):
    """ Add more untagged anomalies """
    if row["targets_temp"] == 1:
        return 1
    elif (row["targets+1"] == 1) & (row["targets-1"] == 1):
        return 1
    else:
        return 0


def clean_dataset(df, targets=True):
    """Clean raw data and append targets if exist """
    if targets is True:
        df["vel_flag"] = df.apply(
            lambda row: label(
                row, "Raw Average Velocity (m/s)", "Final Raw Average Velocity (m/s)"
            ),
            axis=1,
        )
        df["level_flag"] = df.apply(
            lambda row: label(row, "MP1 UpDEPTH_1 (mm)", "Final Raw Level (mm)"), axis=1
        )
        df["flow_flag"] = df.apply(
            lambda row: label(row, "MP1 FLOW1 (l/s)", "Final Raw Flow (l/s)"), axis=1
        )
        df["targets_temp"] = df.apply(lambda row: label_anomaly(row), axis=1)

        df["targets+1"] = df.targets_temp.shift(-1).rolling(2).max()
        df["targets-1"] = df.targets_temp.rolling(2).max()
        df["targets"] = df.apply(lambda row: add_anomalies(row), axis=1)

    df['yyyy/MM/dd HH:mm:ss'] = pd.to_datetime(df['yyyy/MM/dd HH:mm:ss'])
    df["hour"] = df['yyyy/MM/dd HH:mm:ss'].dt.hour
    df["day"] = df['yyyy/MM/dd HH:mm:ss'].dt.day
    df["month"] = df['yyyy/MM/dd HH:mm:ss'].dt.month
    df = df.sort_values(by=["site", 'yyyy/MM/dd HH:mm:ss'])
    df = df.reset_index(drop=True)
    final = df.drop("site", axis=1)
    final["date"] = final["yyyy/MM/dd HH:mm:ss"]
    final = final[COLS]
    return final


def prepare_raw_data(path, date1=None, date2=None):
    """ Prepare data from raw, downloaded dataset"""
    filenames = next(walk(path), (None, None, []))[2]
    filenames = [x[:-4] for x in filenames]
    sites = pd.DataFrame()
    for file in filenames:
        print("loaded: ", file)
        df = pd.read_csv(path + "/" + file + ".csv",
                         low_memory=False, header=2)
        df["site"] = file
        sites = sites.append(df)
    if date1 is not None:
        sites = sites[sites["yyyy/MM/dd HH:mm:ss"].between(date1, date2)]
    final = clean_dataset(sites).reset_index(drop=True)
    return final


def append_history(X, y=None, t=T, step=STEP, targets=True):
    """ Append last T observations and reduce dataset with steps """
    if targets is None:
        Xs = []
        for i in range(t, len(X), step):
            # try:
            # if ((X[SITES].iloc[i] == X[SITES].iloc[(i - T)]).all() == True):
            v = X.iloc[(i - t): i].values
            Xs.append(v)  # checking if sites are matching
        # else:
        #   print("Different Site ", i)
        #  empty = np.full([T, X.shape[1]], 0, dtype=float)
        # Xs.append(empty)
        X = np.array(Xs)
        return X

    else:
        Xs, ys = [], []
        for i in range(T, len(X), step):
            # try:
            # if ((X[SITES].iloc[i] == X[SITES].iloc[(i - T)]).all() == True):
            v = X.iloc[(i - T): i].values
            # labels = y.iloc[i : i + T]
            labels = y.iloc[i]  # stats.tmax(labels)
            Xs.append(v)  # checking if sites are matching
            ys.append(labels)
        # else:
        #   print("Different Site ", i)
        #  empty = np.full([T, X.shape[1]], 0, dtype=float)
        # Xs.append(empty)
        # ys.append(1)

        X = np.array(Xs)
        y = np.array(ys).reshape(-1, 1)
        return X, y


# ----- Model training --------------------------------


def read_split_data(path):
    """ Read and split dataset into train, val, test """
    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"])
    start_date = pd.to_datetime(START_DATE)
    end_date = pd.to_datetime(END_DATE)

    train_split = 0.8
    test_split = 0.1

    diff = (end_date - start_date).days
    train_end = start_date + pd.DateOffset(days=diff * train_split)
    val_end = start_date + pd.DateOffset(days=diff * (1 - test_split))

    df_train = df[df["date"].between(
        start_date, train_end)].reset_index(drop=True)
    df_val = df[df["date"].between(train_end, val_end)].reset_index(drop=True)
    df_test = df[df["date"].between(val_end, end_date)].reset_index(drop=True)
    print(len(df_train))
    print(len(df_val))
    print(len(df_test))

    return df_train, df_val, df_test


def prepare_dataset(test, train=None, val=None, training=True):
    """ Scale features and prepare dataset """
    if training is not True:

        # for col in FEATURE_LIST:
        #   if col not in test:
        #      test[col] = 0

        scaler = load(open(SCALER_PATH, "rb"))
        test.loc[:, FEATURE_LIST] = scaler.transform(
            test[FEATURE_LIST].to_numpy())
        X_test = append_history(X=test[FEATURE_LIST], targets=None)
        print(f"Test data dimensions: {X_test.shape}")
        return X_test

    else:
        dataframes = [train, val, test]

        # for df in dataframes:
        #   for col in FEATURE_LIST:
        #      if col not in df:
        #         df[col] = 0

        scaler = RobustScaler()
        scaler = scaler.fit(train[FEATURE_LIST].to_numpy())
        dump(scaler, open(SCALER_PATH, "wb"))
        # scaler = load(open(SCALER_PATH, "rb"))

        for df in dataframes:
            df.loc[:, FEATURE_LIST] = scaler.transform(
                df[FEATURE_LIST].to_numpy())

        X_train, y_train = append_history(
            train[FEATURE_LIST], train[TARGET_COL])
        X_test, y_test = append_history(test[FEATURE_LIST], test[TARGET_COL])
        X_val, y_val = append_history(val[FEATURE_LIST], val[TARGET_COL])

        print(f"Test data dimensions: {X_test.shape}, {y_test.shape}")
        print(f"Train data dimensions: {X_train.shape}, {y_train.shape}")
        print(f"Val data dimensions: {X_val.shape}, {y_val.shape}")

        return X_train, y_train, X_test, y_test, X_val, y_val


def custom_f1(y_true, y_pred):  # taken from old keras source code
    """ F1 Score to be used in metrics """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


METRICS = [
    custom_f1,
    keras.metrics.BinaryAccuracy(name="binary_accuracy", threshold=0.5),
    keras.metrics.AUC(name="auc"),
]


def set_model_weights(y_train, y_val, class_weight, checkpoint_path):
    """ Compute model weights """
    weights = compute_sample_weight(class_weight, y=y_train)
    val_weights = compute_sample_weight(class_weight, y=y_val)

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
        keras.callbacks.ModelCheckpoint(
            monitor="val_loss",
            filepath=checkpoint_path,
            save_weights_only=True,
            verbose=1,
        ),
    ]

    return callbacks, weights, val_weights, class_weight, METRICS


def evaluate_training(history):
    """ Evaluate training """
    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].plot(history.history["loss"], color="b", label="Training loss")
    axs[0].plot(history.history["val_loss"],
                color="r", label="Validation loss")
    axs[0].set_title("Loss curves")
    axs[0].legend(loc="best", shadow=True)

    axs[1].plot(history.history["custom_f1"],
                color="b", label="Training custom_f1")
    axs[1].plot(
        history.history["val_custom_f1"], color="r", label="Validation custom_f1",
    )
    axs[1].set_title("custom_f1 curves")
    axs[1].legend(loc="best", shadow=True)
    plt.show()


# ----- Results visualisation --------------------------------


def plot_cm(y_true, y_pred, class_names):
    """ Plot confusion matrix """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    ax = sns.heatmap(
        cm, annot=True, fmt="d", cmap=sns.diverging_palette(220, 20, n=7), ax=ax
    )

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    b, t = plt.ylim()
    b += 0.5
    t -= 0.5
    plt.ylim(b, t)
    plt.show()


def show_metrics(y_pred, y_test):
    """ Default metric calculation """
    score = sum(y_pred == y_test) / len(y_test)
    print(f"Prediction binary_accuracy (mapped) = {float(score)*100}%")
    F1 = f1_score(y_test, y_pred)
    print(f"Prediction F-score (mapped) = {F1}")
    plot_cm(y_test, y_pred, ["normal", "anomaly"])
    print("\n Classification Report : \n",
          classification_report(y_test, y_pred))
    return score, F1


def add_anomalies_pred(row):
    """ Add more untagged anomalies """
    if row["pred_temp"] == 1:
        return 1
    elif (row["pred+1"] == 1) & (row["pred-1"] == 1):
        return 1
    else:
        return 0


def plot_signal_hat(Y_test, Y_hat):
    """ Plot signals """
    fig = plt.figure()
    plt.plot(Y_hat)
    plt.plot(Y_test)
    plt.legend(["target", "prediction"])
    plt.title("Pediction on test data")
    plt.show()


def show_results(model, X_test, raw_data):
    """ Show prediction results (specific for window model) """
    # if step == None:
    #   step = STEP
    y_pred = model.predict(X_test)

    # try:
    #   fpr, tpr, thre = roc_curve(raw_data["targets"], y_pred)
    #    plt.plot(fpr, tpr)
    # except:
    #   print("ROC Error")

    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1

    # pred = pd.DataFrame(y_pred)
    # pred = pd.DataFrame({col: np.repeat(pred[col], step) for col in pred.columns})
    # raw_data = raw_data.reset_index(drop=True)[0 : len(pred)]
    # raw_data["pred"] = pred.reset_index(drop=True)
    # raw_data.pred = raw_data.pred.astype(int)
    raw_data["pred_temp"] = y_pred.astype(int)
    raw_data["pred+1"] = raw_data.pred_temp.shift(-1).rolling(2).max()
    raw_data = raw_data.fillna(0)
    raw_data["pred-1"] = raw_data.pred_temp.rolling(2).max()
    raw_data = raw_data.fillna(0)
    raw_data["pred"] = raw_data.apply(
        lambda row: add_anomalies_pred(row), axis=1)

    # try:
    show_metrics(raw_data["pred"], raw_data[TARGET_COL])
    # except:
    # print("no test labels")

    return raw_data


def show_random_results(df, col, site, compare=None):
    """ Show sample dates when anomalies happened """
    df = df[df[site] == 1]
    samples = df[df["targets"] == 1][["date"]].sample(n=3)
    samples["date"] = pd.to_datetime(samples["date"]).apply(lambda x: x.date())
    samples["date1"] = samples["date"] + timedelta(days=14)
    samples = samples.drop_duplicates(subset=["date"])

    for x in samples.index:
        date = str(samples["date"][x])
        date1 = str(samples["date1"][x])
        print("LSTMv2.1")
        plot_sample(df, col, TARGET_COL, date, date1, pred_col="pred")
        if compare is True:
            print("LSTMv2.2")
            plot_sample(df, col, TARGET_COL, date, date1, pred_col="pred2")


def plot_sample(df, feature, baseline, date1, date2, pred_col=None):
    """ Plot sample comparison of prediction and ground truth """
    if pred_col is None:
        pred_col = "pred"

    after_start_date = df["date"] >= date1
    before_end_date = df["date"] <= date2
    between_two_dates = after_start_date & before_end_date
    data = df[["date", feature, pred_col, baseline]].loc[between_two_dates]

    fig, ax = plt.subplots(figsize=(20, 7))
    a = data.loc[data[pred_col] == 1, ["date", feature]]
    b = data.loc[data[baseline] == 1, ["date", feature]]
    c = data.loc[(data[baseline] == 1) & (
        data[pred_col] == 1), ["date", feature]]

    ax.scatter(
        pd.to_datetime(data["date"]), data[feature], color="black", label="Normal", s=20
    )
    ax.scatter(
        pd.to_datetime(a["date"]),
        a[feature],
        color="blue",
        label="False Positive",
        s=30,
        marker="o",
    )
    ax.scatter(
        pd.to_datetime(b["date"]),
        b[feature],
        color="red",
        label="False Negative",
        s=30,
        marker="o",
    )
    ax.scatter(
        pd.to_datetime(c["date"]),
        c[feature],
        color="green",
        label="Detected anomaly",
        s=30,
        marker="o",
    )

    ax.xaxis_date()
    plt.legend()
    fig.autofmt_xdate()
    plt.show()
