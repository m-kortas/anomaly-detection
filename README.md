
# Time Series Anomaly Detection Models

Machine Learning models to detect anomalies on time series data.

The models are deep learning classification models (deep neural networks) built on Bidirectional LSTM, LSTM and Fully Convolutional Neural Network.
       
## Models

For the training, the data were merged into 5-hours-long chunks (5 minutes x (T = 60)).
If a chunk had an anomaly then whole chunk was labeled as anomaly. 
This way we reduce amount of data (5 times) and balance the dataset 
(since classed are very unbalanced - small amount of anomalies).

We can ajust all parameters (including T time and STEP). 

Features used: velocity, hour, minute, day, month, flow, level
1. Bidirectional LSTM v1 

First model - LSTMv1
Long short-term memory (LSTM) is an artificial recurrent neural network (RNN)
architecture. Unlike standard feedforward neural networks, LSTM has 
feedback connections. It can process not only single data points 
(such as images), but also entire sequences of data.

A common LSTM unit is composed of a cell, an input gate, an output gate 
and a forget gate. The cell remembers values over arbitrary time intervals 
and the three gates regulate the flow of information into and out of the cell.

LSTM networks are well-suited to classifying, processing and making 
predictions based on time series data, since there can be lags of 
unknown duration between important events in a time series. LSTMs were 
developed to deal with the vanishing gradient problem that can be encountered 
when training traditional RNNs. Relative insensitivity to gap length is an 
advantage of LSTM over RNNs, hidden Markov models and other sequence learning 
methods in numerous applications.

A Bidirectional LSTM, or biLSTM, is a sequence processing model that 
consists of two LSTMs: one taking the input in a forward direction, 
and the other in a backwards direction. BiLSTMs effectively increase 
the amount of information available to the network, improving the context 
available to the algorithm (e.g. knowing what words immediately follow and
precede a word in a sentence).

2. LSTM v2 


## Lessons Learned

- The data is very unbalanced (very few anomalies - True Positives)
- Supervised models work generally better than unsupervised (checked Isolation Tree, KMeans)
- Issues with tensorflow instalation on Kubernetes (TO DO)
- LSTM perform much better than Fully Convolutional Neural Networks on this specific task

## Roadmap

- Article on various ML methods for anomaly detection
https://neptune.ai/blog/anomaly-detection-in-time-series


- Trying LSTM autoencoders 
https://towardsdatascience.com/lstm-autoencoder-for-extreme-rare-event-classification-in-keras-ce209a224cfb


- Image-based anomaly detection

Transforming channel signals into spectrograms and feeding it as an image to Convolutional Neural Network.
We could train model on all sites and channels at once (more data) and create spectrograms for each day/hour (depending on business requirements.)

We could also plot geolocation on a country's topography map - to retrieve geospacial information and correlations between sites as well as climate information. 

Inspiration: 

https://towardsdatascience.com/sound-based-bird-classification-965d0ecacb2b

https://www.ntt-review.jp/archive/ntttechnical.php?contents=ntr201708fa5.html

## Tech Stack

- keras
- pyspark
- scikit-learn
- tensorflow

## Installation

Install tensorflow with conda (Python 3.7)

```bash
    conda create -n tf2 python=3.7 ipython ipykernel
    conda activate tf2
    conda install -c anaconda tensorflow
    python -m ipykernel install --user --name=tf2
```

Install all the packages with pip 

```bash
    pip install -r requirements.txt
```
## Usage/Examples

```python

import utils
import LSTMv1

df_train, df_val, df_test = utils.read_split_data(PATH)   #split data for training and prepare dataset
raw_data = df_test.copy()[utils.T :].reset_index(drop=True)  
X_train, y_train, X_test, y_test, X_val, y_val = utils.prepare_dataset(
     df_test, df_train, df_val
)

history, model = LSTMv1.create_train_model(X_train, y_train, X_val, y_val)  #train model 

utils.evaluate_training(history)    #evaluate training
utils.show_results(model, X_test, raw_data)

```

```python

import pandas as pd
import utils
from tensorflow.keras import models

df = utils.prepare_raw_data(PATH)  #prepare dataset (raw velocity data)
raw_data = df.copy()[utils.T :].reset_index(drop=True)  
X_test = utils.prepare_dataset(df, training=False)

model = models.load_model(MODEL_PATH, compile=False)  #load model 
model.compile(loss="binary_crossentropy", metrics=utils.METRICS, optimizer="adam")

result = utils.show_results(model, X_test, raw_data) #show results
raw_data["pred"] = result

utils.show_random_results(raw_data, "velocity", SITE, compare=False)

```

## Feedback

If you have any feedback, please reach out to me :) 

## Contributing

Contributions are always welcome!


## Authors

- [@m-kortas](https://www.github.com/m-kortas) Magdalena Kortas

