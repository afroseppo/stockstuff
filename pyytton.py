from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import numpy as np

import sklearn as sk
import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

import urllib.request, json
from pandas.io.json import json_normalize

## general variables

apiKey = "HGW5QA1U49S75JXH" #alphavantage API key
URL = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=MSFT&interval=5min&outputsize=full&apikey=" #alphavantage url


# dl the json file with url and apikey combination
def downloadJson(key, url):
    with urllib.request.urlopen(url + key) as connection:
        data = json.load(connection)
        return data


jsonData = downloadJson(apiKey, URL)

# jsonData into pandas dataframe

dataFrame = pd.DataFrame.from_dict(jsonData['Time Series (5min)'], orient="index")

# split the data into train and test data


train, test = train_test_split(dataFrame, test_size = 0.2)
train, val = train_test_split(dataFrame, test_size = 0.2)

print(len(train))
print(len(test))
print(len(val))


# tf.data from pandas dataframe
