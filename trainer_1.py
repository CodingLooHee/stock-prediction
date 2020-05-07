import utils
import numpy as np
import pandas as pd
import pathlib
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

PATH_CWD = pathlib.Path.cwd()
PATH_STORAGE = PATH_CWD.joinpath('data')

PATH_DATA_MSFT = PATH_STORAGE.joinpath('MSFT.csv')


data = pd.read_csv(PATH_DATA_MSFT)
data = utils.data.auto.data_auto(data)

print('Data shape:', data.shape)
print('Data example:', data[0])

data_x = utils.data.batcher.get_batched_data(data, 32)
data_y = utils.data.batcher.get_batched_data(data, 1)

data_x = data_x[:-1]

data_y = data_y.reshape([-1, 6])
data_y = data_y[32:, 4]

print('Batched data x shape:', data_x.shape)
print('Batched data y shape:', data_y.shape)

model = utils.model.get_model.model_v1(input_shape=(32, 6))
model.summary()
