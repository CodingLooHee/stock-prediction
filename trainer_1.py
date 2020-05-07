import utils
import numpy as np
import pandas as pd
import pathlib

PATH_CWD = pathlib.Path.cwd()
PATH_STORAGE = PATH_CWD.joinpath('data')

PATH_DATA_MSFT = PATH_STORAGE.joinpath('MSFT.csv')


raw = pd.read_csv(PATH_DATA_MSFT)
raw = utils.datautils.auto.data_auto(raw)

print('Data shape:', raw.shape)
print('Data example:', raw[0])
