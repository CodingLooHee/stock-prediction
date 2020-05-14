import utils
import pandas as pd
import pathlib


PATH_CWD = pathlib.Path.cwd()
PATH_STORAGE = PATH_CWD.joinpath('data')
PATH_DATA_AAPL = PATH_STORAGE.joinpath('AAPL.csv')
print('Data path:', PATH_DATA_AAPL)


data = pd.read_csv(PATH_DATA_AAPL)
data = utils.data.auto.data_auto(data)
print('Data shape:', data.shape)


env = utils.env.wrapper.StockerV0(data)
print(env.reset())
