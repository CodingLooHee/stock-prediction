import utils
import pandas as pd
import pathlib


PATH_CWD = pathlib.Path.cwd()
PATH_STORAGE = PATH_CWD.joinpath('data')
PATH_DATA_MSFT = PATH_STORAGE.joinpath('MSFT.csv')


data = pd.read_csv(PATH_DATA_MSFT)
data = utils.data.auto.data_auto(data)


env = utils.env.wrapper.StockerV0(data)

