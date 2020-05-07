import yfinance as yf
import pathlib

PATH_ROOT = pathlib.Path.cwd()
PATH_STORAGE = PATH_ROOT.joinpath('data')

if not pathlib.Path.exists(PATH_STORAGE):
    pathlib.Path.mkdir(PATH_STORAGE)

stock_name = input('Stock: ').upper()
period = input('Period: ')

PATH_FILE = PATH_STORAGE.joinpath(f'{stock_name.upper()}.csv')

if pathlib.Path.exists(PATH_FILE):
    print('\nDo you want to overwrite?')
    if (flag := input('(y/n)? ')) not in ['y', 'Y']:
        exit(0)

data = yf.download(stock_name, period=period)
data.to_csv(PATH_FILE)
