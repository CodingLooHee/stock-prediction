import utils
import pandas as pd
import pathlib
import matplotlib.pyplot as plt


PATH_CWD = pathlib.Path.cwd()
PATH_STORAGE = PATH_CWD.joinpath('data')
PATH_DATA_MSFT = PATH_STORAGE.joinpath('MSFT.csv')


data = pd.read_csv(PATH_DATA_MSFT)
data = utils.data.auto.data_auto(data)

print('Data shape:', data.shape)
print('Data example:', data[0])

data_x = utils.data.batcher.get_batched_data(data, 32)
data_x = data_x[:-1]

data_y = data[32:, 4]

print('Batched data x shape:', data_x.shape)
print('Batched data y shape:', data_y.shape)

model = utils.model.get_model.model_v1(input_shape=(32, 6))
model.summary()

model.fit(data_x, data_y, epochs=5, verbose=2)

predicted_y = []
for i in data_x:
    predicted_y.append(model.predict(i.reshape([1, 32, 6])))

plt.plot(data_y)
plt.plot(predicted_y)
