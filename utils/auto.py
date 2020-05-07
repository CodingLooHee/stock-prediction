import numpy as np

def data_auto(data):
    data.pop('Date')
    data_max = data.max(axis=0)
    data_min = data.min(axis=0)

    data = (data - data_min) / (data_max - data_min)
    
    data = np.array(data)

    return data
