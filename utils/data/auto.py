import numpy as _np
import pandas as _pd

def data_auto(data: _pd.DataFrame) -> _np.ndarray:
    data.pop('Date')
    data_max = data.max(axis=0)
    data_min = data.min(axis=0)

    data = (data - data_min) / (data_max - data_min)
    
    data = _np.array(data)

    return data
