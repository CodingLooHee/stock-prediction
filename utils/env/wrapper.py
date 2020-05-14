import numpy as _np

class StockerV0:
    def __init__(self, data):
        if not isinstance(data, _np.ndarray):
            raise TypeError('Data type must be numpy array')
        try:
            assert data.shape[1] == 6
        except:
            raise IndexError('Data must have 6 index following by: Open, High, Low, Close Adj, Close and Volume')

        self._data = data

