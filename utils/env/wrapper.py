import numpy as _np
import random as _random

class StockerV0:
    def __init__(self, data, startup_range=[0, 100]):
        # Test for data
        if not isinstance(data, _np.ndarray):
            raise TypeError('Data type must be numpy array')
        try:
            assert data.shape[1] == 6
        except:
            raise IndexError('Data must have 6 index following by: Open, High, Low, Close Adj, Close and Volume')

        # Test for startup_range
        if len(startup_range) != 2:
            raise IndexError('startup_range must be 2 len list or tuple')
        if not isinstance(startup_range[0], int) or not isinstance(startup_range[1], int):
            raise TypeError('startup_range must be integer')
        if startup_range[0] < 0 or startup_range[1] < 0:
            raise ValueError('startup_range must be positive range')
        if startup_range[0] > startup_range[1]:
            raise ValueError('startup_range must be [min, max] format')

        # If all parameter is in correct form. It's good to go.
        self.data = data
        self._startup_range = startup_range
        self._ptr = _random.randrange(*self._startup_range)
    
    def reset(self):
        self._ptr = _random.randrange(*self._startup_range)
        return self.data_at_ptr
    
    @property
    def data_at_ptr(self):
        return self.data[self._ptr]
