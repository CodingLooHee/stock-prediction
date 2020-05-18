import numpy as _np
import random as _random

class StockerV0:
    def __init__(self, data, startup_range=[0, 100], startup_money=100, max_hold=200, min_money=20):
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
        if startup_range[1] > len(data):
            raise IndexError('startup_range is higher than data index')

        # If all parameter is in correct form. It's good to go.
        self.data = data
        self._money = startup_money
        self._STARTUP_MONEY = startup_money
        self._STARTUP_RANGE = startup_range
        self._MAX_HOLD = max_hold
        self._MIN_MONEY = min_money
        self._LAST_DATA_INDEX = len(data) - 1
        self._ptr = _random.randrange(*self._STARTUP_RANGE)
        self._stock_quantity = 0
    
    def reset(self):
        self._ptr = _random.randrange(*self._STARTUP_RANGE)
        self._stock_quantity = 0
        self._money = self._STARTUP_MONEY
        return self._get_env()[0]
    
    def step(self, action):
        if action == 0:
            self._hold_stock()
        if action == 1:
            self._buy_stock()
        if action == 2:
            self._sell_stock()
        return self._get_env()
    
    def _buy_stock(self):
        if self._money >= self._price_at_ptr and self._MAX_HOLD > self._stock_quantity:
            self._stock_quantity += 1
            self._money -= self._price_at_ptr
        self._ptr += 1

    def _sell_stock(self):
        if self._stock_quantity > 0:
            self._stock_quantity -= 1
            self._money += self._price_at_ptr
        self._ptr += 1

    def _hold_stock(self):
        self._ptr += 1
    
    def _get_env(self):
        return _np.concatenate([self._data_at_ptr, [self._money / self._STARTUP_MONEY,
                                                    self._stock_quantity / self._MAX_HOLD]]),\
               self._money - self._STARTUP_MONEY,\
               self._is_end()

    def _is_end(self):
        if self._ptr >= self._LAST_DATA_INDEX:
            return True
        if self._money < self._MIN_MONEY:
            return True
        return False

    @property
    def _data_at_ptr(self):
        return self.data[self._ptr]

    @property
    def _price_at_ptr(self):
        return self._data_at_ptr[4]
