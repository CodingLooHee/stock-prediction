import numpy as _np
from typing import Iterable as _Iterable
from typing import List as _List
from typing import Any as _Any


def get_batched_data(data, chunk: int) -> _np.ndarray:
    return _np.array([i for i in _gen_batched_data(data, chunk)])

def _gen_batched_data(data, chunk: int) -> _Iterable[_List[_Any]]:
    for index in range(0, len(data) - chunk + 1):
        yield data[index:index + chunk]
