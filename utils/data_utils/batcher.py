import numpy as np
from typing import Iterable, List, Any


def get_batched_data(data, chunk: int) -> np.ndarray:
    return np.array([i for i in _gen_batched_data(data, chunk)])

def _gen_batched_data(data, chunk: int) -> Iterable[List[Any]]:
    for index in range(0, len(data) - chunk + 1):
        yield data[index:index + chunk]
