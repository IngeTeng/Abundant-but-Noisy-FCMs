import numpy as np
from functools import reduce


def gen_random_number(start,end,number):
    nums = [np.random.uniform(start, end) for x in range(0, number)]
    sum = reduce(lambda x, y: x + y, nums)
    norm = [x / sum for x in nums]
    return norm

