#!/usr/bin/env python

from itertools import count
from typing import Any, List

from project_functions import repeat_last

a = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

yy = repeat_last(a)

for i in range(25):

    if i < 25:
        print(i, next(yy))
    else:
        break
