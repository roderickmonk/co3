import numpy as np
import time
from multiprocessing import Pool

import regr_process_0_0_4_layers4 as regr

num_processes  = 10
num_trials     = 10
start_seed     = 80

seeds = np.arange(num_trials) + start_seed

t0 = time.time()
if __name__ == '__main__':
    with Pool( num_processes ) as p:
        print(p.map( regr.regression_process, seeds))

print( time.time() - t0)
