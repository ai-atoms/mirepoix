import os
import numpy as np

ptargets_folder = 'data/datasets/d567b/ptargets'

for file in os.listdir(ptargets_folder):
    a = np.load(f'{ptargets_folder}/{file}')
    print (a)
    # break