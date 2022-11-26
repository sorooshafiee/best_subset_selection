import glob
import os
from os.path import join
import numpy as np

DIR_DATA = join(".", "data")
DATASETS = glob.glob(DIR_DATA + "/*.csv")
DATASETS.sort()

with open("./bash_file.sh", "w") as bash_file:
    for dataset in DATASETS:
        print("python subset_selection.py --method \"AIC\" --bigM 2 --dataset \"{}\"".format(dataset),
              file=bash_file)
        print("python subset_selection.py --method \"BIC\" --bigM 2 --dataset \"{}\"".format(dataset),
              file=bash_file)
        print("python subset_selection.py --method \"MSE\" --bigM 2 --dataset \"{}\"".format(dataset),
              file=bash_file)