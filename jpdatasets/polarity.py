import os, sys
import numpy as np
import csv

def load_data(seed=0):
    d = os.path.dirname(sys.modules['jpdatasets'].__file__)
    file_path = os.path.join(d, 'data/polarity.csv')

    with open(file_path) as f:
        r = csv.reader(f, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
        next(r)  # skip header
        data = [row for row in r]

    np.random.seed(seed=seed)
    np.random.shuffle(data)
    
    train, test = np.split(data, [int(len(data) * 0.7)])

    x_train = train[:, 0]
    y_train = train[:, 1].astype(int)
    x_test = test[:, 0]
    y_test = test[:, 1].astype(int)

    return (x_train, y_train), (x_test, y_test)

