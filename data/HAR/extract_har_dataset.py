import pandas as pd
import numpy as np
from collections import OrderedDict
import scipy.io

DATADIR      = 'UCI HAR Dataset'
DATASET_NAME = 'UCI_HAR_DATASET.mat'

SIGNALS = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z"
]


def load_data():
    """
    Obtain the dataset from multiple files.
    Returns: X_train, X_test, y_train, y_test
    """
    X_train, X_test = load_signals('train'), load_signals('test')
    y_train, y_test = load_y('train'), load_y('test')

    return X_train, X_test, y_train, y_test


def load_signals(subset):
    signals_data = []

    for signal in SIGNALS:
        #filename = f'{DATADIR}/{subset}/Inertial Signals/{signal}_{subset}.txt'
        filename = '{}/{}/Inertial Signals/{}_{}.txt'.format(DATADIR,subset,signal,subset)
        signals_data.append(
            _read_csv(filename).as_matrix()
        )

    # Transpose is used to change the dimensionality of the output,
    # aggregating the signals by combination of sample/timestep.
    # Resultant shape is (7352 train/2947 test samples, 128 timesteps, 9 signals)
    return np.transpose(signals_data, (1, 2, 0))

def _read_csv(filename):
    return pd.read_csv(filename, delim_whitespace=True, header=None)

def load_y(subset):
    """
    The objective that we are trying to predict is a integer, from 1 to 6,
    that represents a human activity. We return a binary representation of
    every sample objective as a 6 bits vector using One Hot Encoding
    (https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html)
    """
    #filename = f'{DATADIR}/{subset}/y_{subset}.txt'
    filename = '{}/{}/y_{}.txt'.format(DATADIR,subset,subset)

    y = _read_csv(filename)[0]

    return pd.get_dummies(y).as_matrix()

def create_mat(_x_train, _x_test, _y_train, _y_test):
    dict_           = []
    dict_           = OrderedDict()
    dict_['X_train'] = _x_train
    dict_['Y_train'] = _y_train
    dict_['X_test'] = _x_test
    dict_['Y_test'] = _y_test
    scipy.io.savemat(DATASET_NAME, dict_)





if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test = load_data()
    create_mat(_x_train=X_train, _x_test=X_test, _y_train=Y_train, _y_test=Y_test)
