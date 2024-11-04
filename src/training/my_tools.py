from time import time
import os
import datetime

def show_t(t0):
    t = time() - t0
    t_string = str(int(t / 60)) + ' min ' + str(int(t % 60)) + ' sec'
    print(t_string)

def get_t(t0):
    t = time() - t0
    t_string = str(int(t / 60)) + ' min ' + str(int(t % 60)) + ' sec'
    return t_string

def timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d %H.%M.%S')

def make_dir(x):
    """
    creates a folder with path x if it does not already exist and returns x
    """
    if not os.path.exists(x):
        os.makedirs(x)
    return x
