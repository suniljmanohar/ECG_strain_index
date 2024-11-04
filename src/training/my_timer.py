from time import time
import datetime


class Timer:
    def __init__(self, start_now):
        if start_now:
            self.start()

    def start(self):
        self.t0 = time()

    def reset(self):
        self.t0 = 0

    def elapsed(self, units):
        # units = 's' for seconds as a string, 'm' for minutes as a string
        # 's_float' for seconds as a float, 'm_float) for minutes as a float
        t = time() - self.t0
        if units == 'm':
            output = str(int(t / 60)) + ' min ' + str(int(t % 60)) + ' sec'
        elif units == 's':
            output = '{} sec'.format(t)
        elif units == 'm_float':
            output = t / 60
        elif units == 's_float':
            output = t
        return output


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
