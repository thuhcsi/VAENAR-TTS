import os
import sys


class Logger(object):
    def __init__(self, log_dir, log_f='train.log'):
        self.terminal = sys.stdout
        log_fn = os.path.join(log_dir, log_f)
        self.log = open(log_fn, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass
