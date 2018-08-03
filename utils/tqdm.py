import sys
import numpy as np
import contextlib

from tqdm import tqdm

class TqdmFile(object):
    dummy_file = None
    def __init__(self, dummy_file):
        self.dummy_file = dummy_file

    def write(self, x):
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.dummy_file)

@contextlib.contextmanager
def stdout_to_tqdm():
    save_stdout = sys.stdout
    try:
        sys.stdout = TqdmFile(sys.stdout)
        yield save_stdout
    except Exception as exc:
        raise exc
    finally:
        sys.stdout = save_stdout
