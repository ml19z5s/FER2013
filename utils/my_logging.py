import logging
import os
import sys
import time


def get_logger(file_path, dataset, model, args=None):
    begin_time = time.strftime("%m%d_%H%M", time.localtime())
    log_file = f'{model}_{begin_time}.log'
    if not os.path.exists(file_path):
        os.mkdir(file_path)
        print(f'Created directory: {file_path}')
    if not os.path.exists(os.path.join(file_path, dataset)):
        os.mkdir(os.path.join(file_path, dataset))
    filename = os.path.join(file_path, dataset, log_file)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter('%(message)s'))
    sh.setLevel(logging.INFO)
    if "pydevd" not in sys.modules:
        fh = logging.FileHandler(filename=filename, mode='w')
        fh.setFormatter(logging.Formatter('%(message)s'))
        fh.setLevel(logging.INFO)
    logger = logging.getLogger('training logger')
    logger.addHandler(sh)
    if "pydevd" not in sys.modules:
        logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    return logger
