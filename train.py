#!/usr/bin/env python
import os

import json
import torch
import numpy as np
import queue
import pprint
import random
import argparse
import importlib
import threading
import traceback

from tqdm import tqdm
from utils import stdout_to_tqdm
from config import system_configs
from nnet.py_factory import NetworkFactory
from torch.multiprocessing import Process, Queue, Pool
from db.datasets import datasets

torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser(description="Train CornerNet")
    parser.add_argument("cfg_file", help="config file", type=str)
    parser.add_argument("--iter", dest="start_iter",
                        help="train at iteration i",
                        default=0, type=int)
    parser.add_argument("--threads", dest="threads", default=4, type=int)

    args = parser.parse_args()
    return args

def prefetch_data(db, queue, sample_data, data_aug):
    ind = 0
    print("start prefetching data...")
    np.random.seed(os.getpid())
    while True:
        try:
            data, ind = sample_data(db, ind, data_aug=data_aug)
            queue.put(data)
        except Exception as e:
            traceback.print_exc()
            raise e

def pin_memory(data_queue, pinned_data_queue, sema):
    while True:
        data = data_queue.get()

        data["xs"] = [x.pin_memory() for x in data["xs"]]
        data["ys"] = [y.pin_memory() for y in data["ys"]]

        pinned_data_queue.put(data)

        if sema.acquire(blocking=False):
            return

def init_parallel_jobs(dbs, queue, fn, data_aug):
    tasks = [Process(target=prefetch_data, args=(db, queue, fn, data_aug)) for db in dbs]
    for task in tasks:
        task.daemon = True
        task.start()
    return tasks

def train(training_dbs, validation_db, start_iter=0):
    learning_rate    = system_configs.learning_rate
    max_iteration    = system_configs.max_iter
    pretrained_model = system_configs.pretrain
    snapshot         = system_configs.snapshot
    val_iter         = system_configs.val_iter
    display          = system_configs.display
    decay_rate       = system_configs.decay_rate
    stepsize         = system_configs.stepsize

    # getting the size of each database
    training_size   = len(training_dbs[0].db_inds)
    validation_size = len(validation_db.db_inds)

    # queues storing data for training
    training_queue   = Queue(system_configs.prefetch_size)
    validation_queue = Queue(5)

    # queues storing pinned data for training
    pinned_training_queue   = queue.Queue(system_configs.prefetch_size)
    pinned_validation_queue = queue.Queue(5)

    # load data sampling function
    data_file   = "sample.{}".format(training_dbs[0].data)
    sample_data = importlib.import_module(data_file).sample_data

    # allocating resources for parallel reading
    training_tasks   = init_parallel_jobs(training_dbs, training_queue, sample_data, True)
    if val_iter:
        validation_tasks = init_parallel_jobs([validation_db], validation_queue, sample_data, False)

    training_pin_semaphore   = threading.Semaphore()
    validation_pin_semaphore = threading.Semaphore()
    training_pin_semaphore.acquire()
    validation_pin_semaphore.acquire()

    training_pin_args   = (training_queue, pinned_training_queue, training_pin_semaphore)
    training_pin_thread = threading.Thread(target=pin_memory, args=training_pin_args)
    training_pin_thread.daemon = True
    training_pin_thread.start()

    validation_pin_args   = (validation_queue, pinned_validation_queue, validation_pin_semaphore)
    validation_pin_thread = threading.Thread(target=pin_memory, args=validation_pin_args)
    validation_pin_thread.daemon = True
    validation_pin_thread.start()

    print("building model...")
    nnet = NetworkFactory(training_dbs[0])

    if pretrained_model is not None:
        if not os.path.exists(pretrained_model):
            raise ValueError("pretrained model does not exist")
        print("loading from pretrained model")
        nnet.load_pretrained_params(pretrained_model)

    if start_iter:
        learning_rate /= (decay_rate ** (start_iter // stepsize))

        nnet.load_params(start_iter)
        nnet.set_lr(learning_rate)
        print("training starts from iteration {} with learning_rate {}".format(start_iter + 1, learning_rate))
    else:
        nnet.set_lr(learning_rate)

    print("training start...")
    nnet.cuda()
    nnet.train_mode()
    with stdout_to_tqdm() as save_stdout:
        for iteration in tqdm(range(start_iter + 1, max_iteration + 1), file=save_stdout, ncols=80):
            training = pinned_training_queue.get(block=True)
            training_loss = nnet.train(**training)

            if display and iteration % display == 0:
                print("training loss at iteration {}: {}".format(iteration, training_loss.item()))
            del training_loss

            if val_iter and validation_db.db_inds.size and iteration % val_iter == 0:
                nnet.eval_mode()
                validation = pinned_validation_queue.get(block=True)
                validation_loss = nnet.validate(**validation)
                print("validation loss at iteration {}: {}".format(iteration, validation_loss.item()))
                nnet.train_mode()

            if iteration % snapshot == 0:
                nnet.save_params(iteration)

            if iteration % stepsize == 0:
                learning_rate /= decay_rate
                nnet.set_lr(learning_rate)

    # sending signal to kill the thread
    training_pin_semaphore.release()
    validation_pin_semaphore.release()

    # terminating data fetching processes
    for training_task in training_tasks:
        training_task.terminate()
    for validation_task in validation_tasks:
        validation_task.terminate()

if __name__ == "__main__":
    args = parse_args()

    cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")
    with open(cfg_file, "r") as f:
        configs = json.load(f)
            
    configs["system"]["snapshot_name"] = args.cfg_file
    system_configs.update_config(configs["system"])

    train_split = system_configs.train_split
    val_split   = system_configs.val_split

    print("loading all datasets...")
    dataset = system_configs.dataset
    # threads = max(torch.cuda.device_count() * 2, 4)
    threads = args.threads
    print("using {} threads".format(threads))
    training_dbs  = [datasets[dataset](configs["db"], train_split) for _ in range(threads)]
    validation_db = datasets[dataset](configs["db"], val_split)

    print("system config...")
    pprint.pprint(system_configs.full)

    print("db config...")
    pprint.pprint(training_dbs[0].configs)

    print("len of db: {}".format(len(training_dbs[0].db_inds)))
    train(training_dbs, validation_db, args.start_iter)
