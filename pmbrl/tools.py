# pylint: disable=not-callable
# pylint: disable=no-member

import os
import json
import cv2
import pickle

import numpy as np
import torch

MODEL_FILE = "model_{}.pth"
BUFFER_FILE = "buffer_{}.npz"
METRICS_FILE = "metrics_{}.json"
LOGFILE = "out.txt"


def save_model(logdir, ensemble, reward, optim, episode):
    path = os.path.join(logdir, MODEL_FILE.format(episode))
    save_dict = {
        "ensemble": ensemble.state_dict(),
        "reward": reward.state_dict(),
        "optim": optim.state_dict(),
    }
    torch.save(save_dict, path)
    log("Saved _models_ at `{}`".format(path))


def save_metrics(logdir, metrics, episode):
    path = os.path.join(logdir, METRICS_FILE.format(episode))
    _save_json(path, metrics)
    log("Saved _metrics_ at path `{}`".format(path))


def save_buffer(logdir, buffer, episode):
    path = os.path.join(logdir, BUFFER_FILE.format(episode))
    buffer.save_data(path)
    log("Saved _buffer_ at path `{}`".format(path))


def _save_json(path, obj):
    with open(path, "w") as file:
        json.dump(obj, file)


def _save_pickle(path, obj):
    with open(path, "wb") as pickle_file:
        pickle.dump(obj, pickle_file)


def log(string):
    f = open(LOGFILE, "a+")
    f.write("\n")
    f.write(str(string))
    print(string)
    f.close()
