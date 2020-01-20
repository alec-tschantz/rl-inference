# pylint: disable=not-callable
# pylint: disable=no-member

import os
import json
import cv2
import pickle

import numpy as np
import torch

MODEL_FILE = "model_{}.pth"
BUFFER_FILE = "buffer.npz"
METRICS_FILE = "metrics.json"
NORMALIZER_FILE = "normalizer.pth"
LOGFILE = "out.txt"


def logdir_exists(logdir):
    return os.path.exists(logdir)


def init_dirs(logdir):
    os.makedirs(logdir, exist_ok=True)


def load_buffer(logdir, buffer):
    buffer_path = os.path.join(logdir, BUFFER_FILE)
    buffer.load_data(buffer_path)
    return buffer


def load_model_dict(logdir, episode):
    model_path = os.path.join(logdir, MODEL_FILE.format(episode))
    return torch.load(model_path)


def load_metrics(logdir):
    path = os.path.join(logdir, METRICS_FILE)
    return _load_json(path)


def load_normalizer(logdir):
    path = os.path.join(logdir, NORMALIZER_FILE)
    return _load_pickle(path)


def save_model(logdir, ensemble, reward, optim, episode):
    path = os.path.join(logdir, MODEL_FILE.format(episode))
    save_dict = {
        "ensemble": ensemble.state_dict(),
        "reward": reward.state_dict(),
        "optim": optim.state_dict(),
    }
    torch.save(save_dict, path)
    log("Saved _models_ at `{}`".format(path))


def save_metrics(logdir, metrics):
    path = os.path.join(logdir, METRICS_FILE)
    _save_json(path, metrics)
    log("Saved _metrics_ at path `{}`".format(path))


def save_buffer(logdir, buffer):
    path = os.path.join(logdir, BUFFER_FILE)
    buffer.save_data(path)
    log("Saved _buffer_ at path `{}`".format(path))


def save_normalizer(logdir, normalizer):
    path = os.path.join(logdir, NORMALIZER_FILE)
    _save_pickle(path, normalizer)
    log("Saved _normalizer_ at path `{}`".format(path))


def build_metrics():
    return {
        "episode": 0,
        "last_save": 0,
        "ensemble_loss": [],
        "reward_loss": [],
        "train_rewards": [],
        "train_steps": [],
        "test_rewards": [],
        "test_steps": [],
        "total_steps": [],
        "episode_time": [],
    }


def _load_json(path):
    with open(path, "r") as file:
        data = json.load(file)
    return data


def _save_json(path, obj):
    with open(path, "w") as file:
        json.dump(obj, file)


def _load_pickle(path):
    with open(path, "rb") as pickle_file:
        data = pickle.load(pickle_file)
    return data


def _save_pickle(path, obj):
    with open(path, "wb") as pickle_file:
        pickle.dump(obj, pickle_file)


def log(string):
    f = open(LOGFILE, "a+")
    f.write("\n")
    f.write(str(string))
    print(string)
    f.close()
