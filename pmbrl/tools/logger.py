import os

from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    def __init__(self, logdir="log", exp_name=None):
        if exp_name is None:
            path = logdir
        else:
            path = logdir + "/" + exp_name
        self.writer = SummaryWriter(path)

    def log_scalar(self, name, value, episode):
        self.writer.add_scalar(name, value, episode)

    def log_figure(self, name, fig, episode):
        self.writer.add_figure(name, fig, episode, close=True)

    def close(self):
        self.writer.close()
