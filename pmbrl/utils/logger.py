import os
import json
from datetime import datetime


class Logger(object):
    def __init__(self, logdir, seed):
        self.logdir = logdir
        self.seed = seed
        self.path = logdir + "_" + str(seed) + "/"
        self._img_path = self.path + "/img/"
        self.outfile = self.path + "out.txt"
        self.metrics_path = self.path + "metrics.json"
        self.metrics = {}
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self._img_path, exist_ok=True)
        self._init_outfile()
        self._setup_metrics()

    def log(self, string):
        f = open(self.outfile, "a")
        f.write("\n")
        f.write(str(string))
        f.close()
        print(string)

    def log_losses(self, e_loss, r_loss):
        self.metrics["e_losses"].append(e_loss)
        self.metrics["r_losses"].append(r_loss)
        msg = "Ensemble loss {:.2f} / Reward Loss {:.2f}"
        self.log(msg.format(e_loss, r_loss))

    def log_episode(self, reward, steps):
        self.metrics["rewards"].append(reward)
        self.metrics["steps"].append(steps)
        msg = "Rewards {:.2f} / Steps {:.2f}"
        self.log(msg.format(reward, steps))

    def log_time(self, time):
        self.metrics["times"].append(time)
        self.log("Episode time {:.2f}".format(time))

    def save(self):
        self._save_json(self.metrics_path, self.metrics)

    def _init_outfile(self):
        f = open(self.outfile, "w")
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        f.write(current_time)
        f.close()

    def _setup_metrics(self):
        self.metrics = {
            "e_losses": [],
            "r_losses": [],
            "rewards": [],
            "steps": [],
            "times": [],
        }

    def _save_json(self, path, obj):
        with open(path, "w") as file:
            json.dump(obj, file)

    @property
    def img_path(self):
        return self._img_path
