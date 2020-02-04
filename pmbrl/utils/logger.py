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

    def log_stats(self, stats):
        reward_stats, info_stats = stats
        self.metrics['reward_stats'].append(reward_stats)
        self.metrics['info_stats'].append(info_stats)
        msg = "Reward statistics: \n [max {:.2f} min {:.2f} mean {:.2f} std {:.2f}]"
        self.log(
            msg.format(
                reward_stats["max"],
                reward_stats["min"],
                reward_stats["mean"],
                reward_stats["std"],
            )
        )
        msg = "Information gain statistics: \n [max {:.2f} min {:.2f} mean {:.2f} std {:.2f}]"
        self.log(
            msg.format(
                info_stats["max"],
                info_stats["min"],
                info_stats["mean"],
                info_stats["std"],
            )
        )

    def save(self):
        self._save_json(self.metrics_path, self.metrics)
        self.log("Saved _metrics_")

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
            "reward_stats": [],
            "info_stats": [],
        }

    def _save_json(self, path, obj):
        with open(path, "w") as file:
            json.dump(obj, file)

    @property
    def img_path(self):
        return self._img_path
