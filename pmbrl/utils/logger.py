import os
import json
from datetime import datetime
import pprint


class Logger(object):
    def __init__(self, logdir, seed):
        self.logdir = logdir
        self.seed = seed
        self.path = logdir + "_" + str(seed) + "/"
        self.print_path = self.path + "out.txt"
        self.metrics_path = self.path + "metrics.json"
        self.video_dir = self.path + "videos/"
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)
        self.metrics = {}
        self._init_print()
        self._setup_metrics()

    def log(self, string):
        f = open(self.print_path, "a")
        f.write("\n")
        f.write(str(string))
        f.close()
        print(string)

    def log_losses(self, e_loss, r_loss):
        self.metrics["e_losses"].append(e_loss)
        self.metrics["r_losses"].append(r_loss)
        msg = "Ensemble loss {:.2f} / Reward Loss {:.2f}"
        self.log(msg.format(e_loss, r_loss))

    def log_coverage(self, coverage):
        self.metrics["coverage"].append(coverage)
        msg = "Coverage {:.2f}"
        self.log(msg.format(coverage))

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
        self.metrics["reward_stats"].append(reward_stats)
        self.metrics["info_stats"].append(info_stats)
        for key in reward_stats:
            reward_stats[key] = "{:.2f}".format(reward_stats[key])
        for key in info_stats:
            info_stats[key] = "{:.2f}".format(info_stats[key])
        self.log("Reward stats:\n {}".format(pprint.pformat(reward_stats)))
        self.log("Information gain stats:\n {}".format(pprint.pformat(info_stats)))

    def save(self):
        self._save_json(self.metrics_path, self.metrics)
        self.log("Saved _metrics_")

    def get_video_path(self, episode):
        return self.video_dir + "{}.mp4".format(episode)

    def _init_print(self):
        f = open(self.print_path, "w")
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
            "coverage": [],
        }

    def _save_json(self, path, obj):
        with open(path, "w") as file:
            json.dump(obj, file)

