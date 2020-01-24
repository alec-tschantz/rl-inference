python main.py --logdir "log_cheetah_01"
python main.py --logdir "log_cheetah_02"
python main.py --logdir "log_cheetah_03"
python main.py --logdir "log_cheetah_04"
python main.py --logdir "log_cheetah_05"

python main.py --logdir "log_ant_01" --env_name "SparseAnt" --max_episode_len 300 --use_reward False --record_coverage True
python main.py --logdir "log_ant_02" --env_name "SparseAnt" --max_episode_len 300 --use_reward False --record_coverage True
python main.py --logdir "log_ant_03" --env_name "SparseAnt" --max_episode_len 300 --use_reward False --record_coverage True
python main.py --logdir "log_ant_04" --env_name "SparseAnt" --max_episode_len 300 --use_reward False --record_coverage True
python main.py --logdir "log_ant_05" --env_name "SparseAnt" --max_episode_len 300 --use_reward False --record_coverage True