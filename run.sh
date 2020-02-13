# full exploration
python scripts/train.py --config_name="cup_catch" --logdir="cup_catch" --seed=0
python scripts/train.py --config_name="cup_catch" --logdir="cup_catch" --seed=1
python scripts/train.py --config_name="cup_catch" --logdir="cup_catch" --seed=2
python scripts/train.py --config_name="cup_catch" --logdir="cup_catch" --seed=3
python scripts/train.py --config_name="cup_catch" --logdir="cup_catch" --seed=4

# no exploration
python scripts/train.py --config_name="cup_catch" --logdir="cup_catch_no_expl" --strategy="none" --seed=0
python scripts/train.py --config_name="cup_catch" --logdir="cup_catch_no_expl" --strategy="none" --seed=1
python scripts/train.py --config_name="cup_catch" --logdir="cup_catch_no_expl" --strategy="none" --seed=2
python scripts/train.py --config_name="cup_catch" --logdir="cup_catch_no_expl" --strategy="none" --seed=3
python scripts/train.py --config_name="cup_catch" --logdir="cup_catch_no_expl" --strategy="none" --seed=4

# variance
python scripts/train.py --config_name="cup_catch" --logdir="cup_catch_variance" --strategy="variance" --seed=0
python scripts/train.py --config_name="cup_catch" --logdir="cup_catch_variance" --strategy="variance" --seed=1
python scripts/train.py --config_name="cup_catch" --logdir="cup_catch_variance" --strategy="variance" --seed=2
python scripts/train.py --config_name="cup_catch" --logdir="cup_catch_variance" --strategy="variance" --seed=3
python scripts/train.py --config_name="cup_catch" --logdir="cup_catch_variance" --strategy="variance" --seed=4


###############################################################################################
#                                       Ant Maze
#                                 @TODO state space plot
###############################################################################################

# full exploration
python scripts/train.py --config_name="ant_maze" --logdir="ant_maze" --seed=0
python scripts/train.py --config_name="ant_maze" --logdir="ant_maze" --seed=1
python scripts/train.py --config_name="ant_maze" --logdir="ant_maze" --seed=2
python scripts/train.py --config_name="ant_maze" --logdir="ant_maze" --seed=3
python scripts/train.py --config_name="ant_maze" --logdir="ant_maze" --seed=4

# random exploration

python scripts/train.py --config_name="ant_maze" --logdir="ant_maze_random" --strategy="random" --seed=0
python scripts/train.py --config_name="ant_maze" --logdir="ant_maze_random" --strategy="random" --seed=1
python scripts/train.py --config_name="ant_maze" --logdir="ant_maze_random" --strategy="random" --seed=2
python scripts/train.py --config_name="ant_maze" --logdir="ant_maze_random" --strategy="random" --seed=3
python scripts/train.py --config_name="ant_maze" --logdir="ant_maze_random" --strategy="random" --seed=4

# variance
python scripts/train.py --config_name="ant_maze" --logdir="ant_maze_variance" --strategy="variance" --seed=0
python scripts/train.py --config_name="ant_maze" --logdir="ant_maze_variance" --strategy="variance" --seed=1
python scripts/train.py --config_name="ant_maze" --logdir="ant_maze_variance" --strategy="variance" --seed=2
python scripts/train.py --config_name="ant_maze" --logdir="ant_maze_variance" --strategy="variance" --seed=3
python scripts/train.py --config_name="ant_maze" --logdir="ant_maze_variance" --strategy="variance" --seed=4


###############################################################################################
#                                      Mountain Car
#                 @TODO SAC baseline, e-greedy exploration, state space plot
###############################################################################################

# full exploration
python scripts/train.py --config_name="mountain_car" --logdir="mountain_car" --seed=0
python scripts/train.py --config_name="mountain_car" --logdir="mountain_car" --seed=1
python scripts/train.py --config_name="mountain_car" --logdir="mountain_car" --seed=2
python scripts/train.py --config_name="mountain_car" --logdir="mountain_car" --seed=3
python scripts/train.py --config_name="mountain_car" --logdir="mountain_car" --seed=4

# no exploration
python scripts/train.py --config_name="mountain_car" --logdir="mountain_car_no_expl" --strategy="none" --seed=0
python scripts/train.py --config_name="mountain_car" --logdir="mountain_car_no_expl" --strategy="none" --seed=1
python scripts/train.py --config_name="mountain_car" --logdir="mountain_car_no_expl" --strategy="none" --seed=2
python scripts/train.py --config_name="mountain_car" --logdir="mountain_car_no_expl" --strategy="none" --seed=3
python scripts/train.py --config_name="mountain_car" --logdir="mountain_car_no_expl" --strategy="none" --seed=4

# variance
python scripts/train.py --config_name="mountain_car" --logdir="mountain_car_variance" --strategy="variance" --seed=0
python scripts/train.py --config_name="mountain_car" --logdir="mountain_car_variance" --strategy="variance" --seed=1
python scripts/train.py --config_name="mountain_car" --logdir="mountain_car_variance" --strategy="variance" --seed=2
python scripts/train.py --config_name="mountain_car" --logdir="mountain_car_variance" --strategy="variance" --seed=3
python scripts/train.py --config_name="mountain_car" --logdir="mountain_car_variance" --strategy="variance" --seed=4


###############################################################################################
#
#                                         Reacher
#                        
###############################################################################################

# full exploration
python scripts/train.py --config_name="reacher" --logdir="reacher" --seed=0
python scripts/train.py --config_name="reacher" --logdir="reacher" --seed=1
python scripts/train.py --config_name="reacher" --logdir="reacher" --seed=2
python scripts/train.py --config_name="reacher" --logdir="reacher" --seed=3
python scripts/train.py --config_name="reacher" --logdir="reacher" --seed=4

# no exploration
python scripts/train.py --config_name="reacher" --logdir="reacher_no_expl" --strategy="none" --seed=0
python scripts/train.py --config_name="reacher" --logdir="reacher_no_expl" --strategy="none" --seed=1
python scripts/train.py --config_name="reacher" --logdir="reacher_no_expl" --strategy="none" --seed=2
python scripts/train.py --config_name="reacher" --logdir="reacher_no_expl" --strategy="none" --seed=3
python scripts/train.py --config_name="reacher" --logdir="reacher_no_expl" --strategy="none" --seed=4


# variance
python scripts/train.py --config_name="reacher" --logdir="reacher_variance" --strategy="variance" --seed=0
python scripts/train.py --config_name="reacher" --logdir="reacher_variance" --strategy="variance" --seed=1
python scripts/train.py --config_name="reacher" --logdir="reacher_variance" --strategy="variance" --seed=2
python scripts/train.py --config_name="reacher" --logdir="reacher_variance" --strategy="variance" --seed=3
python scripts/train.py --config_name="reacher" --logdir="reacher_variance" --strategy="variance" --seed=4