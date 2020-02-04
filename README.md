# Exploration from exploitation

Repository for `ICML` 2020 submission

### Requirements

- `torch`
- `gym`
- `numpy`
- `scipy`
- `matplotlib` _optional_
- `mujoco_py` _optional_
- `dm_control` _optional_
- `roboschool` _optional_

### Environments

- `SparseMountainCar`: run using the `train_base` script with the `mountain_car` config
- `SparseCartpoleSwingup`: run using the `train_base` script with the `cartpole` config
- `SparseCupCatch`: run using the `train_base` script with the `cup_catch` config

- `SparseHalfCheetahRun`: run using the `train_iter` script with the `half_cheetah_run` config
- `SparseHalfCheetahFlip`: run using the `train_iter` script with the `half_cheetah_flip` config
- `SparseAntMaze`: run using the `train_exploration` script with the `ant_maze` config