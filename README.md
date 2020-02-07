# Exploration from exploitation

Repository for `ICML` 2020 submission

### Requirements

- `torch`
- `gym`
- `mujoco_py` 
- `dm_control` 
- `Box2D`
- `numpy`
- `scipy`
- `matplotlib` 

### Environments

- `SparseMountainCar`: run using the `train` script with the `mountain_car` config
- `CupCatch`: run using the `train` script with the `cup_catch` config
- `HalfCheetahRun`: run using the `train` script with the `half_cheetah_run` config
- `HalfCheetahFlip`: run using the `train` script with the `half_cheetah_flip` config

### Tasks

- Setup action noise exploration and no exploration configs
- Can we do exploration + exploitation vs. exploration vs. exploitation in train phases? 