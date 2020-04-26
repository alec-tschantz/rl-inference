# Reinforcement Learning through Active Inference

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

Repository for ICLR 2020 submission to Bridging AI and Cognitive Science (BAICS). 

[[Paper](https://arxiv.org/abs/2002.12636) / [Presentation](https://baicsworkshop.github.io/program/baics_37.html)]

### Requirements

The `environment.yml` contains the exact package versions, but recent versions of the following packages should be sufficient:

- `torch`
- `gym`
- `numpy`
- `scipy`

If you wish to use `mujoco` or `dm_control` environments you will also require:  

- `mujoco_py` 
- `dm_control` 

### Running

```
git clone https://github.com/alec-tschantz/rl-inference.git
cd rl-inference
python scripts/train.py --config_name="mountain_car" --logdir="mountain_car" --strategy="information" --seed=0
```

The provided configs are [`mountain_car`, `cup_catch`, `half_cheetah_run`,  `half_cheetah_flip` `ant_maze`, `reacher`p

### Acknowledgments

- Alec Tschantz [@alec-tschantz](https://github.com/alec-tschantz)
- Beren Millidge [@bmillidge](https://github.com/BerenMillidge)

The code for the ensemble model was adapted from [max](`https://github.com/nnaisense/max`)