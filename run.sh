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