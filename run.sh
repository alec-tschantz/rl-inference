python main.py --logdir "log_cheetah_vanilla" 
python main.py --logdir "log_cheetah_expl_scale" --expl_scale=0.1
python main.py --logdir "log_cheetah_mpc" --n_candidates=1000 --optimisation_iters=10 --top_candidates=100
python main.py --logdir "log_cheetah_training" --n_train_epochs=100
python main.py --logdir "log_cheetah_expl_scale_large" --expl_scale=0.01
python main.py --logdir "log_cheetah_ensemble_large" --ensemble_size=30
python main.py --logdir "log_cheetah_hidden_large" --hidden_size=500
python main.py --logdir "log_cheetah_horizon" --plan_horizon=20
