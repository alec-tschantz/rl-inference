import matplotlib.pyplot as plt


def plot_trajectory_predictions(pred_states, pred_delta_vars, trajectory):
    plt.cla()
    state_size = pred_states.shape[2]
    ensemble_size = pred_states.shape[1]
    steps = range(pred_states.shape[0])
    fig, axes = plt.subplots(state_size, 1, figsize=(20, 10))
    for i in range(state_size):
        for j in range(ensemble_size):
            top = pred_states[:, j, i] + pred_delta_vars[:, j, i]
            bottom = pred_states[:, j, i] - pred_delta_vars[:, j, i]
            axes[i].plot(pred_states[:, j, i], color="r", label="Predictions")
            axes[i].fill_between(steps, top, bottom, color="r", alpha=0.3)
            axes[i].plot(trajectory[:, i], color="g", label="Trajectory")
    plt.legend()
    return fig
