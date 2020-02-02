import matplotlib.pyplot as plt


def plot_trajectory_evaluation(pred_states, pred_delta_vars, trajectory):
    plt.clf()
    state_size = pred_states.shape[2]
    ensemble_size = pred_states.shape[1]
    steps = range(pred_states.shape[0])
    fig, axes = plt.subplots(state_size, 1, figsize=(15, int(state_size * 2)))
    for i in range(state_size):
        for j in range(ensemble_size):
            top = pred_states[:, j, i] + pred_delta_vars[:, j, i]
            bottom = pred_states[:, j, i] - pred_delta_vars[:, j, i]
            axes[i].plot(pred_states[:, j, i], color="r")
            axes[i].fill_between(steps, top, bottom, color="r", alpha=0.3)
            axes[i].plot(trajectory[:, i], color="g")
            axes[i].spines["top"].set_visible(False)
            axes[i].spines["right"].set_visible(False)
    return fig
