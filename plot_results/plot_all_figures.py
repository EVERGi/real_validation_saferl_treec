from plot_comparison_results import (
    plot_house_3_tree_improvement,
    plot_experiment_simulation,
)
from plot_example_day import plot_example_day
from plot_rl_improvement import rl_progress_plot
from plot_tensorboard_logs import plot_tensorboard_logs
import matplotlib.pyplot as plt


def plot_all_figures():

    plot_experiment_simulation()
    print("Plotted experiment simulation comparison results")
    plot_house_3_tree_improvement()
    print("Plotted house 3 improvement with RBC policy for TreeC algorithm")
    plot_example_day()
    print("Plotted the example day")
    rl_progress_plot()
    print("Plotted the RL improvement over the experiment")
    plot_tensorboard_logs()
    print("Plotted the critic loss and actor loss")

    print("All figures plotted successfully!")
    plt.show()


if __name__ == "__main__":
    plot_all_figures()
