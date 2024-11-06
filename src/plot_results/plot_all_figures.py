from plot_comparison_results import (
    plot_house_3_tree_improvement,
    plot_experiment_simulation,
)
from plot_example_day import plot_example_day
from plot_rl_improvement import rl_progress_plot
from plot_tensorboard_logs import plot_tensorboard_logs
from plot_safety_layer import plot_safety_layer
import matplotlib.pyplot as plt


def plot_all_paper_figures():
    result_dir = "data/results/"

    plot_safety_layer(result_dir)
    print("Plotted the safety layer metrics")
    plot_experiment_simulation(result_dir)
    print("Plotted experiment simulation comparison results")

    plot_house_3_tree_improvement(result_dir)
    print("Plotted house 3 improvement with RBC policy for TreeC algorithm")
    plot_example_day()
    print("Plotted the example day")
    rl_progress_plot(result_dir)
    print("Plotted the RL improvement over the experiment")
    plot_tensorboard_logs(result_dir)
    print("Plotted the critic loss and actor loss")

    print("All figures plotted successfully!")
    plt.show()


def plot_reproduced_figures(result_dir="data/reproduction_results/"):

    plot_experiment_simulation(result_dir)
    print("Plotted experiment simulation comparison results")

    # plot_house_3_tree_improvement(result_dir)
    # print("Plotted house 3 improvement with RBC policy for TreeC algorithm")

    print("All figures plotted successfully!")
    plt.show()


if __name__ == "__main__":
    # plot_all_paper_figures()
    plot_reproduced_figures()
