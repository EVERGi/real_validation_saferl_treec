import argparse

from src.reproduce_svl.custom_experiment import get_results
from src.plot_results.plot_all_figures import (
    plot_all_paper_figures,
    plot_reproduced_figures,
)
from src.training_trees.train import train_all_houses
import os
from os.path import abspath, dirname
from src.forecasting.all_models import paper_forecasting_train


def train_treec(pop_size=1000, gen=1000):
    new_models_dir = "data/new_models/treec/"
    train_all_houses(new_models_dir, pop_size, gen)


def reproduce_paper_results(forecasting_model_dir, treec_model_dir, pretrained_rl_dir):
    get_results(forecasting_model_dir, treec_model_dir, pretrained_rl_dir)


def train_forecasting():
    new_models_dir = "data/new_models/forecasting/"
    paper_forecasting_train(new_models_dir)


def plot_paper_figures():
    plot_all_paper_figures()


def plot_new_reproduced_figures(result_dir="data/reproduction_results/"):

    plot_reproduced_figures(result_dir)


def main():
    os.chdir(dirname(abspath(__file__)))
    # Generate a command line tool that can be used to reproduce the paper results
    # The parser should have the following options:
    # - train_treec: Train new TreeC models with optional pop_size and gen arguments (default pop_size=1000, gen=1000)
    # - train_forecasting: Train new forecasting models
    # - reproduce_paper_results: Reproduce the results from the paper (default forecasting_model_dir="data/ems_models/forecasting/", treec_model_dir="data/ems_models/treec/", pretrained_rl_dir="data/ems_models/pretrain_rl/")
    # - plot_paper_figures: Plot all the figures from the paper
    # - plot_new_reproduced_figures: Plot the reproduced figures (default result_dir="data/reproduction_results/")
    parser = argparse.ArgumentParser(
        description="Reproduce the paper results",
        epilog="""
To plot the figures from the paper:
python reproduce_paper.py --plot_paper_figures\n\n
To reproduce the results from the paper (Change values of forecasting_model_dir, treec_model_dir, and pretrained_rl_dir to use different models than the ones used in the paper):
python reproduce_paper.py --reproduce_paper_results\n\n
To plot the figures of the reproduced results:
python reproduce_paper.py --plot_new_reproduced_figures\n\n
To train new TreeC models with a very small population and few generations (Remove the pop_size and gen arguments to train with the values used in the paper): 
python reproduce_paper.py --train_treec --pop_size 10 --gen 10\n\n
To train new forecasting models:
python reproduce_paper.py --train_forecasting\n\n
""",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--plot_paper_figures",
        action="store_true",
        help="Plot all the figures from the paper",
    )

    group.add_argument(
        "--reproduce_paper_results",
        action="store_true",
        help="Reproduce the results from the paper",
    )
    parser.add_argument(
        "--forecasting_model_dir",
        default="data/ems_models/forecasting/",
        help="Directory of the forecasting models used by the reproduce paper results command (default: data/ems_models/forecasting/)",
    )
    parser.add_argument(
        "--treec_model_dir",
        default="data/ems_models/treec/",
        help="Directory of the TreeC models used by the reproduce paper results command (default: data/ems_models/treec/)",
    )
    parser.add_argument(
        "--pretrained_rl_dir",
        default="data/ems_models/pretrain_rl/",
        help="Directory of the pretrained RL models used by the reproduce paper results command (default: data/ems_models/pretrain_rl/)",
    )

    group.add_argument(
        "--plot_new_reproduced_figures",
        action="store_true",
        help="Plot the figures from the reproduced results",
    )
    parser.add_argument(
        "--reproduced_result_dir",
        default="data/reproduction_results/",
        help="Directory containing the reproduced results used by the plot_new_reproduced_figures command (default: data/reproduction_results/)",
    )

    group.add_argument(
        "--train_treec",
        action="store_true",
        help="Train new TreeC models with optional pop_size and gen arguments",
    )
    parser.add_argument(
        "--pop_size",
        type=int,
        default=1000,
        help="Population size of the TreeC training (default: 1000)",
    )
    parser.add_argument(
        "--gen",
        type=int,
        default=1000,
        help="Number of generations of the TreeC training (default: 1000)",
    )

    group.add_argument(
        "--train_forecasting",
        action="store_true",
        help="Train new forecasting models",
    )

    args = parser.parse_args()

    if args.train_treec:
        train_treec(args.pop_size, args.gen)
    elif args.train_forecasting:
        train_forecasting()
    elif args.reproduce_paper_results:
        reproduce_paper_results(
            args.forecasting_model_dir, args.treec_model_dir, args.pretrained_rl_dir
        )
    elif args.plot_paper_figures:
        plot_paper_figures()
    elif args.plot_new_reproduced_figures:
        plot_new_reproduced_figures(args.reproduced_result_dir)


if __name__ == "__main__":
    # plot_paper_figures()
    main()
