import os
import sys

filed_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(filed_dir)
sys.path.append(parent_dir)

import shutil

from reproduce_svl.reproduce_experiment import (
    ThreeMonthsTree,
    get_model_path,
    run_experiment,
)
from reproduce_svl.reproduce_experiment_gym import run_rl_experiment, ExecRL
from reproduce_svl.custom_classes import (
    RLApproximation,
    RationalSVL,
    SVLTreeBattChargRule,
)
from reproduce_svl.mpc import BugMPC, PerfectExpMPC, SimulExpMPC

from forecasting.model_forecaster import ThreeMonthForecaster
from training_trees.input_function import reduced_state_input_norm_price

from treec.logger import TreeLogger


tmp_results_dir = "tmp_results/"


def get_results(forecasting_model_dir, treec_model_dir, pretrained_rl_dir):

    houses = [1, 2, 3, 5]
    date_vm = "2024-06-17 15:00:00"
    run_date_format = date_vm.replace(" ", "_").replace(":", "")[:-2]

    print("Executing results for MPC, RBC and TreeC")
    get_mpc_rbc_treec_results(
        run_date_format, houses, forecasting_model_dir, treec_model_dir
    )

    print("Executing results for RL")
    get_rl_results(houses, pretrained_rl_dir)

    print("Executing results for perfect MPC")
    get_perfect_mpc_results(run_date_format, houses)

    dir_path = "data/reproduction_results/"
    create_results_reproduction_dir(dir_path)

    print("Results generation finished succesfully")
    print(f"New results have been saved to {dir_path}")
    # Remove the temporary results directory
    shutil.rmtree(tmp_results_dir)


def get_mpc_rbc_treec_results(
    run_date_format, houses, forecasting_model_dir, treec_model_dir
):
    log_folder = f"{tmp_results_dir}experiment_mpc_rbc_treec/"
    CustomMPC = get_custom_mpc_class(forecasting_model_dir)
    CustomTree = get_custom_treec_class(treec_model_dir)

    env_to_ems = {
        "RBC": RationalSVL,
        "MPC": CustomMPC,
        "RL": RationalSVL,
        "Tree": CustomTree,
    }

    run_experiment(
        run_date_format,
        houses,
        log_folder,
        env_to_ems=env_to_ems,
    )


def get_rl_results(houses, pretrained_rl_dir):

    env_to_ems = {
        "RBC": RationalSVL,
        "MPC": RationalSVL,
        "RL": ExecRL,
        "Tree": RationalSVL,
    }
    log_folder = f"{tmp_results_dir}experiment_rl/"
    run_rl_experiment(
        houses,
        log_folder=log_folder,
        env_to_ems=env_to_ems,
        pretrain_dir=pretrained_rl_dir,
    )


def get_perfect_mpc_results(run_date_format, houses):

    log_folder = f"{tmp_results_dir}perfect_mpc/"

    env_to_ems = {
        "RBC": PerfectExpMPC,
        "MPC": PerfectExpMPC,
        "RL": PerfectExpMPC,
        "Tree": PerfectExpMPC,
    }
    run_experiment(
        run_date_format,
        houses,
        log_folder,
        env_to_ems,
    )


def create_results_reproduction_dir(dir_path="data/reproduction_results/"):
    if os.path.exists(dir_path):
        # remove the directory
        shutil.rmtree(dir_path)

    os.makedirs(dir_path, exist_ok=True)
    perfect_mpc_name = "perfect_mpc/"
    perfect_mpc_tmp = f"{tmp_results_dir}{perfect_mpc_name}"
    shutil.copytree(perfect_mpc_tmp, dir_path + perfect_mpc_name)

    tmp_mpc_rbc_dir = f"{tmp_results_dir}experiment_mpc_rbc_treec/"
    reproduction_experiment_dir = dir_path + "experiment_simulation/"
    shutil.copytree(tmp_mpc_rbc_dir, reproduction_experiment_dir)

    tmp_rl_dir = f"{tmp_results_dir}experiment_rl/"

    files_to_copy = [f"house_{i}_RL_grid.csv" for i in [1, 2, 3, 5]]
    for file in files_to_copy:
        shutil.copy(tmp_rl_dir + file, reproduction_experiment_dir + file)


"""
def get_bug_mpc_results():
    log_folder = f"{tmp_results_dir}bug_mpc_data_cons/"

    houses = [1, 2, 3, 5]
    date_vm = "2024-06-17 15:00:00"
    run_date_format = date_vm.replace(" ", "_").replace(":", "")[:-2]

    env_to_ems = {
        "RBC": RationalSVL,
        "MPC": BugMPC,
        "RL": RationalSVL,
        "Tree": RationalSVL,
    }
    run_experiment(
        run_date_format,
        houses,
        log_folder,
        env_to_ems,
    )
"""


def get_custom_mpc_class(forecasting_model_dir):

    class CustomMPC(SimulExpMPC):
        def __init__(self, microgrid, params_evaluation):
            house_num = params_evaluation["house_num"]
            self.forecaster = ThreeMonthForecaster(
                "model", house_num, forecasting_model_dir
            )
            super().__init__(microgrid, params_evaluation)

    return CustomMPC


def get_custom_treec_class(treec_model_dir):
    class CustomTree(SVLTreeBattChargRule):
        def __init__(
            self,
            microgrid,
            house_num,
        ):
            trees = TreeLogger.get_best_trees(
                get_model_path(house_num, treec_model_dir)
            )
            input_func = reduced_state_input_norm_price
            super().__init__(microgrid, trees, input_func, None)

    return CustomTree


if __name__ == "__main__":
    forecasting_model_dir = "data/ems_models/forecasting/"
    treec_model_dir = "data/ems_models/treec/"
    pretrained_rl_dir = "data/ems_models/pretrain_rl/"
    get_results(forecasting_model_dir, treec_model_dir, pretrained_rl_dir)
    # create_results_reproduction_dir()
