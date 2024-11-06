# import working directory for custom classes
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import shutil
from reproduce_svl.custom_classes import (
    SVLTreeBattCharg,
    CustomControl,
    SVLTreeBattChargRule,
)
from reproduce_svl.mpc import SVLMPCManager

from treec.train import tree_train, tree_validate, cmaes_restart_train
from treec.logger import TreeLogger
from simugrid.simulation.config_parser import parse_config_file
from training_trees.evaluation_func import evaluate_microgrid_trees
from training_trees.input_function import (
    reduced_state_input_norm_price,
)

import warnings

warnings.simplefilter("ignore", RuntimeWarning)


def params_house(log_folder, house_num, create_logger=True, params_change={}):
    control_mode = "battery_and_grid"

    common_params = {
        "input_func": reduced_state_input_norm_price,
        "action_names": [[], []],
        "tot_steps": 1720,
        "eval_func": evaluate_microgrid_trees,
        "ManagerClass": SVLTreeBattChargRule,
        "house_num": house_num,
        "render": False,
        "control_mode": control_mode,
        "switch_hour": 15,
    }

    if "tree_nodes" in params_change.keys():
        tree_nodes = params_change["tree_nodes"]
        dimensions = 2 * (3 * tree_nodes + 1)
    else:
        dimensions = 2 * (3 * 20 + 1)
    num_gen = 1000
    algo_type = "tree"
    algo_params = {
        "gen": num_gen,
        "fixed": True,
        "dimension": dimensions,
        "pop_size": 1000,
        "single_threaded": False,
        "pygmo_algo": "cmaes",
    }

    for key, value in params_change.items():
        if key in common_params.keys():
            common_params[key] = value
        elif key in algo_params.keys():
            algo_params[key] = value

    microgrid, logger = setup_grid_logger_houses(
        house_num, algo_type, common_params, algo_params, log_folder, create_logger
    )

    common_params["microgrid"] = microgrid
    common_params["logger"] = logger

    return common_params, algo_params


def setup_grid_logger_houses(
    house_num, algo_type, common_params, algo_params, log_folder, create_logger
):
    filename = "house_train.json"

    config_dir = f"data/houses/house_{house_num}/2024-01-01_0000_2024-04-01_0000/"
    config_file = config_dir + filename
    common_params["config_file"] = config_file
    microgrid = parse_config_file(config_file)
    microgrid.attribute_to_log("Battery_0", "soc")

    if create_logger:
        logger = TreeLogger(log_folder, algo_type, common_params, algo_params)
    else:
        logger = None

    return microgrid, logger


def train_house(house_num, params_change={}, do_restart_train=False):
    log_folder = f"tmp_treec_train/svl_house_{house_num}"

    common_params, algo_params = params_house(
        log_folder, house_num, params_change=params_change
    )
    logger = common_params["logger"]

    if do_restart_train:
        cmaes_restart_train(common_params, algo_params)
    else:
        tree_train(common_params, algo_params)

    return logger.folder_name


def valid_house(
    training_folder, house_num, render=False, create_logger=True, params_change={}
):
    validate_folder = training_folder + "validation/"
    valid_params, _ = params_house(
        validate_folder,
        house_num,
        create_logger=create_logger,
        params_change=params_change,
    )
    valid_params["render"] = render

    prune_params = valid_params
    tree_validate(valid_params, training_folder, prune_params)


def train_and_valid_house(house_num, params_change={}, do_restart_train=False):
    log_folder = train_house(house_num, params_change, do_restart_train)

    valid_house(log_folder, house_num, params_change=params_change)
    return log_folder


def test_custom_ems():
    training_folder = "svl_house_1/house_training_simul_tree_0/"
    params_change = {"ManagerClass": CustomControl}
    valid_house(
        training_folder,
        1,
        render=True,
        create_logger=False,
        params_change=params_change,
    )


def test_MPC_ems():
    training_folder = "svl_house_1/house_training_simul_tree_0/"
    params_change = {"ManagerClass": SVLMPCManager}
    valid_house(
        training_folder,
        1,
        render=True,
        create_logger=False,
        params_change=params_change,
    )


def validate_hpc_runs():
    for house_num in [2, 3, 5]:
        for i in range(5):
            log_folder = (
                f"svl_house_{house_num}/house_training_simul_switch_15:00_tree_{i}/"
            )
            valid_house(log_folder, house_num)

    for i in range(5, 10):
        house_num = 1
        log_folder = (
            f"svl_house_{house_num}/house_training_simul_switch_15:00_tree_{i}/"
        )
        valid_house(log_folder, house_num)


def save_best_model(house_num, log_folders, new_models_dir):
    best_score = None
    best_folder = None
    for log_folder in log_folders:
        models_folder = log_folder + "validation/house_train_tree_0/models/"
        model_file = os.listdir(models_folder)[0]
        model_score = float(model_file.replace(".json", "").replace("model_0_", ""))
        if best_score is None or model_score > best_score:
            best_score = model_score
            best_folder = log_folder

    # Copy the best folder to the new models directory
    target_dir = new_models_dir + f"house_{house_num}/"
    os.makedirs(new_models_dir, exist_ok=True)
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    shutil.copytree(best_folder, target_dir)


def train_all_houses(new_models_dir, pop_size=1000, gen=1000):
    params_change = {
        "pop_size": pop_size,
        "gen": gen,
    }

    for house_num in [1, 2, 3, 5]:
        log_folder_list = []
        for _ in range(5):
            log_folder = train_and_valid_house(house_num, params_change)
            log_folder_list.append(log_folder)
        save_best_model(house_num, log_folder_list, new_models_dir)
    print("Training completed !")
    print(f"New models saved to {new_models_dir}")

    # Remove the temporary training folders
    shutil.rmtree("tmp_treec_train/")


if __name__ == "__main__":
    # for i in [12, 13, 14, 15, 16, 17]:
    #    training_folder = f"svl_house_1/house_training_simul_tree_{i} copy/"
    #    valid_house(training_folder, 1, render=False, create_logger=True)

    # params_change = {
    #    "pygmo_algo": "pso_gen",
    #    "pop_size": 1000,
    #    "gen": 100,
    # }
    params_change = {}
    debug = True
    if debug:
        params_change["single_threaded"] = True
        params_change["pop_size"] = 10

    train_and_valid_house(1, params_change)

    # validate_hpc_runs()

    # train_and_valid_house(2, params_change)
    # test_custom_ems()
    # test_MPC_ems()
