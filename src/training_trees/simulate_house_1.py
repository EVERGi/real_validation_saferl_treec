from simugrid.rewards.reward import DefaultReward
from simugrid.management.rational import RationalManager
from simugrid.simulation.config_parser import parse_config_file
from simugrid.misc.log_plot_micro import plot_hist, plot_attributes
import matplotlib.pyplot as plt
from reproduce_svl.custom_classes import (
    DayAheadEngie,
    DayAheadManager,
    DayAheadPrice,
    SVLHouseContinuousManager,
    SVLMPCManager,
    RBCManager,
    NaiveMPCManager,
)
from train import params_house

import os
import datetime
import seaborn as sns

import json

from treec.logger import TreeLogger


def run_house(config_file, Ems, plot_graph, tree_params=None):
    microgrid = parse_config_file(config_file)
    reward = DayAheadEngie()
    microgrid.set_reward(reward)

    if tree_params is None:
        Ems(microgrid)
    else:
        trees = tree_params["trees"]
        input_func = tree_params["input_func"]
        params_evaluation = tree_params["params_evaluation"]
        Ems(microgrid, trees, input_func, params_evaluation)

    microgrid.attribute_to_log("Battery_0", "soc")

    count = 0

    while microgrid.datetime != microgrid.end_time:
        # for _ in range(24 * 4):
        count += 1
        if count % 100 == 0:
            print(count)
        microgrid.management_system.simulate_step()

    power_hist = microgrid.power_hist
    kpi_hist = microgrid.reward_hist
    attributes_hist = microgrid.attributes_hist
    print(microgrid.tot_reward.KPIs)
    if plot_graph:
        plot_attributes(attributes_hist)
        plot_hist(power_hist, kpi_hist)
        plt.tight_layout()
        plt.show()
    return microgrid


def seperate_opex_in_days(kpi_hist):
    cur_dt = kpi_hist["datetime"][0].replace(hour=0, minute=0, second=0)
    first_opex = kpi_hist["opex"][0]
    dt_list = []
    opex_list = []

    for i, dt in enumerate(kpi_hist["datetime"]):
        new_dt = dt.replace(hour=0, minute=0, second=0)
        if new_dt != cur_dt:
            last_opex = kpi_hist["opex"][i - 1]
            opex_day = last_opex - first_opex
            opex_list.append(opex_day)
            dt_list.append(cur_dt)
            cur_dt = new_dt
            first_opex = kpi_hist["opex"][i]

    day_opex = [dt_list[1:], opex_list[1:]]

    return day_opex


def plot_day_opex(day_opex, label=None):
    plt.plot(day_opex[0], day_opex[1], label=label)
    # Get datetime tikcs to not overlap but not xticks rotation
    plt.gcf().autofmt_xdate()
    if label is not None:
        plt.legend()


def substract_no_batt(day_opex_batt, day_opex_no_batt):
    opex_diff = [i - j for i, j in zip(day_opex_batt[1], day_opex_no_batt[1])]
    day_opex_diff = [day_opex_batt[0], opex_diff]
    return day_opex_diff


def log_day_opex(day_opex, file_name):
    with open(file_name, "w") as f:
        for i in range(len(day_opex[0])):
            f.write(str(day_opex[0][i]) + "," + str(day_opex[1][i]) + "\n")


def read_day_opex(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()
        day_opex = [[], []]
        for line in lines:
            line = line.replace("\n", "").split(",")
            dt_0 = datetime.datetime.strptime(line[0], "%Y-%m-%d %H:%M:%S%z")
            day_opex[0].append(dt_0)
            day_opex[1].append(float(line[1]))
    return day_opex


def get_day_opex(
    config_file, Ems, recalculate=False, plot_graph=False, tree_params=None
):
    if plot_graph:
        recalculate = True
    folder_name = "results/day_opex/"
    if tree_params is None:
        path_end = ""
    else:
        path_end = tree_params["params_evaluation"]["ems_exten"]
    path_end += config_file.split("/")[-1]

    file_name = f"{folder_name}{Ems.__name__}_{path_end}"

    if recalculate or not os.path.isfile(file_name):
        microgrid = run_house(config_file, Ems, plot_graph, tree_params)
        kpi_hist = microgrid.reward_hist
        day_opex = seperate_opex_in_days(kpi_hist)
        log_day_opex(day_opex, file_name)
    else:
        day_opex = read_day_opex(file_name)

    return day_opex


def calc_mpc_house_1_and_3():
    house_1_batt_config = "data/houses_env/pv_1_cons_7.csv"
    Ems = SVLMPCManager
    get_day_opex(house_1_batt_config, Ems)

    house_3_batt_config = "data/houses_env/pv_3_cons_7.csv"
    Ems = SVLMPCManager
    get_day_opex(house_3_batt_config, Ems)


def plot_comp_mpc_rbc_house_1():
    house_1_batt_config = "data/houses_env/pv_1_cons_7.csv"
    Ems = RationalManager
    day_opex_batt = get_day_opex(house_1_batt_config, Ems)

    house_1_batt_config = "data/houses_env/pv_1_cons_7.csv"
    Ems = SVLMPCManager
    day_opex_mpc = get_day_opex(house_1_batt_config, Ems)

    house_1_no_batt_config = "data/houses_env/pv_1_cons_7_no_batt.csv"
    Ems = RationalManager
    day_opex_no_batt = get_day_opex(house_1_no_batt_config, Ems)

    day_opex_diff = substract_no_batt(day_opex_batt, day_opex_no_batt)
    day_opex_diff_mpc = substract_no_batt(day_opex_mpc, day_opex_no_batt)

    plt.figure()
    plot_day_opex(day_opex_diff_mpc)
    plot_day_opex(day_opex_diff)
    plt.show()


def plot_comp_rbc_house_1_and_3():
    house_1_batt_config = "data/houses_env/pv_1_cons_7.csv"
    Ems = RBCManager
    day_opex_1 = get_day_opex(house_1_batt_config, Ems)

    house_3_batt_config = "data/houses_env/pv_3_cons_7.csv"
    Ems = RBCManager
    day_opex_3 = get_day_opex(house_3_batt_config, Ems)

    house_1_no_batt_config = "data/houses_env/pv_1_cons_7_no_batt.csv"
    Ems = RationalManager
    day_opex_1_no_batt = get_day_opex(house_1_no_batt_config, Ems)

    house_3_no_batt_config = "data/houses_env/pv_3_cons_7_no_batt.csv"
    Ems = RationalManager
    day_opex_3_no_batt = get_day_opex(house_3_no_batt_config, Ems)

    day_opex_1_diff = substract_no_batt(day_opex_1, day_opex_1_no_batt)
    day_opex_3_diff = substract_no_batt(day_opex_3, day_opex_3_no_batt)

    plt.figure()
    plot_day_opex(day_opex_1_diff)
    plot_day_opex(day_opex_3_diff)
    plt.show()


def compare_mpc_rbc_tree():
    house_1_batt_config = "data/houses_env/pv_1_cons_7.csv"
    Ems = RBCManager
    day_opex_rbc = get_day_opex(house_1_batt_config, Ems)

    Ems = SVLMPCManager
    day_opex_mpc = get_day_opex(
        house_1_batt_config, Ems, recalculate=True, plot_graph=True
    )

    house_1_no_batt_config = "data/houses_env/pv_1_cons_7_no_batt.csv"
    Ems = RationalManager
    day_opex_no_batt = get_day_opex(house_1_no_batt_config, Ems)

    model_folder = (
        "svl_house_1/pv_1_cons_7_tree_119/validation/pv_1_cons_7_tree_0/models/"
    )
    day_opex_tree_h1 = get_day_opex_tree(house_1_batt_config, model_folder)

    day_opex_diff_rbc_h1 = substract_no_batt(day_opex_rbc, day_opex_no_batt)
    day_opex_diff_mpc_h1 = substract_no_batt(day_opex_mpc, day_opex_no_batt)
    day_opex_diff_tree_h1 = substract_no_batt(day_opex_tree_h1, day_opex_no_batt)

    plt.figure()
    plot_day_opex(day_opex_diff_rbc_h1, label="RBC house 1")
    plot_day_opex(day_opex_diff_mpc_h1, label="MPC house 1")
    plot_day_opex(day_opex_diff_tree_h1, label="Tree house 1")

    house_3_batt_config = "data/houses_env/pv_3_cons_7.csv"
    Ems = RBCManager
    day_opex_rbc = get_day_opex(house_3_batt_config, Ems)

    Ems = SVLMPCManager
    day_opex_mpc = get_day_opex(house_3_batt_config, Ems)

    model_folder = "svl_house_3/pv_3_cons_7_tree_6/models/"
    day_opex_tree_h3 = get_day_opex_tree(house_3_batt_config, model_folder)

    house_3_no_batt_config = "data/houses_env/pv_3_cons_7_no_batt.csv"
    Ems = RationalManager
    day_opex_no_batt = get_day_opex(house_3_no_batt_config, Ems)

    day_opex_diff_rbc_h3 = substract_no_batt(day_opex_rbc, day_opex_no_batt)
    day_opex_diff_mpc_h3 = substract_no_batt(day_opex_mpc, day_opex_no_batt)
    day_opex_diff_tree_h3 = substract_no_batt(day_opex_tree_h3, day_opex_no_batt)

    box_plot_data = {
        ("RBC", "H1"): day_opex_diff_rbc_h1[1],
        ("Tree", "H1"): day_opex_diff_tree_h1[1],
        ("MPC", "H1"): day_opex_diff_mpc_h1[1],
        ("RBC", "H3"): day_opex_diff_rbc_h3[1],
        ("Tree", "H3"): day_opex_diff_tree_h3[1],
        ("MPC", "H3"): day_opex_diff_mpc_h3[1],
    }
    plt.figure()
    plot_day_opex(day_opex_diff_rbc_h3, label="RBC house 3")
    plot_day_opex(day_opex_diff_mpc_h3, label="MPC house 3")
    plot_day_opex(day_opex_diff_tree_h3, label="Tree house 3")

    box_plot_comparison(box_plot_data)
    plt.show()


def box_plot_comparison(box_plot_data):
    df_dict = {"EMS": [], "House": [], "Opex": []}
    for key, items in box_plot_data.items():
        for opex_value in items:
            df_dict["EMS"].append(key[0])
            df_dict["House"].append(key[1])
            df_dict["Opex"].append(opex_value)

    plt.figure()

    sns.boxplot(data=df_dict, x="House", y="Opex", hue="EMS")
    # plt.show()


def get_ems_and_trees(model_folder):
    folder_split = model_folder.split("/")
    house_num = int(folder_split[0].replace("svl_house_", ""))
    train_num = folder_split[1].split("_")[-1]
    Ems = SVLHouseContinuousManager
    trees = TreeLogger.get_best_trees(model_folder)
    house_params, _ = params_house("", house_num, create_logger=False)
    house_params["ems_exten"] = f"h_{house_num}_t_{train_num}"

    return Ems, trees, house_params


def get_day_opex_tree(config_file, model_folder, recalculate=False, plot_graph=False):
    Ems, trees, house_params = get_ems_and_trees(model_folder)
    params_tree = {
        "trees": trees,
        "input_func": house_params["input_func"],
        "params_evaluation": house_params,
    }
    microgrid = get_day_opex(config_file, Ems, recalculate, plot_graph, params_tree)
    return microgrid


def plot_measured_vs_real_soc(house_num):
    # house_config = f"data/house_{house_num}/2023-09-14_12:15:00_2023-09-15_22:15:00/house_batt_fixed.json"
    house_config = f"data/house_{house_num}/2023-09-14_12:15:00_2023-09-15_22:15:00/house_batt_fixed.json"
    # Ems = NaiveMPCManager
    Ems = SVLMPCManager
    # Ems = RBCManager
    Ems = RationalManager
    microgrid = run_house(house_config, Ems, False)
    battery = microgrid.assets[0]

    microgrid_soc = microgrid.attributes_hist[battery]
    microgrid_soc = microgrid.attributes_hist[battery]

    env_soc_svl = microgrid.environments[0].env_values["soc_svl"].csv_list
    soc_svl = dict()
    soc_svl["datetime"] = [i[0] for i in env_soc_svl]
    soc_svl["soc"] = [i[1] for i in env_soc_svl]

    plt.figure()
    plt.plot(microgrid_soc["datetime"], microgrid_soc["soc"], label="simulation")
    plt.plot(soc_svl["datetime"], soc_svl["soc"], label="measurement")

    plt.ylim([-0.05, 1.05])
    plt.ylabel("SOC (-)")
    # Remove datetime overlap
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    """
    model_folder = (
        "svl_house_1/pv_1_cons_7_tree_119/validation/pv_1_cons_7_tree_0/models/"
    )
    Ems, trees, house_params = get_ems_and_trees(model_folder)

    house_1_batt_config = "data/houses_env/pv_1_cons_7.csv"
    day_opex_tree_h1 = get_day_opex_tree(house_1_batt_config, model_folder)

    house_1_no_batt_config = "data/houses_env/pv_1_cons_7_no_batt.csv"
    Ems = RationalManager
    day_opex_no_batt = get_day_opex(house_1_no_batt_config, Ems)
    day_opex_diff_tree_h1 = substract_no_batt(day_opex_tree_h1, day_opex_no_batt)
    plot_day_opex(day_opex_diff_tree_h1, label="Tree house 1")
    """
    # compare_mpc_rbc_tree()
    # Ems = NaiveMPCManager
    # Ems = SVLMPCManager
    # Ems = RBCManager
    # Ems = RationalManager

    compare_mpc_rbc_tree()
    # plot_measured_vs_real_soc(3)

    # svl_soc = microgrid.attributes_hist["Battery_0"]["svl_soc"]
    # get_day_opex(house_config, Ems, plot_graph=True)
