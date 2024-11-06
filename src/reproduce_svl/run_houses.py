import sys

import os

filed_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(filed_dir)
sys.path.append(parent_dir)

from reproduce_svl.custom_classes import DayAheadEngie, RationalSVL, NoEMSSVL
from reproduce_svl.mpc import NaiveMPCManager, SVLMPCManager
from simugrid.simulation.config_parser import parse_config_file
from simugrid.misc.log_plot_micro import plot_hist, plot_attributes
from simugrid.management.rational import RationalManager
import json
from reproduce_svl.custom_classes import SVLTreeBattChargRule, RationalSVL
from treec.tree import BinaryTreeFixed

import matplotlib.pyplot as plt


def switch_hour_config(config_file, switch_hour):
    if switch_hour is not None:
        config_json = json.load(open(config_file, "r"))
        config_json["Environments"]["Environment 0"].pop(
            "./../../common_env/first_ev_schedule.csv"
        )
        config_json["Environments"]["Environment 0"][
            f"./../../common_env/first_ev_schedule_switch_{switch_hour}00.csv"
        ] = None
        config_file = config_file.replace(".json", f"_switch_{switch_hour}00.json")
        with open(config_file, "w") as f:
            json.dump(config_json, f, indent=4)

    return config_file


def run_house(config_file, Ems, plot_graph, switch_hour=None, tree_params=None):
    # config_file = switch_hour_config(config_file, switch_hour)

    microgrid = parse_config_file(config_file)
    reward = DayAheadEngie()
    microgrid.set_reward(reward)

    if tree_params is None:
        ems = Ems(microgrid)
    else:
        trees = tree_params["trees"]
        input_func = tree_params["input_func"]
        params_evaluation = tree_params["params_evaluation"]
        ems = Ems(microgrid, trees, input_func, params_evaluation)

    ems.set_hour_day_switch(switch_hour)

    microgrid.attribute_to_log("Battery_0", "soc")
    microgrid.attribute_to_log("Charger_0", "soc")
    microgrid.attribute_to_log("Charger_0", "soc_f")
    microgrid.attribute_to_log("Charger_0", "det")

    count = 0

    while microgrid.datetime != microgrid.end_time:
        # for _ in range(24 * 4):
        count += 1
        if count % 100 == 0:
            print(count)
        microgrid.management_system.simulate_step()
        charger_att = microgrid.attributes_hist[ems.charger]
        last_soc = charger_att["soc"][-1]
        last_soc_f = charger_att["soc_f"][-1]
        last_det = charger_att["det"][-1]
        cur_dt = microgrid.utc_datetime
        bat_soc = microgrid.attributes_hist[ems.battery]["soc"][-1]
        if last_det == 1 and (last_soc != last_soc_f):
            print(cur_dt)
            print(last_soc, last_soc_f)
        if bat_soc < 0.99 and cur_dt.hour == 15 and cur_dt.minute == 0:
            print(cur_dt)
            print(bat_soc)

    power_hist = microgrid.power_hist
    kpi_hist = microgrid.reward_hist
    attributes_hist = microgrid.attributes_hist
    print(microgrid.tot_reward.KPIs)

    datetime = attributes_hist[list(attributes_hist.keys())[0]]["datetime"]
    # soc = attributes_hist[list(attributes_hist.keys())[0]]["soc"]
    # soc_charger = attributes_hist[list(attributes_hist.keys())[1]]["soc"]
    # soc_final = attributes_hist[list(attributes_hist.keys())[1]]["soc_f"]

    # Write datetime index in ISO format
    datetime = [d.isoformat() for d in datetime]

    # Write datetime and soc to csv in two columns
    # write_file("soc.csv", datetime, soc)
    # write_file("soc_charger.csv", datetime, soc_charger)
    # write_file("soc_final.csv", datetime, soc_final)

    if plot_graph:
        plot_hist(power_hist, kpi_hist)
        plot_attributes(attributes_hist)
        plt.tight_layout()
        plt.show()
    return microgrid


def write_file(filename, datetime, soc):
    with open(filename, "w") as f:
        f.write("datetime,soc\n")
        for i in range(len(datetime)):
            f.write(f"{datetime[i]},{soc[i]}\n")


def reproduce_batt_fixed_svl(house_num, plot_graph=True):
    config_file = f"data/houses/house_{house_num}/2023-09-08_10:00:00_2023-10-27_16:00:00/house_training_simul.json"
    Ems = RationalSVL
    # Ems = SVLMPCManager
    microgrid = run_house(config_file, Ems, plot_graph, switch_hour=15)
    return microgrid


def print_power_grid(house_num):
    microgrid = reproduce_batt_fixed_svl(house_num, True)
    print(microgrid)
    grid_power = [p.electrical for p in microgrid.power_hist[0]["PublicGrid_0"]]
    pos_power = [p for p in grid_power if p > 0]
    neg_power = [p for p in grid_power if p < 0]

    print(f"Grid offtake: {sum(pos_power)/4}")
    print(f"Grid injection: {sum(neg_power)/4}")


def test_mpc(house_num, forecast_mode="perfect"):
    house_num = 1
    config_file = f"data/houses/house_{house_num}/2023-09-08_10:00:00_2023-10-27_16:00:00/house_training_simul.json"
    switch_hour = 15

    # config_file = switch_hour_config(config_file, switch_hour)
    microgrid = parse_config_file(config_file)
    reward = DayAheadEngie()
    microgrid.set_reward(reward)

    params_house = {"house_num": 1}
    ems = SVLMPCManager(
        microgrid, params_evaluation=params_house, forecast_mode=forecast_mode
    )
    ems.set_hour_day_switch(switch_hour)

    microgrid.attribute_to_log("Battery_0", "soc")

    count = 0

    while microgrid.datetime != microgrid.end_time:
        # for i in range(200):
        count += 1
        if count % 100 == 0:
            print(count)
        microgrid.management_system.simulate_step()

    power_hist = microgrid.power_hist
    kpi_hist = microgrid.reward_hist
    attributes_hist = microgrid.attributes_hist
    print(microgrid.tot_reward.KPIs)

    plot_hist(power_hist, kpi_hist)
    plot_attributes(attributes_hist)
    plt.tight_layout()
    plt.show()


def test_batt_rule_ems(house_num):
    def empty_input_func(empty_params):
        return [[], []], [[], []]

    config_file = f"data/houses/house_{house_num}/2023-09-08_10:00:00_2023-10-27_16:00:00/house_training_simul.json"
    Ems = SVLTreeBattChargRule
    tree_batt = BinaryTreeFixed([0], [], [])
    tree_charg = BinaryTreeFixed([0], [], [])

    trees = [tree_batt, tree_charg]
    tree_params = {
        "trees": trees,
        "input_func": empty_input_func,
        "params_evaluation": {},
    }
    switch_hour = 15
    run_house(config_file, Ems, True, switch_hour=switch_hour, tree_params=tree_params)

    Ems = RationalSVL
    run_house(config_file, Ems, True, switch_hour=switch_hour)


def create_switch_config_houses():
    for house_num in [2, 3, 5]:
        config_file = f"data/houses/house_{house_num}/2023-09-08_10:00:00_2023-10-27_16:00:00/house_training_simul.json"
        switch_hour_config(config_file, 15)


if __name__ == "__main__":
    # for house_num in [1, 2, 3, 5]:
    #    print_power_grid(house_num)
    # print_power_grid(1)
    # test_mpc(1, "model")
    # test_batt_rule_ems(1)
    # reproduce_batt_fixed_svl(house_num)
    # test_mpc(2)
    # test_batt_rule_ems(5)
    # create_switch_config_houses()
    # reproduce_batt_fixed_svl(1, True)
    for house_num in [1, 2, 3, 5]:
        config_file = f"data/houses/house_{house_num}/2024-01-01_0000_2024-04-01_0000/house_train.json"

        run_house(
            config_file,
            NoEMSSVL,
            True,
            switch_hour=15,
        )
