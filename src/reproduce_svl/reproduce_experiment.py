import copy
import sys
import os
import pytz
import numpy as np
import json

from simugrid.rewards.reward import Reward
from simugrid.assets import PublicGrid
from simugrid.misc.log_plot_micro import log_micro
import datetime
import pandas as pd
from treec.utils import denormalise_input, normalise_input

filed_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(filed_dir)
sys.path.append(parent_dir)

from reproduce_svl.custom_classes import (
    DayAheadEngie,
    SVLTreeBattCharg,
    RationalSVL,
    SVLTreeBattChargRule,
    RandomEMS,
    RLApproximation,
    bound,
)
from simugrid.simulation.config_parser import parse_config_file
from simugrid.misc.log_plot_micro import plot_hist, plot_attributes
from reproduce_svl.run_switch_results import (
    calc_previous_cost,
    calc_cost_no_ems,
    calc_perf_mpc_cost,
)
from treec.logger import TreeLogger
from training_trees.input_function import (
    reduced_state_input_norm_price,
    reduced_state_no_real_price,
)
from reproduce_svl.mpc import (
    SVLMPCManager,
    ModelMPCManager,
    ExperimentMPC,
    SimulExpMPC,
    PerfectExpMPC,
    BugMPC,
)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm


def get_model_path(house_num, tree_model_dir):

    validation_folder = tree_model_dir + f"house_{house_num}/validation/"
    for folder in os.listdir(validation_folder):
        if os.path.isdir(folder):
            break
    complete_path = os.path.join(validation_folder, folder, "models/")

    return complete_path


class ExperimentReward(Reward):
    def __init__(self):
        list_KPI = ["Tree", "MPC", "RBC", "RL"]
        Reward.__init__(self, list_KPI)

        self.peak_power = {ems: 0 for ems in list_KPI}

        self.day_score = {}

    def calculate_kpi(self):
        microgrid = self.microgrid
        success_status = microgrid.environments[0].env_values["success"].value
        if success_status == "failed":
            return

        time_step_h = microgrid.time_step.total_seconds() / 3600

        for asset in microgrid.assets:
            if isinstance(asset, PublicGrid):
                cur_ems = microgrid.environments[0].env_values["ems_used"].value
                env = asset.parent_node.environment
                kwh_offtake_cost = env.env_values["kwh_offtake_cost"].value
                capacity_tariff = env.env_values["capacity_tariff"].value  # €/kW/month
                exp_capacity_tariff = capacity_tariff * 12 / 30
                year_cost = env.env_values["year_cost"].value
                injection_extra = env.env_values["injection_extra"].value
                offtake_extra = env.env_values["offtake_extra"].value

                grid_energy = asset.power_output.electrical * time_step_h

                day_ahead_price = env.env_values["day_ahead_price"].value

                if grid_energy > 0:
                    day_ahead_price = day_ahead_price + offtake_extra
                    if day_ahead_price > 0:
                        day_ahead_cost = -grid_energy * day_ahead_price * 1.06
                    else:
                        day_ahead_cost = -grid_energy * day_ahead_price
                    trans_dis = -grid_energy * kwh_offtake_cost
                else:
                    day_ahead_price = day_ahead_price + injection_extra
                    day_ahead_cost = -grid_energy * day_ahead_price
                    trans_dis = 0
                cap_tariff = self.calc_capacity_tarriff(
                    asset.power_output.electrical, exp_capacity_tariff
                )
                time_step_prop_year = time_step_h / (365 * 24)
                year_fixed = time_step_prop_year * year_cost

                self.KPIs[cur_ems] += (
                    day_ahead_cost + trans_dis + cap_tariff + year_fixed
                )
        self.update_day_score()

    def calc_capacity_tarriff(self, grid_power, capacity_tariff):
        utc_datetime = self.microgrid.utc_datetime
        time_step = self.microgrid.time_step

        start_month = datetime.datetime(utc_datetime.year, utc_datetime.month, 1, 0, 0)
        start_month = start_month.replace(tzinfo=pytz.utc)

        cur_ems = self.microgrid.environments[0].env_values["ems_used"].value
        # Get the first day of next month

        prev_peak_power = self.peak_power[cur_ems]

        if grid_power < 2.5:
            grid_power = 2.5

        if grid_power > self.peak_power[cur_ems]:
            self.peak_power[cur_ems] = grid_power

        peak_diff = self.peak_power[cur_ems] - prev_peak_power
        return -peak_diff * capacity_tariff

    def update_day_score(self):
        microgrid = self.microgrid
        cur_ems = microgrid.environments[0].env_values["ems_used"].value
        cur_datetime = microgrid.datetime.replace(tzinfo=None)
        start_day = cur_datetime.replace(hour=15, minute=0)
        if cur_datetime < start_day:
            start_day = start_day - datetime.timedelta(days=1)

        start_day_minus_t = start_day - microgrid.time_step

        start_day_score = 0

        for i, dt in enumerate(microgrid.reward_hist["datetime"]):
            dt = dt.replace(tzinfo=None)

            if dt == start_day_minus_t:
                start_day_score = microgrid.reward_hist[cur_ems][i]
                break

        day_score = self.KPIs[cur_ems] - start_day_score
        self.day_score[start_day] = [day_score, cur_ems]


class SeptemberTree(SVLTreeBattChargRule):
    def __init__(self, microgrid, house_num):
        tree_model_dir = f"training_trees/tree_models_september_train/"
        trees = TreeLogger.get_best_trees(get_model_path(house_num, tree_model_dir))
        input_func = reduced_state_input_norm_price
        super().__init__(microgrid, trees, input_func, None)


class ThreeMonthsTree(SVLTreeBattChargRule):
    def __init__(
        self,
        microgrid,
        house_num,
    ):
        tree_model_dir = f"training_trees/tree_models_three_months_train/"
        trees = TreeLogger.get_best_trees(get_model_path(house_num, tree_model_dir))
        input_func = reduced_state_input_norm_price
        super().__init__(microgrid, trees, input_func, None)


def calc_real_costs(microgrid, simul_calc=True):
    # TODO add taxes add injection extra cost
    env_values = microgrid.environments[0].env_values
    start_time = microgrid.start_time
    end_time = microgrid.end_time
    cur_time = start_time
    time_step = microgrid.time_step
    time_step_h = time_step.total_seconds() / 3600

    day_ahead_price = env_values["day_ahead_price"]
    public_grid_import = env_values["PublicGrid_0_import"]
    public_grid_export = env_values["PublicGrid_0_export"]
    status = env_values["success"]
    ems_used = env_values["ems_used"]
    cur_ems = None

    price_per_ems_per_day = dict()
    capacity_ems = dict()

    if simul_calc:
        public_grid_electrical = [
            p.electrical for p in microgrid.power_hist[0]["PublicGrid_0"]
        ]

    public_grid_values = list()
    counter = -1
    while cur_time != end_time:
        counter += 1
        cur_status = status.get_forecast(cur_time, cur_time + time_step)["values"][0]
        if cur_status == "failed":
            cur_time += microgrid.time_step
            continue

        env_ems = ems_used.get_forecast(cur_time, cur_time + time_step)["values"][0]
        if env_ems != cur_ems:
            cur_ems = env_ems
            if cur_ems not in price_per_ems_per_day:
                price_per_ems_per_day[cur_ems] = {
                    "datetime": [],
                    "cost": [],
                    "cost_sum_grid": [],
                    "day_ahead_cost": [],
                    "offtake_cost": [],
                    "yearly_cost": [],
                    "capacity_cost": [],
                    "real_import": [],
                    "real_export": [],
                    "grid_peak": [],
                    "simulated_power": [],
                    "real_power": [],
                }

        day_ahead_price_value = day_ahead_price.get_forecast(
            cur_time, cur_time + time_step
        )["values"][0]
        public_grid_import_value = public_grid_import.get_forecast(
            cur_time, cur_time + time_step
        )["values"][0]
        public_grid_export_value = public_grid_export.get_forecast(
            cur_time, cur_time + time_step
        )["values"][0]

        cur_power = (public_grid_import_value - public_grid_export_value) / time_step_h
        public_grid_values.append(cur_power)

        # Only take into account import for capacity tariff

        cur_max_cap = capacity_ems.get(cur_ems, None)

        sum_day, sum_offtake, sum_capacity, sum_yearly = calc_cost_import_export(
            microgrid,
            public_grid_import_value,
            public_grid_export_value,
            day_ahead_price_value,
            cur_max_cap,
            sum_grid=True,
        )
        cost_sum_grid = sum_day + sum_offtake + sum_capacity + sum_yearly

        day_ahead_cost, offtake_cost, capacity_cost, yearly_cost = (
            calc_cost_import_export(
                microgrid,
                public_grid_import_value,
                public_grid_export_value,
                day_ahead_price_value,
                cur_max_cap,
            )
        )
        total_cost = day_ahead_cost + offtake_cost + capacity_cost + yearly_cost

        cur_grid_peak = public_grid_import_value / time_step_h

        if cur_ems not in capacity_ems:
            capacity_ems[cur_ems] = 2.5

        if cur_grid_peak > capacity_ems[cur_ems]:
            capacity_ems[cur_ems] = cur_grid_peak

        price_per_ems_per_day[cur_ems]["datetime"].append(cur_time)
        price_per_ems_per_day[cur_ems]["cost"].append(total_cost)
        price_per_ems_per_day[cur_ems]["cost_sum_grid"].append(cost_sum_grid)
        price_per_ems_per_day[cur_ems]["capacity_cost"].append(capacity_cost)
        price_per_ems_per_day[cur_ems]["day_ahead_cost"].append(day_ahead_cost)
        price_per_ems_per_day[cur_ems]["offtake_cost"].append(offtake_cost)
        price_per_ems_per_day[cur_ems]["yearly_cost"].append(yearly_cost)
        price_per_ems_per_day[cur_ems]["real_import"].append(
            public_grid_import_value / time_step_h
        )
        price_per_ems_per_day[cur_ems]["real_export"].append(
            -public_grid_export_value / time_step_h
        )

        if simul_calc:
            simul_power = public_grid_electrical[counter]
            price_per_ems_per_day[cur_ems]["simulated_power"].append(simul_power)
        price_per_ems_per_day[cur_ems]["real_power"].append(cur_power)
        cur_time += microgrid.time_step

    grid_import = public_grid_import.get_forecast(start_time, end_time)["values"]
    grid_export = public_grid_export.get_forecast(start_time, end_time)["values"]
    grid_import = [e / time_step_h for e in grid_import]
    grid_export = [-e / time_step_h for e in grid_export]

    TIMESTEPS_MONTH = 30 * 24 * 4
    for ems in price_per_ems_per_day.keys():
        cost = price_per_ems_per_day[ems]["cost"]
        cost_sum_grid = price_per_ems_per_day[ems]["cost_sum_grid"]
        capacity_cost = price_per_ems_per_day[ems]["capacity_cost"]
        timesteps = len(price_per_ems_per_day[ems]["datetime"])
        timsteps_prop = timesteps / TIMESTEPS_MONTH
        price_per_ems_per_day[ems]["norm_capacity"] = [
            timsteps_prop * c for c in capacity_cost
        ]
        price_per_ems_per_day[ems]["norm_price"] = [
            cost[i] + (timsteps_prop - 1) * capacity_cost[i] for i in range(timesteps)
        ]
        price_per_ems_per_day[ems]["norm_price_sum_grid"] = [
            cost_sum_grid[i] + (timsteps_prop - 1) * capacity_cost[i]
            for i in range(timesteps)
        ]
        if simul_calc:
            plt.figure()

            plt.plot(
                price_per_ems_per_day[ems]["datetime"],
                price_per_ems_per_day[ems]["real_power"],
                label="Real",
            )

            plt.plot(
                price_per_ems_per_day[ems]["datetime"],
                price_per_ems_per_day[ems]["simulated_power"],
                label="Simulation",
            )
            plt.legend()
            plt.title(ems)

    return price_per_ems_per_day


def calc_real_costs_logs(log_folder):
    houses = [1, 2, 3, 5]
    emss = ["RL", "RBC", "Tree", "MPC"]
    elec_tariffs = json.load(open(f"{log_folder}elec_tariffs.json", "r"))

    real_costs = {ems: [] for ems in emss}
    for house_num in houses:
        for ems in emss:
            house_ems_file = f"{log_folder}house_{house_num}_{ems}_grid.csv"

    return real_costs


def calc_cost_import_export(
    microgrid,
    public_grid_import_value,
    public_grid_export_value,
    day_ahead_price_value,
    cur_max_cap,
    sum_grid=False,
):
    time_step_h = microgrid.time_step.total_seconds() / 3600
    env_values = microgrid.environments[0].env_values
    capacity_tariff = env_values["capacity_tariff"].value
    kwh_offtake_cost = env_values["kwh_offtake_cost"].value
    year_cost = env_values["year_cost"].value
    injection_extra = env_values["injection_extra"].value
    offtake_extra = env_values["offtake_extra"].value

    if cur_max_cap is None:
        cur_max_cap = 2.5
        capacity_diff = 2.5
        first_step = True
    else:
        capacity_diff = 0
        first_step = False

    cur_grid_peak = public_grid_import_value / time_step_h

    grid_sum_energy = public_grid_import_value - public_grid_export_value

    # if sum_grid:
    #    cur_grid_peak = cur_power
    # else:
    cur_grid_peak = public_grid_import_value / time_step_h

    if cur_grid_peak > cur_max_cap:
        if not first_step:
            capacity_diff = cur_grid_peak - cur_max_cap
        else:
            capacity_diff = cur_grid_peak

    if sum_grid:
        public_grid_import_value = max(grid_sum_energy, 0)
        public_grid_export_value = -min(grid_sum_energy, 0)

    if day_ahead_price_value > 0:
        day_ahead_cost = (
            -public_grid_import_value * (day_ahead_price_value + offtake_extra) * 1.06
        )
    else:
        day_ahead_cost = -public_grid_import_value * (
            day_ahead_price_value + offtake_extra
        )

    day_ahead_cost += (
        day_ahead_price_value + injection_extra
    ) * public_grid_export_value

    offtake_cost = -kwh_offtake_cost * public_grid_import_value
    capacity_cost = -capacity_tariff * capacity_diff
    yearly_cost = -year_cost / 365 / 24 * time_step_h

    return day_ahead_cost, offtake_cost, capacity_cost, yearly_cost


def run_switch_from_env_house(
    house_num, date_range, plot_graph, env_to_ems, log_folder=None, use_cons_data=False
):
    switch_hour = 15
    config_file = (
        f"data/houses/house_{house_num}/{date_range}/house_batt_not_fixed.json"
    )

    if use_cons_data:
        # Read config_file and replace "./environment/cons.csv" with "./../../common_env/SFH19_2023_2024_15min_original.csv"
        with open(config_file, "r") as f:
            file_content = f.read()
        file_content = file_content.replace(
            "./environment/cons.csv",
            "./../../common_env/SFH19_2023_2024_15min_original.csv",
        )
        original_cons_conf = config_file.replace(
            "house_batt_not_fixed", "house_original_cons"
        )
        with open(original_cons_conf, "w+") as f:
            f.write(file_content)
        config_file = original_cons_conf

    microgrid = parse_config_file(config_file)
    reward = ExperimentReward()
    microgrid.set_reward(reward)

    env_keys_to_log = [
        "day_ahead_price",
        "soc_i_0",
        "soc_f_0",
        "p_max_0",
        "capa_0",
        "PublicGrid_0_export",
        "PublicGrid_0_import",
        "ems_used",
        "success",
    ]
    for env_key in env_keys_to_log:
        microgrid.env_to_log(env_key)

    microgrid.attribute_to_log("Battery_0", "soc")
    # count = 0

    current_opex = 0
    day_counter = 0

    ems_used_value = microgrid.environments[0].env_values["ems_used"]

    EmsClass = env_to_ems[ems_used_value.value]

    if issubclass(EmsClass, SVLMPCManager):
        params_house = {"house_num": house_num, "ems_set": ems_used_value.value}
        ems = EmsClass(microgrid, params_evaluation=params_house)
    elif issubclass(EmsClass, SVLTreeBattCharg):
        ems = EmsClass(microgrid, house_num)
    else:
        ems = EmsClass(microgrid)
    ems.set_hour_day_switch(switch_hour)
    prev_microgrid = copy.deepcopy(microgrid)
    prev_ems_used = ems_used_value.value

    # Calculate number of time_steps from microgrid.datetime to microgrid.end_time
    num_time_steps = int(
        (microgrid.end_time - microgrid.datetime).total_seconds()
        / microgrid.time_step.total_seconds()
    )
    # while microgrid.datetime != microgrid.end_time:
    for _ in tqdm(range(num_time_steps)):

        NewEmsClass = env_to_ems[ems_used_value.value]
        ems_switch = ems_used_value.value != prev_ems_used
        if ems_switch:
            prev_ems_used = ems_used_value.value
            calc_cost = False
            if calc_cost:
                cumul_opex = microgrid.tot_reward.KPIs["opex"]
                ems_opex = cumul_opex - current_opex
                perf_ems_opex, perf_microgrid = calc_perf_mpc_cost(
                    prev_microgrid, house_num, microgrid.utc_datetime, current_opex
                )

                current_opex = cumul_opex

                calc_opex = calc_previous_cost(microgrid, switch_hour)

                no_ems_opex = calc_cost_no_ems(microgrid, switch_hour)

                prev_microgrid = copy.deepcopy(microgrid)

                ems_worse = perf_ems_opex - ems_opex

                print(f"EMS {EmsClass.__name__} improve: {ems_worse}")

                # Save scores in csv file
                with open("results/scores.csv", "a") as f:
                    f.write(
                        f"{house_num},{EmsClass.__name__},{ems_worse},{day_counter}\n"
                    )

                day_counter += 1

            EmsClass = NewEmsClass
            if issubclass(EmsClass, SVLMPCManager):
                params_house = {"house_num": house_num, "ems_set": ems_used_value.value}
                ems = EmsClass(microgrid, params_evaluation=params_house)
            elif issubclass(EmsClass, SVLTreeBattCharg):
                ems = EmsClass(microgrid, house_num)
            else:
                ems = EmsClass(microgrid)

            ems.set_hour_day_switch(switch_hour)
        # for _ in range(24 * 4):
        # count += 1
        # if count % 100 == 0:
        #    print(count)
        #    if count == 600:
        #        print(count)
        microgrid.management_system.simulate_step()

    power_hist = microgrid.power_hist
    kpi_hist = microgrid.reward_hist
    attributes_hist = microgrid.attributes_hist
    # print("Simulation costs:")
    # print(microgrid.tot_reward.KPIs)

    real_costs = calc_real_costs(microgrid)
    # print("Real costs:")
    # for ems in real_costs.keys():

    #    cost_sum = sum(real_costs[ems]["cost"])
    #    print(f"{ems}: {cost_sum}")
    # print("Norm costs:")
    # for ems in real_costs.keys():
    #    cost_sum = sum(real_costs[ems]["norm_price"])
    #    print(f"{ems}: {cost_sum}")

    if log_folder is not None:
        log_results(microgrid, log_folder, house_num)

    if plot_graph:
        plot_hist(power_hist, kpi_hist)
        plot_attributes(attributes_hist)
        plt.tight_layout()
        plt.show()

    return real_costs


def log_results(microgrid, log_folder, house_num):
    start_t = microgrid.start_time
    end_t = microgrid.end_time

    house_folder = f"{log_folder}house_{house_num}/"
    os.makedirs(house_folder, exist_ok=True)
    log_micro(microgrid, house_folder)

    emss = ["RL", "RBC", "Tree", "MPC"]
    for ems in emss:
        house_ems_file = f"{log_folder}house_{house_num}_{ems}_grid.csv"
        with open(house_ems_file, "w") as f:
            f.write("datetime,grid_power,grid_import,grid_export\n")

    env_values = microgrid.environments[0].env_values

    grid_power = microgrid.power_hist[0]["PublicGrid_0"]
    grid_power = [p.electrical for p in grid_power]

    datetimes_power = microgrid.power_hist[0]["datetime"]
    datetimes_power = [d.replace(tzinfo=None) for d in datetimes_power]

    env_info = {
        "success": None,
        "ems_used": None,
        "PublicGrid_0_import": None,
        "PublicGrid_0_export": None,
    }
    for key in env_info.keys():
        all_env_values = env_values[key].get_forecast(start_t, end_t)
        env_info[key] = all_env_values["values"]
        datetimes_env = all_env_values["datetime"]

    datetimes_env = [d.replace(tzinfo=None) for d in datetimes_env]

    for i, dt in enumerate(datetimes_env):
        success = env_info["success"][i]
        if success == "failed":
            continue
        ems = env_info["ems_used"][i]
        ts_h = microgrid.time_step.total_seconds() / 3600
        grid_import = env_info["PublicGrid_0_import"][i] / ts_h
        grid_export = env_info["PublicGrid_0_export"][i] / ts_h

        grid_index = datetimes_power.index(dt)
        grid_power_val = grid_power[grid_index]

        dt_str = dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        house_ems_file = f"{log_folder}house_{house_num}_{ems}_grid.csv"
        with open(house_ems_file, "a") as f:
            f.write(f"{dt_str},{grid_power_val},{grid_import},{grid_export}\n")

    # Log tariffs
    tariffs_const_to_log = [
        "capacity_tariff",
        "kwh_offtake_cost",
        "year_cost",
        "injection_extra",
        "offtake_extra",
    ]
    elec_tariffs = {tariff: None for tariff in tariffs_const_to_log}

    for tariff in tariffs_const_to_log:
        elec_tariffs[tariff] = env_values[tariff].value

    day_ahead_forec = env_values["day_ahead_price"].get_forecast(start_t, end_t)
    day_ahead_forec["datetime"] = [
        d.strftime("%Y-%m-%dT%H:%M:%SZ") for d in day_ahead_forec["datetime"]
    ]

    elec_tariffs["day_ahead_price"] = day_ahead_forec

    elec_tariffs["time_step_h"] = ts_h
    json.dump(elec_tariffs, open(f"{log_folder}elec_tariffs.json", "w"))


def run_experiment(
    latest_end_dt,
    houses=[1, 2, 3, 5],
    log_folder=None,
    env_to_ems=None,
    use_cons_data=False,
):
    date_range = f"2024-04-08_1500_{latest_end_dt}"

    if env_to_ems is None:
        env_to_ems = {
            "RBC": RationalSVL,
            "MPC": SimulExpMPC,
            "RL": RLApproximation,
            "Tree": ThreeMonthsTree,
        }
    plot_graph = False
    norm_cost = {"Tree": 0, "MPC": 0, "RBC": 0, "RL": 0}
    for house_num in houses:
        if house_num == 5:
            print(f"Running house 4:")
        else:
            print(f"Running house {house_num}:")
        real_costs = run_switch_from_env_house(
            house_num, date_range, plot_graph, env_to_ems, log_folder, use_cons_data
        )
        for ems in real_costs.keys():
            norm_cost[ems] += sum(real_costs[ems]["norm_price"])

    # print("Total real costs:")
    # for ems in norm_cost.keys():
    #     print(f"{ems}: {norm_cost[ems]}")


def calc_all_half_results(run_date_format, calc_half=True):
    houses = [1, 2, 3, 5]

    norm_cost = {"Tree": 0, "MPC": 0, "RBC": 0, "RL": 0}
    norm_sum_grid = {"Tree": 0, "MPC": 0, "RBC": 0, "RL": 0}

    ems_order = ["RL", "RBC", "Tree", "MPC"]
    cost_types_dict = {
        "norm_capacity": "Capacity",
        "day_ahead_cost": "Day ahead",
        "offtake_cost": "Offtake",
        "yearly_cost": "Yearly",
    }
    stacked_bar_plot = {
        "norm_capacity": [0 for _ in ems_order],
        "day_ahead_cost": [0 for _ in ems_order],
        "offtake_cost": [0 for _ in ems_order],
        "yearly_cost": [0 for _ in ems_order],
    }

    cumul_plots = {
        1: {i: [] for i in ems_order},
        2: {i: [] for i in ems_order},
        3: {i: [] for i in ems_order},
        5: {i: [] for i in ems_order},
    }

    for house_num in houses:
        config_file = (
            f"data/houses/house_{house_num}/{run_date_format}/house_batt_not_fixed.json"
        )

        microgrid = parse_config_file(config_file)
        if calc_half:
            microgrid.end_time = microgrid.end_time.replace(
                year=2024, month=5, day=18, hour=15, minute=0
            )
        real_costs = calc_real_costs(microgrid, simul_calc=False)

        env_values = microgrid.environments[0].env_values
        capacity_tariff = env_values["capacity_tariff"].value

        for ems in real_costs.keys():
            norm_cost[ems] += sum(real_costs[ems]["norm_price"])
            norm_sum_grid[ems] += sum(real_costs[ems]["norm_price_sum_grid"])

            cumul_norm_price = []

            for i in range(len(real_costs[ems]["norm_price"])):
                cumul_norm_price.append(-sum(real_costs[ems]["norm_price"][: i + 1]))

            cumul_plots[house_num][ems].append(cumul_norm_price)

            cumul_cap = []
            for i in range(len(real_costs[ems]["norm_capacity"])):
                TIMESTEPS_MONTH = 30 * 24 * 4.0
                timesteps = len(real_costs[ems]["datetime"])
                timsteps_prop = timesteps / TIMESTEPS_MONTH
                cap_val = -sum(real_costs[ems]["norm_capacity"][: i + 1])
                cap_val = cap_val / capacity_tariff / timsteps_prop
                cumul_cap.append(cap_val)

            cumul_plots[house_num][ems].append(cumul_cap)

            for cost_type in cost_types_dict.keys():
                sum_cost = sum(real_costs[ems][cost_type])
                stacked_bar_plot[cost_type][ems_order.index(ems)] -= sum_cost

    print("Total real costs:")
    for ems in norm_cost.keys():
        print(f"{ems}: {norm_cost[ems]}")

    print("Total sum grid costs:")
    for ems in norm_cost.keys():
        print(f"{ems} sum grid: {norm_sum_grid[ems]}")

    plt.figure()
    # for cost_type in cost_types:
    #    plt.bar(ems_order, stacked_bar_plot[cost_type], label=cost_type)

    bottom = np.array([0.0 for _ in ems_order])
    for boolean, weight_count in stacked_bar_plot.items():
        width = 0.5
        p = plt.bar(
            ems_order,
            weight_count,
            width,
            label=cost_types_dict[boolean],
            bottom=bottom,
        )
        bottom += np.array(weight_count)
    plt.legend()
    plt.title("Costs per EMS")

    for i in [0, 1]:
        for house_num in cumul_plots.keys():
            plt.figure()
            for ems in cumul_plots[house_num].keys():
                plt.plot(
                    cumul_plots[house_num][ems][i],
                    label=f"{ems}",
                )
            plt.title(f"House {house_num}")
            plt.xlabel("15 min time-steps")
            if i == 0:
                plt.ylabel("Cumulative total cost (€)")
            else:
                plt.ylabel("Electricity peak (kW)")
            plt.legend()
    plt.show()


def sum_hist_cost(hist_cost, remove_capacity=False):
    tot_cost = None
    if remove_capacity:
        costs = ["day_ahead", "offtake", "constant"]
    else:
        costs = ["day_ahead", "offtake", "capacity", "constant"]
    for cost in costs:
        cost_hist = hist_cost[f"{cost}_hist"]
        if tot_cost is None:
            tot_cost = -np.array(cost_hist)
        else:
            tot_cost += -np.array(cost_hist)
    return tot_cost


def sum_to_non_cumul(tot_cost):
    non_cumul = []
    num_days = int(tot_cost.shape[0] / 4)
    for i in range(4):
        non_cumul.append(tot_cost[i * num_days])
        for j in range(1, num_days):
            non_cumul.append(
                tot_cost[i * num_days + j] - tot_cost[i * num_days + j - 1]
            )

    return np.array(non_cumul)


def make_results_data_cons():

    download = False
    houses = [1, 2, 3, 5]
    date_vm = "2024-06-17 15:00:00"
    run_date_format = date_vm.replace(" ", "_").replace(":", "")[:-2]

    use_cons_data = True

    log_folder = "results/experiment_data_cons/"

    get_and_run_latest_experiment(
        run_date_format, download, houses, log_folder, use_cons_data=use_cons_data
    )

    log_folder = "results/perfect_mpc_data_cons/"

    env_to_ems = {
        "RBC": PerfectExpMPC,
        "MPC": PerfectExpMPC,
        "RL": PerfectExpMPC,
        "Tree": PerfectExpMPC,
    }
    get_and_run_latest_experiment(
        run_date_format,
        download,
        houses,
        log_folder,
        env_to_ems,
        use_cons_data=use_cons_data,
    )

    log_folder = "results/bug_mpc_data_cons/"

    env_to_ems = {
        "RBC": RationalSVL,
        "MPC": BugMPC,
        "RL": RLApproximation,
        "Tree": ThreeMonthsTree,
    }
    get_and_run_latest_experiment(
        run_date_format,
        download,
        houses,
        log_folder,
        env_to_ems,
        use_cons_data=use_cons_data,
    )


def make_bug_results():
    download = False
    houses = [1, 2, 3, 5]
    date_vm = "2024-06-17 15:00:00"
    run_date_format = date_vm.replace(" ", "_").replace(":", "")[:-2]
    env_to_ems = {
        "RBC": RationalSVL,
        "MPC": BugMPC,
        "RL": RLApproximation,
        "Tree": ThreeMonthsTree,
    }
    log_folder = "results/bug_mpc_new_start_day_8_house_1/"
    get_and_run_latest_experiment(
        run_date_format,
        download,
        houses,
        log_folder,
        env_to_ems,
    )


class ThreeMonthsBattRBC(ThreeMonthsTree):
    def get_actions(self):
        actions, leaf_indexes = SVLTreeBattCharg.get_actions(self)
        charger_action = actions[1]

        battery = self.battery
        min_power_batt = -battery.max_consumption_power
        max_power_batt = battery.max_production_power
        FIXED_ACTION = 0
        batt_action = FIXED_ACTION

        RULE_LIMIT = 0.1
        if batt_action < RULE_LIMIT:
            consumer = self.consumer
            consumer_power = consumer.power_limit_low.electrical
            pv = self.pv
            pv_power = pv.power_limit_high.electrical
            charger_power = bound(
                self.charger.power_limit_low.electrical,
                self.charger.power_limit_high.electrical,
                charger_action,
            )

            batt_power = -(consumer_power + pv_power + charger_power)
            actions[0] = batt_power
        else:
            batt_action = (batt_action - RULE_LIMIT) / (1 - RULE_LIMIT)
            actions[0] = denormalise_input(batt_action, min_power_batt, max_power_batt)

        return actions, leaf_indexes


def get_results_house_3_tree():
    houses = [3]
    env_to_ems = {
        "RBC": RationalSVL,
        "MPC": RationalSVL,
        "RL": RationalSVL,
        "Tree": ThreeMonthsTree,
    }
    download = False

    date_vm = "2024-06-17 15:00:00"
    run_date_format = date_vm.replace(" ", "_").replace(":", "")[:-2]

    log_folder = None

    get_and_run_latest_experiment(
        run_date_format, download, houses, log_folder, env_to_ems=env_to_ems
    )

    log_folder = "results/replace_batt_rbc_house3/"

    env_to_ems = {
        "RBC": RationalSVL,
        "MPC": RationalSVL,
        "RL": RationalSVL,
        "Tree": ThreeMonthsBattRBC,
    }
    get_and_run_latest_experiment(
        run_date_format,
        download,
        houses,
        log_folder,
        env_to_ems,
    )
    plt.show()


if __name__ == "__main__":

    get_results_house_3_tree()
