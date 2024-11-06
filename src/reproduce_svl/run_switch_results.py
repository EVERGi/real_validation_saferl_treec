import copy
import sys
import os


filed_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(filed_dir)
sys.path.append(parent_dir)

from reproduce_svl.custom_classes import DayAheadEngie, RationalSVL, SVLTreeBattCharg
from simugrid.simulation.config_parser import parse_config_file
from simugrid.simulation.power import Power
from reproduce_svl.mpc import SVLMPCManager, ModelMPCManager
from simugrid.misc.log_plot_micro import plot_hist, plot_attributes

import matplotlib.pyplot as plt

import numpy as np

# Supress UserWarning and FutureWarning from seaborn
import warnings

warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)


def remove_charger(power_hist, start_ind):
    grid_power = power_hist[0]["PublicGrid_0"]
    charger_power = power_hist[0]["Charger_0"]

    for i, charg_pow in enumerate(charger_power):
        if i >= start_ind:
            grid_power[i] += charg_pow
            charger_power[i] -= charg_pow
    return power_hist


def add_simple_charger(power_hist, new_charger_power, start_ind):
    grid_power = power_hist[0]["PublicGrid_0"]
    charger_power = power_hist[0]["Charger_0"]

    for i, charg_pow in enumerate(new_charger_power):
        power_ind = i + start_ind
        grid_power[power_ind] += Power(charg_pow)
        charger_power[power_ind] -= Power(charg_pow)
    return power_hist


def simple_charger(charger_env_hist, charger_asset):
    session_power = list()
    charger_power = list()
    for i, capa in enumerate(charger_env_hist["capa_0"]):
        if capa != 0:
            soc_i = charger_env_hist["soc_i_0"][i]
            p_max = charger_env_hist["p_max_0"][i]
            soc_f = charger_env_hist["soc_f_0"][i]
            state = {
                "det": 100,
                "size": capa,
                "soc": soc_i,
                "soc_f": soc_f,
                "soc_min": soc_i,
                "max_charge": p_max,
                "max_discharge": 0,
            }
            session_power = calc_charge_session(state, charger_asset)
        if len(session_power) != 0:
            charger_power.append(session_power[0])
            session_power = session_power[1:]
        else:
            charger_power.append(0)

    return charger_power


def calc_charge_session(state, charger_asset):
    soc_i = state["soc"]
    prev_soc = soc_i
    soc_f = state["soc_f"]
    p_max = state["max_charge"]
    time_step = copy.deepcopy(charger_asset.dt_h)
    session_power = list()
    while prev_soc < soc_f:
        prev_soc, average_pow = charger_asset.ev_model(state, p_max, soc_i)
        if len(session_power) == 0:
            last_pow = average_pow
        else:
            num_sess = len(session_power)
            last_pow = (num_sess + 1) * average_pow - np.mean(session_power) * (
                num_sess
            )
        session_power.append(last_pow)

        charger_asset.dt_h += time_step

    charger_asset.dt_h = time_step
    return session_power


def shift_to_simple_charger(power_hist, env_hist, asset_list, start_ind):
    power_hist = copy.deepcopy(power_hist)
    for asset in asset_list:
        if asset.name == "Charger_0":
            charger_asset = asset
            break
    charger_env = {
        "capa_0": env_hist[0]["capa_0"][start_ind:],
        "soc_i_0": env_hist[0]["soc_i_0"][start_ind:],
        "p_max_0": env_hist[0]["p_max_0"][start_ind:],
        "soc_f_0": env_hist[0]["soc_f_0"][start_ind:],
    }
    charger_power = simple_charger(charger_env, charger_asset)
    power_hist = remove_charger(power_hist, start_ind)
    power_hist = add_simple_charger(power_hist, charger_power, start_ind)
    return power_hist


def remove_battery(power_hist, start_ind):
    power_hist = copy.deepcopy(power_hist)
    grid_power = power_hist[0]["PublicGrid_0"]
    if "Battery_0" in power_hist[0].keys():
        battery_power = power_hist[0]["Battery_0"]
    else:
        battery_power = power_hist[1]["Battery_0"]

    for i, batt_pow in enumerate(battery_power):
        if i >= start_ind:
            grid_power[i] += batt_pow
            battery_power[i] -= batt_pow

    return power_hist


def calc_cost_no_ems(microgrid, switch_hour):
    power_hist = microgrid.power_hist
    grid_datetime = power_hist[0]["datetime"]
    start_ind = 0
    for i, dt in enumerate(grid_datetime[::-1]):
        if dt.hour == switch_hour and dt.minute == 0:
            start_ind = len(grid_datetime) - i - 1
            break
    env_hist = microgrid.env_hist

    power_hist = shift_to_simple_charger(
        power_hist, env_hist, microgrid.assets, start_ind
    )
    power_hist = remove_battery(power_hist, start_ind)

    grid_datetime = grid_datetime[start_ind:]
    grid_hist = power_hist[0]["PublicGrid_0"][start_ind:]
    grid_hist = [i.electrical for i in grid_hist]

    grid_power = {"datetime": grid_datetime, "power": grid_hist}

    tarrif_datetime = env_hist[0]["datetime"][start_ind:]
    tarrif_hist = env_hist[0]["day_ahead_price"][start_ind:]
    grid_tariff = {"datetime": tarrif_datetime, "tariff": tarrif_hist}

    return calc_cost(grid_power, grid_tariff)


def calc_cost(grid_power, grid_tariff):
    TRANS_DIS_COST = 0.095
    datetime_grid = grid_power["datetime"]
    datetime_price = grid_tariff["datetime"]
    time_step = datetime_grid[1] - datetime_grid[0]
    time_step_h = time_step.total_seconds() / 3600

    for i, dt_price in enumerate(datetime_price):
        if dt_price == datetime_grid[0]:
            start_ind = i
        elif dt_price == datetime_grid[-1]:
            end_ind = i + 1
            break
    datetime_price = datetime_price[start_ind:end_ind]
    tariff = grid_tariff["tariff"][start_ind:end_ind]

    total_cost = 0
    for i, grid_pow in enumerate(grid_power["power"]):
        grid_energy = grid_pow * time_step_h
        total_cost -= grid_energy * tariff[i]
        if grid_pow > 0:
            distribution_cost = grid_energy * TRANS_DIS_COST
            total_cost -= distribution_cost

    return total_cost


def calc_previous_cost(microgrid, switch_hour):
    power_hist = microgrid.power_hist
    grid_datetime = power_hist[0]["datetime"]
    start_ind = 0
    for i, dt in enumerate(grid_datetime[::-1]):
        if dt.hour == switch_hour and dt.minute == 0:
            start_ind = len(grid_datetime) - i - 1
            break

    grid_datetime = grid_datetime[start_ind:]
    grid_hist = power_hist[0]["PublicGrid_0"][start_ind:]
    grid_hist = [i.electrical for i in grid_hist]

    grid_power = {"datetime": grid_datetime, "power": grid_hist}

    env_hist = microgrid.env_hist
    tarrif_datetime = env_hist[0]["datetime"][start_ind:]
    tarrif_hist = env_hist[0]["day_ahead_price"][start_ind:]
    grid_tariff = {"datetime": tarrif_datetime, "tariff": tarrif_hist}

    return calc_cost(grid_power, grid_tariff)


def runs_switch_house(house_num, plot_graph, switch_hour=15, tree_params=None):
    ems_to_switch = [RationalSVL, SVLMPCManager, ModelMPCManager]
    run_given_switch_house(
        house_num, plot_graph, ems_to_switch, switch_hour, tree_params
    )


def run_same_ems(house_num, plot_graph, Ems, switch_hour=15, tree_params=None):
    ems_to_switch = [Ems]
    run_given_switch_house(
        house_num, plot_graph, ems_to_switch, switch_hour, tree_params
    )


def run_all_same_ems():
    ems_to_switch = [RationalSVL, SVLMPCManager, ModelMPCManager]
    for house in [1, 2, 3]:
        for Ems in ems_to_switch:
            run_same_ems(house, plot_graph=False, Ems=Ems)


def calc_perf_mpc_cost(microgrid, house_num, utc_datetime, current_opex):
    switch_hour = microgrid.management_system.hour_day_switch

    params_evaluation = {"house_num": house_num}
    ems = SVLMPCManager(
        microgrid, params_evaluation=params_evaluation, forecast_mode="perfect"
    )

    ems.set_hour_day_switch(switch_hour)
    ems.current_month_peak = microgrid.tot_reward.cur_peak_power

    while microgrid.utc_datetime != utc_datetime:
        microgrid.management_system.simulate_step()

    opex = microgrid.tot_reward.KPIs["opex"]

    perf_opex = opex - current_opex

    return perf_opex, microgrid


def run_given_switch_house(
    house_num, plot_graph, ems_to_switch, switch_hour, tree_params
):
    # config_file = f"data/houses/house_{house_num}/2023-09-08_10:00:00_2023-10-27_16:00:00/house_training_simul.json"
    config_file = f"data/houses/house_{house_num}/2023-09-08_10:00:00_2023-10-27_16:00:00/house_training_simul_switch_1500.json"
    house_num_to_ind = {1: 0, 2: 1, 3: 2, 5: 3}

    # config_file = switch_hour_config(config_file, switch_hour)

    microgrid = parse_config_file(config_file)
    microgrid.attribute_to_log("Battery_0", "soc")
    microgrid.attribute_to_log("Charger_0", "soc")

    reward = DayAheadEngie()
    microgrid.set_reward(reward)
    prev_microgrid = copy.deepcopy(microgrid)

    first_ems_ind = house_num_to_ind[house_num] % len(ems_to_switch)
    FirstEms = ems_to_switch[first_ems_ind]
    if issubclass(FirstEms, SVLMPCManager):
        params_house = {"house_num": house_num}
        ems = FirstEms(microgrid, params_evaluation=params_house)
    elif issubclass(FirstEms, SVLTreeBattCharg):
        trees = tree_params["trees"]
        input_func = tree_params["input_func"]
        params_evaluation = tree_params["params_evaluation"]
        ems = FirstEms(microgrid, trees, input_func, params_evaluation)
    else:
        ems = FirstEms(microgrid)

    next_ems_ind = (first_ems_ind + 1) % len(ems_to_switch)

    ems.set_hour_day_switch(switch_hour)
    env_keys_to_log = ["day_ahead_price", "soc_i_0", "soc_f_0", "p_max_0", "capa_0"]
    for env_key in env_keys_to_log:
        microgrid.env_to_log(env_key)

    count = 0

    current_opex = 0
    NextEms = FirstEms
    day_counter = 0
    while microgrid.datetime != microgrid.end_time:
        if (
            microgrid.utc_datetime.hour == switch_hour
            and microgrid.utc_datetime.minute == 0
        ):
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
            if ems_worse < 0:
                plot_microgrid(perf_microgrid)
                plot_microgrid(microgrid)
                plt.show()

            # print(f"EMS {NextEms.__name__} improve: {ems_improve}")
            print(f"EMS {NextEms.__name__} worse than perfect: {ems_worse}")
            # Save scores in csv file
            with open("results/scores.csv", "a") as f:
                f.write(f"{house_num},{NextEms.__name__},{ems_worse},{day_counter}\n")

            day_counter += 1

            NextEms = ems_to_switch[next_ems_ind]

            if issubclass(NextEms, SVLMPCManager):
                params_house = {"house_num": house_num}
                ems = NextEms(microgrid, params_evaluation=params_house)
                ems.current_month_peak = reward.cur_peak_power
            elif issubclass(NextEms, SVLTreeBattCharg):
                trees = tree_params["trees"]
                input_func = tree_params["input_func"]
                params_evaluation = tree_params["params_evaluation"]
                ems = NextEms(microgrid, trees, input_func, params_evaluation)
            else:
                ems = NextEms(microgrid)

            ems.set_hour_day_switch(switch_hour)
            next_ems_ind = (next_ems_ind + 1) % len(ems_to_switch)
        # for _ in range(24 * 4):
        count += 1
        if count % 100 == 0:
            print(count)

        microgrid.management_system.simulate_step()

    print(microgrid.tot_reward.KPIs)

    if plot_graph:
        plot_microgrid(microgrid, plot_graph)
        plt.show()

    return microgrid


def plot_microgrid(microgrid):
    power_hist = microgrid.power_hist
    kpi_hist = microgrid.reward_hist
    attributes_hist = microgrid.attributes_hist

    plot_hist(power_hist, kpi_hist)
    plot_attributes(attributes_hist)
    plt.tight_layout()
