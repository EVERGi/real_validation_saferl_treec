import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os
import sys

filed_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(filed_dir)
sys.path.append(parent_dir)


from simugrid.simulation.config_parser import parse_config_file
from reproduce_svl.custom_classes import DayAheadEngie
from reproduce_svl.mpc import NaiveMPCManager, SVLMPCManager
from simugrid.misc.log_plot_micro import plot_hist, plot_attributes


def plot_ev(file_ev_schedule):
    plt.figure()
    ev_schedule = pd.read_csv(file_ev_schedule)

    # Pandas df to dictionary
    ev_schedule = ev_schedule.to_dict(orient="list")
    # datetime list to datetime object in ISO 8601 format
    ev_schedule["datetime"] = [
        datetime.datetime.fromisoformat(d.replace("Z", "+00:00"))
        for d in ev_schedule["datetime"]
    ]
    ev_datetime = ev_schedule["datetime"]
    det_0 = ev_schedule["det_0 (/)"]

    car_connected = list()
    connected_time_left = list()
    count_connected = 0
    for det_val in det_0:
        if det_val > 0:
            count_connected = round(det_val * 4)
        if count_connected > 0:
            connected_time_left.append(count_connected / 4)
        else:
            connected_time_left.append(0)
        count_connected -= 1
        if count_connected < 0:
            car_connected.append(0)
        else:
            car_connected.append(1)
    plt.figure()
    plt.plot(ev_datetime, connected_time_left)

    connected_quarter_hour = [[] for _ in range(0, 96)]
    time_left_quarter_hour = [[] for _ in range(0, 96)]
    for i, connected_quart in enumerate(car_connected):
        connected_quarter_hour[i % 96].append(connected_quart)
        time_left_quarter_hour[i % 96].append(connected_time_left[i])

    x_axis = [ev_datetime[i] for i in range(0, 96)]
    x_axis = [i.strftime("%H:%M") for i in x_axis]
    x_axis = [value if i % 8 == 0 else "" for i, value in enumerate(x_axis)]

    average_connected = [
        np.average(connected_quart) * 100 for connected_quart in connected_quarter_hour
    ]
    average_time_left = [
        np.average(time_left_quart) for time_left_quart in time_left_quarter_hour
    ]
    plt.title("Percentage of time connected")
    plt.ylabel("Percentage of time connected (%)")
    plt.plot(average_connected)

    plt.xticks(range(0, 96), x_axis)

    plt.xlim(0, 95)
    plt.ylim(0, 100)
    plt.tight_layout()

    plt.figure()
    plt.title("Average time to end of charging session")
    plt.ylabel("Average time left (hours)")
    plt.plot(average_time_left)
    plt.xticks(range(0, 96), x_axis)
    plt.xlim(0, 95)
    plt.tight_layout()

    # plt.plot(ev_datetime, car_connected)


def jan_mar_schedule():
    ev_schedule = "data/houses/common_env/ev_schedule_jan_mar_2023_2024.csv"
    plot_ev(ev_schedule)


class PerfectNoSwitchMPC(SVLMPCManager):
    def __init__(self, microgrid, params_evaluation, horizon=24):
        super().__init__(microgrid, params_evaluation, forecast_mode="perfect")
        self.horizon = horizon

    def calculate_horizon(self):
        return self.horizon


def write_file(filename, datetime, soc):
    with open(filename, "w") as f:
        f.write("datetime,soc\n")
        for i in range(len(datetime)):
            f.write(f"{datetime[i]},{soc[i]}\n")


def get_MPC_no_switch_results():
    horizon = 24
    for house_num in [5]:  # 1, 2, 3, 5]:
        config_file = f"data/houses/house_{house_num}/2024-01-01_0000_2024-04-01_0000/house_no_switch.json"
        microgrid = parse_config_file(config_file)
        reward = DayAheadEngie()
        microgrid.set_reward(reward)
        params_evaluation = {"house_num": house_num}

        ems = PerfectNoSwitchMPC(microgrid, params_evaluation, horizon)

        microgrid.attribute_to_log("Battery_0", "soc")

        i = 0
        while microgrid.datetime != microgrid.end_time:
            microgrid.management_system.simulate_step()
            i += 1
            if i % 100 == 0:
                print(i)

        power_hist = microgrid.power_hist
        kpi_hist = microgrid.reward_hist
        attributes_hist = microgrid.attributes_hist

        datetime = attributes_hist[list(attributes_hist.keys())[0]]["datetime"]
        soc = attributes_hist[list(attributes_hist.keys())[0]]["soc"]
        write_file(
            f"data/results/house_{house_num}_MPC_perfect_forec_{horizon}h_soc.csv",
            datetime,
            soc,
        )

        print(microgrid.tot_reward.KPIs)

        plot_hist(power_hist, kpi_hist)
        plot_attributes(attributes_hist)
        plt.tight_layout()
        plt.show()


def plot_MPC_no_switch_results():
    horizon = 24
    plt.figure()
    soc_quarter_hour = [[] for _ in range(0, 96)]

    for house_num in [1, 2, 3, 5]:
        file_soc = (
            f"data/results/house_{house_num}_MPC_perfect_forec_{horizon}h_soc.csv"
        )

        soc_df = pd.read_csv(file_soc)
        soc = soc_df.to_dict(orient="list")
        # datetime list to datetime object
        soc["datetime"] = [
            datetime.datetime.fromisoformat(d.replace("Z", "+00:00"))
            for d in soc["datetime"]
        ]
        soc_datetime = soc["datetime"]
        soc = soc["soc"]

        for i, soc_quart in enumerate(soc):
            soc_quarter_hour[i % 96].append(soc_quart)

    x_axis = [soc_datetime[i] for i in range(0, 96)]
    x_axis = [i.strftime("%H:%M") for i in x_axis]
    x_axis = [value if i % 8 == 0 else "" for i, value in enumerate(x_axis)]

    average_soc = [np.average(soc_quart) for soc_quart in soc_quarter_hour]
    std_soc = [np.std(soc_quart) for soc_quart in soc_quarter_hour]
    low_bound = [average_soc[i] - std_soc[i] for i in range(0, 96)]
    high_bound = [average_soc[i] + std_soc[i] for i in range(0, 96)]

    plt.boxplot(soc_quarter_hour, labels=x_axis)

    plt.title("State of charge per quarter hour")

    plt.ylabel("State of charge (-)")
    plt.show()


if __name__ == "__main__":
    # get_MPC_no_switch_results()
    # jan_mar_schedule()
    plot_MPC_no_switch_results()
