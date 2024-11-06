import pandas as pd
import datetime
import numpy as np

import matplotlib.pyplot as plt


def analyse_ev_for_forecasting():
    plot_ev_arrival()
    plt.show()


def analyse_common_env(house_num=1):
    analysis_folder = "data/houses/common_env/analysis_files/"
    file_ev_schedule = "data/houses/common_env/first_ev_schedule.csv"
    file_soc = analysis_folder + f"soc_{house_num}.csv"
    file_ev_soc = analysis_folder + f"soc_charger_{house_num}.csv"
    file_ev_soc_f = analysis_folder + f"soc_final_{house_num}.csv"
    plot_ev_soc(file_ev_soc, file_ev_soc_f)

    plot_ev(file_ev_schedule)
    plot_soc(file_soc)

    plt.show()


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


def plot_ev_soc(file_ev_soc, file_ev_soc_f):
    avg_full, x_axis = get_average_idle(file_ev_soc, file_ev_soc_f, "full")
    avg_start, x_axis = get_average_idle(file_ev_soc, file_ev_soc_f, "start")
    avg_connected, x_axis = get_average_idle(file_ev_soc, file_ev_soc_f, "connected")
    plt.figure()
    # plt.plot(avg_full, label="connected or at final soc")
    plt.plot(avg_start, label="connected but not at init soc")
    plt.plot(avg_connected, label="connected")

    plt.title("Percentage of time idle")
    plt.ylabel("Percentage of time car is in state of legend (%)")
    plt.xticks(range(0, 96), x_axis)
    plt.xlim(0, 95)
    plt.ylim(0, 100)
    plt.legend()


def plot_ev_arrival():
    start_dt = datetime.datetime(2023, 9, 8, 10, 0, 0)
    end_dt = datetime.datetime(2023, 10, 27, 16, 0, 0)
    file_ev_schedule = "data/houses/common_env/first_ev_schedule.csv"
    ev_schedule_df = pd.read_csv(file_ev_schedule)
    ev_schedule = ev_schedule_df.to_dict(orient="list")
    ev_schedule["datetime"] = [
        datetime.datetime.fromisoformat(d.replace("Z", "+00:00")).replace(tzinfo=None)
        for d in ev_schedule["datetime"]
    ]

    dets = ev_schedule["det_0 (/)"]
    detentions = list()
    soc_i = list()
    soc_f = list()
    hour_of_day = list()

    for i, det in enumerate(dets):
        if start_dt <= ev_schedule["datetime"][i] < end_dt and det != 0:
            detentions.append(det)
            soc_i.append(ev_schedule["soc_i_0 (/)"][i])
            soc_f.append(ev_schedule["soc_f_0 (/)"][i])
            hour_of_day.append(ev_schedule["datetime"][i].hour)
    print(f"Median det time: {np.median(detentions)}")
    print(f"Median soc final: {np.median(soc_f)}")
    # Plot histogram of detentions
    plt.figure()
    plt.hist(detentions, bins=range(0, 25))
    plt.title("Histogram of detentions")
    plt.xlabel("Detention (hours)")
    plt.ylabel("Number of detentions")
    plt.xlim(0, 24)
    plt.xticks(range(0, 25))
    plt.tight_layout()

    # Plot hour vs detention
    plt.figure()
    plt.scatter(hour_of_day, detentions)
    plt.title("Detention vs hour of day")
    plt.xlabel("Hour of day")
    plt.ylabel("Detention (hours)")
    plt.xlim(0, 24)
    plt.xticks(range(0, 25))
    plt.tight_layout()

    # Plot detention vs soc_i
    plt.figure()
    plt.scatter(soc_i, detentions)
    plt.title("Detention vs initial state of charge")
    plt.xlabel("Initial state of charge (-)")
    plt.ylabel("Detention (hours)")
    plt.xlim(0, 1)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.tight_layout()


def get_average_idle(file_ev_soc, file_ev_soc_f, log_when="full"):
    ev_soc_df = pd.read_csv(file_ev_soc)
    ev_soc = ev_soc_df.to_dict(orient="list")
    ev_soc["datetime"] = [
        datetime.datetime.fromisoformat(d.replace("Z", "+00:00"))
        for d in ev_soc["datetime"]
    ]

    ev_soc_f_df = pd.read_csv(file_ev_soc_f)
    ev_soc_f = ev_soc_f_df.to_dict(orient="list")
    ev_soc_f["datetime"] = [
        datetime.datetime.fromisoformat(d.replace("Z", "+00:00"))
        for d in ev_soc_f["datetime"]
    ]

    file_ev_schedule = "data/houses/common_env/first_ev_schedule.csv"
    ev_schedule_df = pd.read_csv(file_ev_schedule)
    ev_schedule = ev_schedule_df.to_dict(orient="list")
    ev_schedule["datetime"] = [
        datetime.datetime.fromisoformat(d.replace("Z", "+00:00"))
        for d in ev_schedule["datetime"]
    ]

    ev_soc_i = ev_schedule["soc_i_0 (/)"]
    index = 0
    ev_idle_state = list()
    cur_soc_i = 0
    for i, soc in enumerate(ev_soc_i):
        NUM_SIGN = 4
        if soc != 0:
            cur_soc_i = np.format_float_positional(
                soc,
                precision=NUM_SIGN,
                unique=False,
                fractional=False,
                trim="k",
            )
        dt = ev_schedule["datetime"][i]
        if index == len(ev_soc["datetime"]):
            break

        if dt == ev_soc["datetime"][index]:
            cur_soc = np.format_float_positional(
                ev_soc["soc"][index],
                precision=NUM_SIGN,
                unique=False,
                fractional=False,
                trim="k",
            )
            cur_soc_f = np.format_float_positional(
                ev_soc_f["soc"][index],
                precision=NUM_SIGN,
                unique=False,
                fractional=False,
                trim="k",
            )
            at_end_of_soc = cur_soc == cur_soc_f
            at_start_soc = cur_soc == cur_soc_i
            zero_np_form = np.format_float_positional(
                0,
                precision=NUM_SIGN,
                unique=False,
                fractional=False,
                trim="k",
            )
            if cur_soc_f == zero_np_form:
                ev_idle_state.append(0)
            elif log_when == "full" and at_end_of_soc:
                ev_idle_state.append(0)
            elif log_when == "start" and at_start_soc:
                ev_idle_state.append(0)
            else:
                ev_idle_state.append(1)
            index += 1

    idle_state_quarter_hour = [[] for _ in range(0, 96)]
    for i, idle_state_quart in enumerate(ev_idle_state):
        idle_state_quarter_hour[i % 96].append(idle_state_quart)

    shift = 40
    idle_state_quarter_hour = (
        idle_state_quarter_hour[-shift:] + idle_state_quarter_hour[:-shift]
    )

    x_axis = [ev_soc["datetime"][i] for i in range(0, 96)]
    x_axis = [i.strftime("%H:%M") for i in x_axis]
    x_axis = [value if i % 8 == 0 else "" for i, value in enumerate(x_axis)]
    x_axis = x_axis[-shift:] + x_axis[:-shift]

    average_idle = [
        np.average(idle_quart) * 100 for idle_quart in idle_state_quarter_hour
    ]

    return average_idle, x_axis


def plot_soc(file_soc):
    plt.figure()
    soc_df = pd.read_csv(file_soc)
    soc = soc_df.to_dict(orient="list")
    # datetime list to datetime object
    soc["datetime"] = [
        datetime.datetime.fromisoformat(d.replace("Z", "+00:00"))
        for d in soc["datetime"]
    ]
    soc_datetime = soc["datetime"]
    soc = soc["soc"]

    soc_quarter_hour = [[] for _ in range(0, 96)]
    for i, soc_quart in enumerate(soc):
        soc_quarter_hour[i % 96].append(soc_quart)

    shift = 40
    soc_quarter_hour = soc_quarter_hour[-shift:] + soc_quarter_hour[:-shift]

    x_axis = [soc_datetime[i] for i in range(0, 96)]
    x_axis = [i.strftime("%H:%M") for i in x_axis]
    x_axis = [value if i % 8 == 0 else "" for i, value in enumerate(x_axis)]
    x_axis = x_axis[-shift:] + x_axis[:-shift]

    average_soc = [np.average(soc_quart) for soc_quart in soc_quarter_hour]
    std_soc = [np.std(soc_quart) for soc_quart in soc_quarter_hour]
    low_bound = [average_soc[i] - std_soc[i] for i in range(0, 96)]
    high_bound = [average_soc[i] + std_soc[i] for i in range(0, 96)]

    plt.boxplot(soc_quarter_hour, labels=x_axis)

    plt.title("State of charge per quarter hour")

    plt.ylabel("State of charge (-)")


if __name__ == "__main__":
    # analyse_common_env(5)
    # analyse_ev_for_forecasting()
    # plot_ev_arrival()
    file_ev_schedule = "data/houses/common_env/ev_schedule_jan_mar_2023_2024.csv"
    file_ev_schedule = "data/houses/common_env/ev_schedule_2023_2024.csv"
    plot_ev(file_ev_schedule)
    plt.show()
