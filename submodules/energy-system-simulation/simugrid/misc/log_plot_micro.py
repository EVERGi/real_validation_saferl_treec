import os
import pandas
import matplotlib.pyplot as plt
import re
from pathlib import Path
import shutil
import datetime
import numpy as np
import matplotlib.colors as mcolors
import io
import cv2
from simugrid.simulation.power import Power


def setup_logs(microgrid, directory, scenario, iter, config_file="", log=True):
    if not log:
        return ""

    dir_path = Path(directory)
    pattern = re.compile(directory[:-1] + "_[0-9]+")
    parent = dir_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    max_int = -1

    for filepath in os.listdir(parent):
        complete_path = str(parent).replace("\\", "/") + "/" + filepath
        if pattern.match(complete_path):
            end_int = int(re.search(r"\d+$", complete_path).group())
            if end_int > max_int:
                max_int = end_int
    max_int += 1

    result_path = Path(str(dir_path) + "_" + str(max_int))
    result_path.mkdir()

    if config_file != "":
        shutil.copyfile(config_file, str(result_path) + "/microgrid_config.csv")

    ems = microgrid.management_system

    if ems.__class__.__name__ == "BaselineExecuter":
        model_path = Path(ems.cfg_file)

        shutil.copyfile(model_path, str(result_path) + "/" + model_path.name)

    param_filepath = str(result_path) + "/microgrid_param.csv"
    with open(param_filepath, "w+") as param_file:
        param_file.write('"scenario","' + scenario + '"\n')
        param_file.write('"EMS_class","' + ems.__class__.__name__ + '"\n')

        param_file.write('"time_step","' + str(microgrid.time_step) + '"\n')

        param_file.write(
            '"start_t","'
            + str(microgrid.datetime.strftime("%d/%m/%Y %H:%M:%S"))
            + '"\n'
        )
        param_file.write('"iter","' + str(iter) + '"\n')

        if ems.__class__.__name__ == "BaselineExecuter":
            model_path = Path(ems.cfg_file)
            gym_env = ems.env.envs[0]

            param_file.write('"model","' + ems.model.__class__.__name__ + '"\n')
            param_file.write('"model_path","' + model_path.name + '"\n')
            param_file.write('"gym_env","' + gym_env.spec.id + '"\n')
            param_file.write('"episode_steps","' + str(gym_env.episode_steps) + '"\n')
            param_file.write('"KPI","' + str(gym_env.KPI) + '"\n')

    microgrid.log_dir = str(result_path) + "/"

    return str(result_path) + "/"


def log_power_hist(node_pow, file_path):
    if len(node_pow) == 1:
        return

    file = open(file_path, "w+")
    elec_head = [i + "_elec" for i in node_pow.keys()][1:]
    heat_head = [i + "_heat" for i in node_pow.keys()][1:]

    head = ",".join([list(node_pow.keys())[0]] + elec_head + heat_head) + "\n"
    file.write(head)
    for j in range(len(node_pow["datetime"])):
        elec_line = ""
        heat_line = ""
        for key, value in node_pow.items():
            if key == "datetime":
                start_line = value[j].strftime("%d/%m/%Y %H:%M:%S")
            else:
                elec_line += "," + str(value[j].electrical)
                heat_line += "," + str(value[j].heating)

        file.write(start_line + elec_line + heat_line + "\n")
    file.close()


def log_reward_hist(reward_hist, filepath):
    file = open(filepath, "w+")

    KPI_headers = ",".join(reward_hist.keys())
    line = KPI_headers + "\n"
    file.write(line)

    for i in range(len(reward_hist["datetime"])):
        values = ",".join(
            [str(value[i]) for key, value in reward_hist.items() if key != "datetime"]
        )
        datetime_rew = reward_hist["datetime"][i].strftime("%d/%m/%Y %H:%M:%S")
        line = datetime_rew + "," + values + "\n"
        file.write(line)
    file.close()


def log_branch(branch_hist, filepath):
    if len(branch_hist) == 1:
        return

    file = open(filepath, "w+")

    head_line = ",".join(branch_hist.keys()) + "\n"
    file.write(head_line)
    for j in range(len(branch_hist["datetime"])):
        end_line = ""
        for key, value in branch_hist.items():
            if key == "datetime":
                start_line = value[j].strftime("%d/%m/%Y %H:%M:%S")
            else:
                if j < len(value):
                    end_line += "," + str(value[j])
        file.write(start_line + end_line + "\n")
    file.close()


def log_attributes(attribute_hist, filepath):
    file = open(filepath, "w+")

    head_line = ",".join(attribute_hist.keys()) + "\n"
    file.write(head_line)

    for i, _ in enumerate(attribute_hist["datetime"]):
        line_list = [
            str(value[i])
            if key != "datetime"
            else value[i].strftime("%d/%m/%Y %H:%M:%S")
            for key, value in attribute_hist.items()
        ]
        line = ",".join(line_list) + "\n"
        file.write(line)
    file.close()


def log_micro(microgrid, directory, log=True):
    if not log:
        return

    # Log the power output for each asset
    for i, node_pow in enumerate(microgrid.power_hist):
        file_path = directory + "node" + str(i) + ".csv"

        log_power_hist(node_pow, file_path)

    # Log the total reward
    filename = directory + "total_reward.csv"

    log_reward_hist(microgrid.reward_hist, filename)

    # Log branches
    for i, branch_hist in enumerate(microgrid.branches_hist):
        filepath = directory + "branch" + str(i) + ".csv"

        log_branch(branch_hist, filepath)

    # Log asset attributes
    for asset, attributes in microgrid.attributes_hist.items():
        filepath = directory + asset.name + "_asset.csv"
        log_attributes(attributes, filepath)


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def plot_hist(power_hist, reward_hist, show=False):
    figures_log = plot_power_hist(power_hist)

    figures_log += plot_reward_hist(reward_hist)

    if show:
        plt.show()
    return figures_log


def plot_power_hist(power_hist):
    """
    Plot power profiles and logged rewards over time.
    """
    # plots_nparray = list()
    figures_log = list()
    comb_power_hist = dict()
    for node_hist in power_hist:
        for key, value in node_hist.items():
            comb_power_hist[key] = value

    asset_count = dict()
    for key in comb_power_hist.keys():
        asset_type = key.split("_")[0]
        if asset_type not in asset_count.keys():
            asset_count[asset_type] = list()
        asset_count[asset_type].append(key)

    for key, value in asset_count.items():
        if len(value) > 5:
            comb_power_hist[key] = [Power() for _ in comb_power_hist[value[0]]]
            for asset in value:
                for i, power in enumerate(comb_power_hist[asset]):
                    comb_power_hist[key][i] += power

                del comb_power_hist[asset]

    names = list(comb_power_hist.keys())
    names.remove("datetime")
    x = comb_power_hist["datetime"]

    positif_offset = [0] * len(x)
    negatif_offset = [0] * len(x)
    offset = [0] * len(x)

    stackplot_dict = dict()
    for name in names:
        y_df = comb_power_hist[name]
        y_df = [i.electrical for i in y_df]

        stackplot_dict[name] = [[], []]
        for val in y_df:
            if val < 0:
                stackplot_dict[name][0].extend((-val, -val))
                stackplot_dict[name][1].extend((0, 0))
            else:
                stackplot_dict[name][0].extend((0, 0))
                stackplot_dict[name][1].extend((val, val))

        for time in range(len(y_df)):
            if y_df[time] >= 0:
                offset[time] = positif_offset[time]
            else:
                offset[time] = negatif_offset[time]

        for time in range(len(y_df)):
            if y_df[time] >= 0:
                positif_offset[time] += y_df[time]
            else:
                negatif_offset[time] += y_df[time]

    fig = plt.figure()
    size_list = len(stackplot_dict) * 2
    values_to_plot = [0] * (size_list)
    count = 0

    color_name = [col for col in mcolors.TABLEAU_COLORS] * 100

    for key, value in stackplot_dict.items():
        values_to_plot[int(size_list / 2) - count - 1] = value[0]
        values_to_plot[int(size_list / 2) + count] = value[1]
        count += 1
    new_x = list()
    time_step = x[1] - x[0]
    for x_date in x:
        new_x.append(x_date - time_step / 2)
        new_x.append(x_date + time_step / 2)
    colors = color_name[:count] + color_name[:count][::-1]
    labels = list(stackplot_dict.keys())[::-1]

    plt.stackplot(new_x, values_to_plot, labels=labels, baseline="sym", colors=colors)

    plt.title("Electric power production of assets")
    plt.ylabel("Power (kW)")
    plt.legend()

    fig.autofmt_xdate()
    figures_log.append(fig)

    return figures_log


def plot_reward_hist(reward_hist):
    figures_log = list()
    for key in reward_hist:
        if key == "datetime":
            continue
        fig = plt.figure()
        x = reward_hist["datetime"]
        y = reward_hist[key]
        plt.plot(x, y)
        plt.title(key)

        fig.autofmt_xdate()
        figures_log.append(fig)

    return figures_log


def plot_branches(branches_hist, show=False):
    fig, ax = plt.subplots(1, 2)
    x = branches_hist[0]["datetime"]
    for i in range(len(branches_hist)):
        ax[0].step(x, branches_hist[i]["power"], label="")
        ax[1].step(x, branches_hist[i]["losses"], label="")

    # X-axis
    ax[0].set_xlabel("Time of the day [h]")
    ax[1].set_xlabel("Time of the day [h]")

    # Y-axis
    ax[0].set_ylabel("Power [kW]")
    ax[1].set_ylabel("Losses [W]")

    # Legend
    ax[0].legend()
    fig.autofmt_xdate()
    if show:
        plt.show()


def plot_attributes(attribute_hist, show=False):
    # plots_nparray = list()
    figures_log = list()

    for asset, attributes in attribute_hist.items():
        if isinstance(asset, str):
            asset_name = asset
        else:
            asset_name = asset.name

        for att_name in attributes:
            if att_name == "datetime":
                continue
            fig = plt.figure()
            x = attributes["datetime"]
            y = attributes[att_name]
            plt.plot(x, y)
            plt.title(asset_name + " " + att_name)
            fig.autofmt_xdate()

            figures_log.append(fig)
            # img_arr = get_img_from_fig(fig)
            # plots_nparray.append(img_arr)

        if show:
            plt.show()

    return figures_log


def plot_files(directory):
    branches_hist = list()
    reward_hist = dict()
    attributes_hist = dict()

    power_hist_files = list()
    for filename in os.listdir(directory):
        if filename[:4] == "node":
            power_hist_files.append((directory + filename))

        elif filename[:6] == "branch":
            branches_hist.append(dict())
            df = pandas.read_csv(directory + filename)
            names = df.columns.values[1:]
            x = df["datetime"]
            x = [datetime.datetime.strptime(i, "%d/%m/%Y %H:%M:%S") for i in x]
            branches_hist[-1]["datetime"] = x

            for i in range(len(df.columns) - 1):
                y_frame = df.iloc[:, i + 1]
                branches_hist[-1][names[i]] = y_frame

        elif filename[:4] == "tota":
            df = pandas.read_csv(directory + filename)

            x = df["datetime"]
            x = [datetime.datetime.strptime(i, "%d/%m/%Y %H:%M:%S") for i in x]
            reward_hist["datetime"] = x
            KPIs = df.columns.values[1:]
            for KPI in KPIs:
                reward_hist[KPI] = df[KPI]

        elif filename[-10:] == "_asset.csv":
            asset_name = filename.replace("_asset.csv", "")
            attributes_hist[asset_name] = dict()
            df = pandas.read_csv(directory + filename)
            x = df["datetime"]
            x = [datetime.datetime.strptime(i, "%d/%m/%Y %H:%M:%S") for i in x]
            attributes_hist[asset_name]["datetime"] = x
            attribute_names = df.columns.values[1:]
            for att_name in attribute_names:
                attributes_hist[asset_name][att_name] = df[att_name]

    power_hist = get_power_hist_df(power_hist_files)

    plot_hist(power_hist, reward_hist)
    if len(branches_hist) != 0:
        plot_branches(branches_hist)
    if len(attributes_hist.keys()) != 0:
        plot_attributes(attributes_hist)
    plt.show()


def get_power_hist_df(files_power_hist):
    power_hist = list()
    for filename in files_power_hist:
        power_hist.append(dict())
        df = pandas.read_csv(filename)
        names = df.columns.values[1:]
        names = set([i.replace("_elec", "").replace("_heat", "") for i in names])
        x = df["datetime"]
        x = [datetime.datetime.strptime(i, "%d/%m/%Y %H:%M:%S") for i in x]
        power_hist[-1]["datetime"] = x

        for name in names:
            elec = df[name + "_elec"]
            heat = df[name + "_heat"]
            powers = [Power(elec_val, heat[i]) for i, elec_val in enumerate(elec)]
            power_hist[-1][name] = powers

    return power_hist


def plot_simulation(microgrid):
    plot_hist(microgrid.power_hist, microgrid.reward_hist)

    if len(microgrid.attributes_hist.keys()) != 0:
        plot_attributes(microgrid.attributes_hist)
    plt.show()
