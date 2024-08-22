import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import copy
import datetime
import pandas as pd
from matplotlib.patches import Ellipse


def calc_import_export_costs(grid_import_export, elec_tariffs):
    day_ahead_price = elec_tariffs["day_ahead_price"]
    capacity_tariff = elec_tariffs["capacity_tariff"]
    exp_cap_tar = capacity_tariff * 12 / 30
    kwh_offtake_cost = elec_tariffs["kwh_offtake_cost"]
    year_cost = elec_tariffs["year_cost"]
    injection_extra = elec_tariffs["injection_extra"]
    offtake_extra = elec_tariffs["offtake_extra"]
    time_step_h = elec_tariffs["time_step_h"]

    day_ahead_cost = 0
    offtake_cost = 0
    capacity_cost = 0
    constant_cost = 0

    capacity_cost = -2.5 * exp_cap_tar
    max_grid_peak = 2.5

    day_ahead_hist = list()
    offtake_hist = list()
    capacity_hist = list()
    constant_hist = list()

    for i in range(len(grid_import_export["import"])):

        grid_import = grid_import_export["import"][i] * time_step_h
        grid_export = grid_import_export["export"][i] * time_step_h
        dt_str = grid_import_export["datetime"][i]
        dt = datetime.datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%SZ")

        if dt.hour == 15 and dt.minute == 0:
            day_ahead_hist.append(0)
            offtake_hist.append(0)
            capacity_hist.append(0)
            constant_hist.append(0)

        price_ind = day_ahead_price["datetime"].index(dt_str)
        cur_day_ahead_price = day_ahead_price["values"][price_ind]

        day_ahead_price_import = cur_day_ahead_price + offtake_extra
        if day_ahead_price_import > 0:
            day_ahead_cost += -grid_import * day_ahead_price_import * 1.06
        else:
            day_ahead_cost += -grid_import * day_ahead_price_import

        day_ahead_price_export = cur_day_ahead_price + injection_extra
        day_ahead_cost += day_ahead_price_export * grid_export
        day_ahead_hist[-1] = day_ahead_cost

        offtake_cost += -kwh_offtake_cost * grid_import
        offtake_hist[-1] = offtake_cost
        cur_grid_peak = grid_import / time_step_h

        if cur_grid_peak > max_grid_peak:
            capacity_cost += -(cur_grid_peak - max_grid_peak) * exp_cap_tar
            max_grid_peak = cur_grid_peak
        capacity_hist[-1] = capacity_cost

        constant_cost += -year_cost / 365 / 24 * time_step_h
        constant_hist[-1] = constant_cost

    costs = {
        "day_ahead": day_ahead_hist,
        "offtake": offtake_hist,
        "capacity": capacity_hist,
        "constant": constant_hist,
    }
    return costs


def calc_from_logs(log_folder, single_val_grid=False):
    houses = [1, 2, 3, 5]
    emss = ["RL", "RBC", "Tree", "MPC"]
    elec_tariffs = json.load(open(f"{log_folder}elec_tariffs.json", "r"))

    different_costs = ["day_ahead", "offtake", "capacity", "constant"]

    real_result = {ems: {cost_type: 0 for cost_type in different_costs} for ems in emss}
    for ems in emss:
        for cost_type in different_costs:
            real_result[ems][f"{cost_type}_hist"] = list()

    simul_result = copy.deepcopy(real_result)

    for house_num in houses:
        for ems in emss:
            house_ems_file = f"{log_folder}house_{house_num}_{ems}_grid.csv"

            result_data = pd.read_csv(house_ems_file).to_dict(orient="list")
            datetimes = result_data["datetime"]
            if single_val_grid:
                single_power = [
                    p - result_data["grid_export"][i]
                    for i, p in enumerate(result_data["grid_import"])
                ]
                single_import = [p if p > 0 else 0 for p in single_power]
                single_export = [-p if p < 0 else 0 for p in single_power]
                real_grid = {
                    "datetime": datetimes,
                    "import": single_import,
                    "export": single_export,
                }
            else:
                real_grid = {
                    "datetime": datetimes,
                    "import": result_data["grid_import"],
                    "export": result_data["grid_export"],
                }

            real_cost = calc_import_export_costs(real_grid, elec_tariffs)

            tot_cost = 0
            for cost_type in real_cost.keys():
                real_result[ems][f"{cost_type}_hist"] += real_cost[cost_type]
                cur_cost = real_cost[cost_type][-1]
                real_result[ems][cost_type] += cur_cost
                tot_cost += cur_cost

            simul_import = [p if p > 0 else 0 for p in result_data["grid_power"]]
            simul_export = [-p if p < 0 else 0 for p in result_data["grid_power"]]
            simul_grid = {
                "datetime": datetimes,
                "import": simul_import,
                "export": simul_export,
            }

            simul_cost = calc_import_export_costs(simul_grid, elec_tariffs)

            tot_cost = 0
            for cost_type in simul_cost.keys():
                simul_result[ems][f"{cost_type}_hist"] += simul_cost[cost_type]
                cur_cost = simul_cost[cost_type][-1]
                simul_result[ems][cost_type] += cur_cost
                tot_cost += cur_cost

    return real_result, simul_result


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


def plot_cost_comparison(all_costs, remove_capacity=False, only_paper_plots=False):
    to_compare = ["Simul", "Perfect"]

    algos_to_plot = ["MPC", "RL", "RBC", "Tree"]
    costs = ["day_ahead", "offtake", "capacity", "constant"]

    for comp in to_compare:
        for algo in algos_to_plot:
            if not (algo == "MPC" and comp == "Simul") and only_paper_plots:
                continue

            comp_cost = sum_hist_cost(all_costs[comp][algo], remove_capacity)
            real_cost = sum_hist_cost(all_costs["Real"][algo], remove_capacity)
            num_days = int(comp_cost.shape[0] / 4)
            if not only_paper_plots:
                plt.figure()
                ax = plt.gca()
                ax.set_axisbelow(True)
                plt.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)
                for i in range(1, 4):
                    plt.axvline(x=i * num_days - 0.5, color="red", linestyle="--")
                for i in range(4):
                    slice_lenght = len(comp_cost) // 4
                    start = i * slice_lenght
                    end = (i + 1) * slice_lenght
                    x = np.arange(start, end)
                    # Matplotlib blue color code
                    blue_color = "#1f77b4"
                    orange_color = "#ff7f0e"
                    if i == 0:
                        label_blue = comp
                        label_orange = "Real"
                    else:
                        label_blue = None
                        label_orange = None
                    plt.plot(
                        x, comp_cost[start:end], label=label_blue, color=blue_color
                    )
                    plt.plot(
                        x, real_cost[start:end], label=label_orange, color=orange_color
                    )
                # Seperate the figure in 4 equal parts with vertical separators

                plt.title(f"{algo} {comp}")
                plt.ylabel("Cumulative cost (€)")
                # Keep only the two first labels in legend
                plt.legend()

            # Difference
            plt.figure()
            ax = plt.gca()
            ax.set_axisbelow(True)
            plt.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)
            for i in range(1, 4):
                plt.axvline(x=i * num_days - 0.5, color="red", linestyle="--")

            comp_non_cuml = sum_to_non_cumul(comp_cost)
            real_non_cumul = sum_to_non_cumul(real_cost)
            diff_cost = real_non_cumul - comp_non_cuml
            x = np.arange(len(diff_cost))
            plt.scatter(x, diff_cost)

            # Plot a horizontal line at 0, very thin
            plt.axhline(0, color="black", linewidth=0.5)
            plt.ylabel("Cost difference per day (€)")

            x_tick = [5.5, 17.5, 29.5, 41.5]
            x_label = ["House 1", "House 2", "House 3", "House 4"]
            plt.xticks(x_tick, x_label)
            plt.xlim(-0.5, 47.5)
            # plt.ylim(-5, 5)

            plt.tight_layout()
            if algo == "MPC" and comp == "Simul":
                pos_circles = [[2, 1.166028], [7, 2.20128], [28, 4.77539]]
                text = ["Day 3", "Day 8", "Day 5"]
                ax = plt.gca()
                width = 1.5
                height = 0.3
                for i, pos in enumerate(pos_circles):
                    circle = Ellipse(
                        pos,
                        width,
                        height,
                        color="#ff7f0e",
                        fill=False,
                    )
                    ax.add_patch(circle)
                    plt.text(
                        pos[0] - 2, pos[1] - 0.42, text[i], fontsize=12, color="#ff7f0e"
                    )
            if not only_paper_plots:
                plt.title(f"{algo} {comp} - Real")


def plot_side_stack_bar(all_costs):
    label_dict = {
        "day_ahead": "Day-ahead",
        "offtake": "Offtake extras",
        "capacity": "Peak",
        "constant": "Yearly",
    }
    stacked_bar_plots = dict()
    emss = ["RL", "RBC", "Tree", "MPC"]
    cost_types = ["day_ahead", "offtake", "capacity", "constant"]

    plt.figure(figsize=(6.5, 5))
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)

    ind = np.arange(len(emss))
    width = 0.5 / len(all_costs.keys())

    # Give me the first four colors of the matplotlib color cycle
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:4]

    for key, values in all_costs.items():
        stacked_bar_plots[key] = {c: [0 for _ in emss] for c in cost_types}

        for i, ems in enumerate(emss):
            for j, cost_type in enumerate(cost_types):
                stacked_bar_plots[key][cost_type][i] = -values[ems][cost_type]

        bottom = np.array([0.0 for _ in emss])
        for key_2, weight_count in stacked_bar_plots[key].items():
            ind_key = list(all_costs.keys()).index(key)
            ind_color = list(stacked_bar_plots[key].keys()).index(key_2)
            if ind_key == 0:
                label = label_dict[key_2]
            else:
                label = None
            if key == "Real":
                hatch = None
            elif key == "Simul":
                hatch = "//"
            elif key == "Perfect":
                hatch = "//"

            if key == "Perfect":

                x_pos = ind + width * ind_key + 0.1
            else:
                x_pos = ind + width * ind_key
            if label == "Tree":
                label = "TreeC"

            p = plt.bar(
                x_pos,
                weight_count,
                width,
                label=label,
                bottom=bottom,
                color=colors[ind_color],
                hatch=hatch,
                edgecolor="black",
            )

            bottom += np.array(weight_count)

    xticks_add = (len(all_costs.keys()) - 1) * width / 2
    xticks_add = width / 2
    xticks_perf = ind + 2 * width + 0.1
    all_ticks = list(ind + xticks_add) + list(xticks_perf)
    # emss_label = emss.replace("Tree", "TreeC")
    emss[2] = "TreeC"
    emss_perf = emss + ["MPC P"] * 4
    plt.xticks(all_ticks, emss_perf)

    ax = plt.gca()  # Define the variable "ax" as the current axes object
    handles, labels = ax.get_legend_handles_labels()

    # manually define a new patch
    patch = mpatches.Patch(facecolor="white", label="Experiment", edgecolor="black")
    simul_patch = mpatches.Patch(
        facecolor="white", label="Simulation", hatch="//", edgecolor="black"
    )
    # perfect_patch = mpatches.Patch(
    #    facecolor="white", label="Perfect forecast MPC", hatch="x", edgecolor="black"
    # )

    # handles is a list, so append manual patch
    handles = handles[::-1]
    handles.append(patch)
    handles.append(simul_patch)
    handles.append(mpatches.Patch(facecolor="white"))
    # handles.append(perfect_patch)

    plt.ylabel("Total cost (€)")

    # plot the legend
    plt.legend(handles=handles, ncol=2)
    plt.tight_layout()


def plot_comparison_results(
    log_folder,
    log_perfect="data/results/perfect_mpc/",
    plot_all_comparison=True,
    only_paper_plots=False,
):

    real_costs, simul_costs = calc_from_logs(log_folder)

    _, perfect_costs = calc_from_logs(log_perfect)

    all_costs = {"Real": real_costs, "Simul": simul_costs, "Perfect": perfect_costs}
    plot_side_stack_bar(all_costs)
    if plot_all_comparison:
        plot_cost_comparison(all_costs, only_paper_plots=only_paper_plots)


def plot_experiment_simulation():
    log_folder = "data/results/experiment_simulation/"
    plot_comparison_results(log_folder, plot_all_comparison=True, only_paper_plots=True)


def plot_house_3_tree_improvement():

    log_folder = "data/results/replace_batt_rbc_house3/"
    real_costs, simul_costs_correction = calc_from_logs(log_folder)

    log_perfect = "data/results/perfect_mpc/"

    _, perfect_costs = calc_from_logs(log_perfect)

    log_folder = "data/results/experiment_simulation/"
    real_costs, simul_costs = calc_from_logs(log_folder)

    to_compare = ["Simul", "Perfect"]

    algos_to_plot = ["Tree"]
    costs = ["day_ahead", "offtake", "capacity", "constant"]

    all_costs = {
        "TreeC": real_costs,
        "TreeC+RBC": simul_costs_correction,
        "MPC P": perfect_costs,
    }

    remove_capacity = False

    x_tick = [5.5, 17.5, 29.5, 41.5]
    x_label = ["House 1", "House 2", "House 3", "House 4"]

    for algo in algos_to_plot:
        plt.figure(figsize=(6.5, 5))
        ax = plt.gca()
        ax.set_axisbelow(True)
        plt.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)
        for key, a_cost in all_costs.items():
            sum_cost = sum_hist_cost(a_cost[algo], remove_capacity)

            num_days = int(sum_cost.shape[0] / 4)
            for i in range(1, 4):
                plt.axvline(x=i * num_days - 0.5, color="red", linestyle="--")
            for i in range(4):
                slice_lenght = len(sum_cost) // 4
                start = i * slice_lenght
                end = (i + 1) * slice_lenght
                x = np.arange(start, end)
                # Matplotlib color palette list
                colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

                color_index = list(all_costs.keys()).index(key)

                if i == 0 or (i == 2 and key == "TreeC+RBC"):
                    label = key
                else:
                    label = None
                if (i != 2 and key != "TreeC+RBC") or (i == 2):
                    # plt.plot(
                    #    x, sum_cost[start:end], label=label, color=colors[color_index]
                    # )  #  , color=blue_color)
                    width = 3
                    if i == 2:
                        x_shift = -width + color_index * width
                    else:
                        x_shift = -width / 2 + (color_index // 2) * width

                    if key == "TreeC":
                        hatch = None
                    else:
                        hatch = "//"

                    plt.bar(
                        x_tick[i] + x_shift,
                        sum_cost[end - 1],
                        color=colors[color_index],
                        label=label,
                        width=width,
                        hatch=hatch,
                    )

            # Seperate the figure in 4 equal parts with vertical separators

        # plt.title(f"{algo}")
        plt.ylabel("Cost per house (€)")

        plt.xticks(x_tick, x_label)
        plt.xlim(-0.5, 47.5)

        ax = plt.gca()  # Define the variable "ax" as the current axes object
        handles, labels = ax.get_legend_handles_labels()

        # manually define a new patch
        patch = mpatches.Patch(facecolor="white", label="Experiment", edgecolor="black")
        simul_patch = mpatches.Patch(
            facecolor="white", label="Simulation", hatch="//", edgecolor="black"
        )
        # perfect_patch = mpatches.Patch(
        #    facecolor="white", label="Perfect forecast MPC", hatch="x", edgecolor="black"
        # )

        # handles is a list, so append manual patch
        # handles = handles[::-1]
        handles.append(patch)
        handles.append(simul_patch)
        handles.append(mpatches.Patch(facecolor="white"))
        # handles.append(perfect_patch)

        # plot the legend
        plt.legend(handles=handles, ncol=2, loc="upper left")

        plt.tight_layout()


if __name__ == "__main__":
    plot_house_3_tree_improvement()
    plot_experiment_simulation()
    plt.show()
