import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import copy
import datetime
import pandas as pd
from plot_comparison_results import calc_import_export_costs


def rl_progress_plot(result_dir="data/results/"):
    # Initialize an empty dataframe
    df_rewards_1 = pd.DataFrame()

    for i in [1, 2, 3, 5]:
        log_folder = f"{result_dir}experiment_simulation/"
        house_num = i
        ems = "RL"

        elec_tariffs = json.load(open(f"{log_folder}elec_tariffs.json", "r"))

        house_ems_file = f"{log_folder}house_{house_num}_{ems}_grid.csv"

        result_data = pd.read_csv(house_ems_file).to_dict(orient="list")
        datetimes = result_data["datetime"]

        real_grid = {
            "datetime": datetimes,
            "import": result_data["grid_import"],
            "export": result_data["grid_export"],
        }

        real_cost = calc_import_export_costs(real_grid, elec_tariffs)

        reward_per_day = np.diff(
            (np.array(real_cost["day_ahead"]) + np.array(real_cost["offtake"])) * 100,
            prepend=0,
        )

        # Add reward_per_day as a column to the dataframe
        df_rewards_1[f"house_{house_num}"] = reward_per_day

    # Initialize an empty dataframe
    df_rewards_2 = pd.DataFrame()

    for i in [1, 2, 3, 5]:
        log_folder = "data/results/perfect_mpc/"
        house_num = i
        ems = "RL"

        elec_tariffs = json.load(open(f"{log_folder}elec_tariffs.json", "r"))

        house_ems_file = f"{log_folder}house_{house_num}_{ems}_grid.csv"

        result_data = pd.read_csv(house_ems_file).to_dict(orient="list")
        datetimes = result_data["datetime"]

        simul_import = [p if p > 0 else 0 for p in result_data["grid_power"]]
        simul_export = [-p if p < 0 else 0 for p in result_data["grid_power"]]
        simul_grid = {
            "datetime": datetimes,
            "import": simul_import,
            "export": simul_export,
        }

        cost = calc_import_export_costs(simul_grid, elec_tariffs)

        reward_per_day = np.diff(
            (np.array(cost["day_ahead"]) + np.array(cost["offtake"])) * 100, prepend=0
        )

        # Add reward_per_day as a column to the dataframe
        df_rewards_2[f"house_{house_num}"] = reward_per_day

    df_delta = df_rewards_1 - df_rewards_2

    from scipy.stats import linregress

    # Define a list of colors
    color_list = ["blue", "orange", "green", "red"]

    # Plotting
    plt.figure(figsize=(12, 8))

    # Plot original data and trend line for each house
    for idx, column in enumerate(df_delta.columns):
        # Fit a linear trend line to each house's values
        slope, intercept, _, _, _ = linregress(df_delta.index + 1, df_delta[column])
        trend_line = intercept + slope * (df_delta.index + 1)

        # Use the same color for both data and trend line
        color = color_list[idx % len(color_list)]

        # Adjust label for house_5 to house_4
        if column == "house_5":
            column_label = "house_4"
        else:
            column_label = column

        column_label = f'House {int(column_label.split("_")[1])}'

        # Plot original data
        plt.plot(
            df_delta.index + 1,
            df_delta[column],
            marker="o",
            markersize=5,
            linestyle="-",
            color=color,
            label=f"{column_label}",
            linewidth=1,
            alpha=0.7,
        )

        # Plot trend line with slightly brighter color
        plt.plot(df_delta.index + 1, trend_line, linestyle="--", color=color)

        # Calculate midpoint of the trend line
        midpoint_index = len(trend_line) // 2
        x_pos = df_delta.index[midpoint_index] + 1
        y_pos = trend_line[midpoint_index] + 2

        # Annotate the slope value in the middle of the trend line
        plt.text(
            x_pos,
            y_pos,
            f"Slope: {slope:.2f}",
            color=color,
            fontsize=12,
            verticalalignment="bottom",
            horizontalalignment="center",
            weight="bold",
        )

    # Add labels and title
    plt.xlabel("Day", fontsize=16)
    plt.ylabel("Real - perfect episode reward [€c]", fontsize=16)
    # plt.title('Training Progress by House', fontsize=16)
    plt.xticks(df_delta.index + 1)  # Adjust ticks to match index + 1
    plt.legend(fontsize=14)

    # Add grid
    plt.grid(True, linestyle="--", alpha=0.7)

    # Limit x axis between 1 and 12
    plt.xlim(1, 12)

    # Adjust tick parameters
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.tick_params(axis="both", which="minor", labelsize=12)

    # Show plot
    plt.tight_layout()


if __name__ == "__main__":
    rl_progress_plot()
    plt.show()
