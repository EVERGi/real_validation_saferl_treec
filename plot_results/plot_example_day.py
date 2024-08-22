import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.dates import DateFormatter


def plot_example_day():
    csv_file = "data/example_day/RBC_house3_5s_30th_may.csv"

    df = pd.read_csv(csv_file)

    df_safety = df["safety"]

    df_safety = df_safety * -10

    # Interpolate only column grid of the dataframe
    df["grid"] = -df["grid"].interpolate()

    average_every = 24

    df_grid = df["grid"]
    df_grid = df_grid.rolling(average_every).mean()
    df_grid = df_grid / 1000

    # Load column only negative
    df["load"] = df["load"].clip(upper=0)

    # df.plot(x="time", y="grid")

    cols_to_stack = ["pv", "load", "bat", "charger"]

    df_plot = df[cols_to_stack]
    df_plot = df_plot / 1000

    # Average on 10 seconds instead of 5

    df_plot = df_plot.rolling(average_every).mean()

    w_minvar1nc_pos = df_plot[df_plot >= 0].fillna(0)
    w_minvar1nc_neg = df_plot[df_plot < 0].fillna(0)
    # initialize stackplot
    # fig, ax = plt.subplots(figsize=(12, 5))
    # Put two plots one on top of the other
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True, height_ratios=[2, 1])
    x = np.arange(0, 10, 2)
    ay = [1, 1.25, 2, 2.75, 3]
    by = [1, 1, 1, 1, 1]
    cy = [2, 1, 2, 1, 2]
    y = np.vstack([ay, by, cy])

    # create and format stackplot
    pos_vstack = np.vstack([w_minvar1nc_pos[col] for col in w_minvar1nc_pos.columns])

    index = df["time"].values
    index = pd.to_datetime(index)

    safety_index = df["time"].values

    battery_color = "#FF6347"

    # Propose me a list of 4 colors that represent solar, electrical load, battery and electric vehicle charger
    # Stay in the matplotlib theme
    colors = ["#FFD700", "#1E90FF", battery_color, "tab:green"]

    labels = ["PV", "Electrical load", "BESS", "EV charger"]

    ax[0].stackplot(
        index,
        pos_vstack,
        labels=labels,
        colors=colors,
    )

    neg_vstack = np.vstack([w_minvar1nc_neg[col] for col in w_minvar1nc_neg.columns])
    ax[0].stackplot(
        index,
        neg_vstack,
        colors=colors,
    )

    # Plot the grid as a line on top
    # Propose me a color that represents the grid
    grid_color = "grey"
    linewidth = 1.5

    ax[0].plot(index, df_grid, label="Grid", color=grid_color, linewidth=linewidth)

    # Only take the values that are lower than -1
    df_safety = df_safety[df_safety < -1]

    # Keep relevant indexes
    df_safety_index = index[df_safety.index]

    # Get indexes before 2024/05/31 11:20
    size_before = df_safety_index[df_safety_index < "2024-05-31 11:20"].size

    first_activated = df_safety[:size_before]
    second_activated = df_safety[size_before:]

    df_grid_first = df_grid[first_activated.index]
    df_grid_second = df_grid[second_activated.index]

    first_activated_index = index[first_activated.index]
    second_activated_index = index[second_activated.index]

    ax[0].plot(
        first_activated_index,
        df_grid_first,
        label="Grid+safety activated",
        color="black",
        linewidth=linewidth,
    )

    ax[0].plot(
        second_activated_index, df_grid_second, color="black", linewidth=linewidth
    )

    # plt.ylabel("Power (kW)")
    ax[0].set_ylabel("Power (kW)")
    # Put legend bottom left
    ax[0].legend(loc="lower center")
    # plt.clf()

    formatter = DateFormatter("%H:%M")
    ax[1].xaxis.set_major_formatter(formatter)

    # Plot columns soc_bat and soc_ev from the original df

    # df["soc_bat"] = df["soc_bat"] / 100
    df["soc_ev"] = df["soc_ev"] * 100

    # df["soc_bat"] = df["soc_bat"].interpolate()
    # df["soc_ev"] = df["soc_ev"].interpolate()

    df["soc_bat"] = df["soc_bat"].rolling(average_every).mean()
    df["soc_ev"] = df["soc_ev"].rolling(average_every).mean()

    ax[1].plot(index, df["soc_bat"], label="BESS", color=battery_color)
    ax[1].plot(index, df["soc_ev"], label="EV", color="tab:green")

    for i in [0, 1]:
        ax[i].set_axisbelow(True)
        ax[i].grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.ylabel("SOC (%)")

    plt.xlim(
        [index[0] - pd.Timedelta(minutes=15), index[-1] + pd.Timedelta(minutes=15)]
    )

    plt.legend(loc="lower left")
    plt.tight_layout()


if __name__ == "__main__":
    plot_example_day()
    plt.show()
