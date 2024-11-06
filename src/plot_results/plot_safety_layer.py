import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import json


def plot_safety_layer(result_dir="data/results/"):
    results = json.load(open(f"{result_dir}safety_layer_results.json", "r"))

    ems_types = ["RL", "RBC", "Tree", "MPC"]
    # Filter out values below 9200 for grid data
    for ems in ems_types:
        for key_corr in ["grid", "correction"]:
            results[key_corr][ems] = pd.Series(results[key_corr][ems])
        results["grid"][ems] = results["grid"][ems][results["grid"][ems] > 9200]

    # Prepare the data for plotting
    ems_labels = ems_types
    total_activated_values = [results["activated"][ems] for ems in ems_types]
    total_crashed_values = [int(results["crashed"][ems] * -1) for ems in ems_types]

    # Data for the violin plots
    correction_data = [results["correction"][ems].values * -1 for ems in ems_types]
    grid_exchange_data = [
        results["grid"][ems].dropna().values - 9200 for ems in ems_types
    ]

    total_exceedance_values = [results["exceedances"][ems] for ems in ems_types]
    total_active_exceedance_values = [
        results["active_exceedances"][ems] for ems in ems_types
    ]

    # Define the positions of the bars on the x-axis
    x = np.arange(len(ems_labels)) * 1.25

    # Set the width of the bars
    width = 0.2

    # Create the bar plot
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plotting the bars on the primary y-axis
    bars1 = ax1.bar(
        x - 2 * width,
        total_activated_values,
        width,
        label="Total Activations (Count)",
        color="tab:blue",
    )
    bars2 = ax1.bar(
        x - 1 * width,
        total_crashed_values,
        width,
        label="Total Crashes (Count)",
        color="tab:orange",
    )

    # Plot the violin plots for correction data
    violin_parts_correction = ax1.violinplot(
        correction_data,
        positions=x + 0.025 + 1 * width,
        widths=width,
        showmeans=False,
        showmedians=True,
    )

    # Customizing the correction violin plots
    for pc in violin_parts_correction["bodies"]:
        pc.set_facecolor("tab:red")
        pc.set_edgecolor("black")
        pc.set_alpha(0.8)
    # Set center line to grey
    violin_parts_correction["cbars"].set_edgecolor("tab:gray")
    for partname in ("cmins", "cmaxes", "cmedians"):
        vp = violin_parts_correction[partname]
        vp.set_edgecolor("black")
        vp.set_linewidth(1)

    # Plot the violin plots for grid exchange data
    violin_parts_grid = ax1.violinplot(
        grid_exchange_data,
        positions=x + 0.025 + 0.01 + 2 * width,
        widths=width,
        showmeans=False,
        showmedians=True,
    )

    # Customizing the grid exchange violin plots
    for pc in violin_parts_grid["bodies"]:
        pc.set_facecolor("tab:purple")
        pc.set_edgecolor("black")
        pc.set_alpha(0.8)
    # Set center line to grey
    violin_parts_grid["cbars"].set_edgecolor("tab:gray")
    for partname in ("cmins", "cmaxes", "cmedians"):
        vp = violin_parts_grid[partname]
        vp.set_edgecolor("black")
        vp.set_linewidth(1)

    # Stacked bar for total and active exceedances
    bars3a = ax1.bar(
        x,
        total_active_exceedance_values,
        width,
        label="Active Grid Exceedances (Count)",
        color="tab:green",
        hatch="//",
        edgecolor="black",
    )
    bars3b = ax1.bar(
        x,
        [
            total_exceedance_values[i] - total_active_exceedance_values[i]
            for i in range(len(ems_labels))
        ],
        width,
        bottom=total_active_exceedance_values,
        label="Total Grid Exceedances (Count)",
        color="tab:green",
    )

    # Adding a custom legend for the violin plots
    violin_patch_correction = mpatches.Patch(
        color="tab:red", label="Median Correction (Watt)"
    )
    violin_patch_grid = mpatches.Patch(
        color="tab:purple", label="Grid Exchange Power (Watt)"
    )

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax1.set_xlabel('EMS Types', fontsize=16)
    # ax1.set_title('Comparison of Safety Layer Results by EMS Type', fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["RL", "RBC", "TreeC", "MPC"], fontsize=16)
    ax1.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)

    # Increase the size of the y-axis tick labels
    ax1.tick_params(axis="y", labelsize=14)

    # Combine legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = [violin_patch_correction, violin_patch_grid], [
        "Correction (Watt)",
        "Grid Exceedance (Watt)",
    ]

    # Reorder the legend to put "Inactive Grid Exceedances" before "Active Grid Exceedances"
    handles1.insert(-1, handles1.pop(-1))
    labels1.insert(-1, labels1.pop(-1))

    ax1.legend(
        handles=handles1 + handles2,
        labels=labels1 + labels2,
        fontsize=12,
        loc="upper left",
    )

    # Attach a text label above each bar, displaying its height
    def autolabel(bars, ax, offset=(0, 0)):
        """Attach a text label above each bar in *bars*, displaying its height."""
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                "{}".format(round(height, 2)),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=offset,  # points offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                color="black",
                fontsize=12,
                weight="bold",
            )

    # Extract and label the medians of the violin plots
    def annotate_violins(violin_parts, ax, data, positions, offset=(0, 3)):
        """Annotate the medians of the violin plots."""
        for i, pos in enumerate(positions):
            median = np.median(data[i])
            ax.annotate(
                "{}".format(int(median)),
                xy=(pos, median),
                xytext=offset,  # points offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                color="black",
                fontsize=12,
                weight="bold",
            )

    autolabel(bars1, ax1)
    autolabel(bars2, ax1)
    autolabel(bars3a, ax1, offset=(0, 0))
    # autolabel(bars3b, ax1)

    # Annotate the medians of the violin plots
    annotate_violins(
        violin_parts_correction,
        ax1,
        correction_data,
        x + 0.025 + 1 * width,
        offset=(0, 0),
    )
    annotate_violins(
        violin_parts_grid,
        ax1,
        grid_exchange_data,
        x + 0.025 + 0.01 + 2 * width,
        offset=(0, 0),
    )

    fig.tight_layout()


if __name__ == "__main__":
    plot_safety_layer()
    plt.show()
