from copy import deepcopy
import matplotlib.pyplot as plt

from reproduce_svl.custom_classes import DayAheadEngie

from simugrid.misc.log_plot_micro import plot_hist, plot_attributes


def evaluate_microgrid_trees(trees, params_evaluation):
    microgrid = params_evaluation["microgrid"]
    input_func = params_evaluation["input_func"]
    ManagerClass = params_evaluation["ManagerClass"]
    render = params_evaluation["render"]

    new_microgrid = deepcopy(microgrid)

    reward = DayAheadEngie()
    new_microgrid.set_reward(reward)

    ems = ManagerClass(new_microgrid, trees, input_func, params_evaluation)
    if "switch_hour" in params_evaluation.keys():
        switch_hour = params_evaluation["switch_hour"]
        ems.set_hour_day_switch(switch_hour)

    all_nodes_visited = []

    while new_microgrid.datetime < new_microgrid.end_time:
        tree_nodes = new_microgrid.management_system.simulate_step()
        all_nodes_visited.append(tree_nodes)

    result = new_microgrid.tot_reward.KPIs["opex"]

    if render:
        power_hist = new_microgrid.power_hist
        kpi_hist = new_microgrid.reward_hist
        attributes_hist = new_microgrid.attributes_hist
        plot_attributes(attributes_hist)
        plot_hist(power_hist, kpi_hist)
        plt.show()

    return result, all_nodes_visited
