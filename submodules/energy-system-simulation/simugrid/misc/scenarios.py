import matplotlib.pyplot as plt

from simugrid.assets import *
from simugrid.management import RationalManager
from simugrid.misc.log_plot_micro import log_micro, plot_hist, setup_logs
from simugrid.rewards.reward import DefaultReward
from simugrid.simulation.config_parser import parse_config_file
from simugrid.utils import DATA_DIR


def config_scenario(config_file, scenario_file={"EMS": RationalManager,
                                                "Reward":DefaultReward}):
    """
    Executes the scenario in the given config file and
    returns the power and kpi values of the simulation

    :param config_file: List of list that contains configuration of simulation
    :type config_file: List
    :param scenario_file: Dictionnary that contains configuration of scenario
    :type scenario_file: dictionnary

    :return:
         - power_hist - All power values of each asset during the simulation.
            The index format to get a final value is
                power_hist[node_num]["datetime" or asset name][index]
         The final value is a datetime or a Power object
         - kpi_hist - All kpi values of each asset during the simulation.
            The index format to get a final value is
                kpi_hist["datetime" or kpi name][index]
         The final value is a datetime or a number
    :rtype: tuple
    """
    microgrid = parse_config_file(config_file)
    reward = scenario_file["Reward"]()
    microgrid.set_reward(reward)

    ems = scenario_file["EMS"]
    ems(microgrid)

    while microgrid.datetime != microgrid.end_time:
        microgrid.management_system.simulate_step()

    power_hist = microgrid.power_hist
    kpi_hist = microgrid.reward_hist
    return power_hist, kpi_hist


if __name__ == "__main__":
    power_hist, reward_hist = config_scenario(
        DATA_DIR + "example_mordor/microgrid_config.csv"
    )
    plot_hist(power_hist, reward_hist)
    plt.show()
