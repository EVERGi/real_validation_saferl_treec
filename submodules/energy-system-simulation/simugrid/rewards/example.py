from simugrid.rewards.reward import Reward
from simugrid.assets.battery import Battery


class ExampleReward(Reward):
    def __init__(self, list_KPI):
        Reward.__init__(self, list_KPI)

    def calculate_kpi(self, time_step):
        # Evaluate OPEX if provided in list_KPI by the user during construction of Reward object
        if "OPEX" in self.KPIs:

            # Evaluate each reward info provided
            for reward_info in self.reward_information:
                # Usefull variables for further calculations
                environment = reward_info.environment
                t_st_hours = time_step.total_seconds() / 3600
                energy = t_st_hours*reward_info.power

                # Set the OPEX for the SolarPv_1 asset to 0 based on its name
                if reward_info.asset.name == "SolarPv_1":
                    self.KPIs["OPEX"] += 0

                # Add a cost to discharging batteries
                elif isinstance(reward_info.asset, Battery):
                    disch_cost_per_kWh = 0.0
                    if energy > 0:
                        self.KPIs["OPEX"] += -energy*disch_cost_per_kWh

                # In all other cases, use the default OPEX function
                else:
                    # Try using the default OPEX function, if fail add 0 to the OPEX value
                    try:
                        self.KPIs["OPEX"] += reward_info.asset.opex()
                    except AttributeError:
                        self.KPIs["OPEX"] += 0

        self.reward_information = list()


# Example of run with new reward calculator
if __name__ == "__main__":
    from simugrid.management.rational import DieterenManager
    from simugrid.misc.log_plot_micro import log_micro, setup_logs
    from simugrid.utils import DATA_DIR
    from simugrid.simulation.config_parser import parse_config_file
    from simugrid.misc.log_plot_micro import plot_files

    config_file = DATA_DIR + "dieteren_case/microgrid_config.csv"
    dieteren = parse_config_file(config_file)

    # Set new reward calculator
    reward = ExampleReward(["OPEX"])
    dieteren.set_reward(reward)

    ems = DieterenManager(dieteren)
    log_dir_name = DATA_DIR + "results/example/"

    iter = 100

    log_dir = setup_logs(dieteren, log_dir_name, "dieteren_microgrid", iter)
    for i in range(iter):
        dieteren.management_system.simulate()
        log_micro(dieteren, log_dir)

    plot_files(log_dir)

