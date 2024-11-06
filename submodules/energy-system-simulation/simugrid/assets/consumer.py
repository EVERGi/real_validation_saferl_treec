from __future__ import annotations

import datetime


from simugrid.assets.asset import Asset, AssetType
from simugrid.simulation.power import Power
from simugrid.simulation.definitions import EnergyVector


class Consumer(Asset):
    """
    Create consumer type asset

    :ivar yearly_consumption: yearly consumption of the asset [kWh]
    :type yearly_consumption: float
    :ivar predefined_profile: predefined profile of the asset
    :type predefined_profile: str
    """

    def __init__(self, node, name):
        """
        Initialization of the Consumer class

        :param node: the parent node of the asset
        :type node: Node
        :param name: name of the asset
        :type name: str
        """
        super().__init__(node, name)

        self.energy_type = {EnergyVector.ELECTRIC, EnergyVector.HEAT, EnergyVector.COLD, EnergyVector.NG}
        self.asset_type = AssetType.CONSUMER

        self.size = 1

    @property
    def environment_keys(self):
        return {
            "PowerProfile": [
                f"{self.name}_{energy_type.name.lower()}"
                for energy_type in self.energy_type
            ]
        }

    def set_attributes(self, var_dict):
        """
        Set the attributes of the asset

        :param var_dict: dictionary with the attributes of the asset
        :type var_dict: dict
        """
        super().set_attributes(var_dict)
        self.max_consumption_power = self.size

    def check_and_set_model(self):
        super().check_and_set_model()

    def set_power_limits(self, environment):
        """
        Set the high and low power limits

        :param environment: Environment with information to set the power limits
        :type environment: Environment
        """
        cur_datetime = self.parent_node.microgrid.datetime
        self.power_limit_low = Power(electrical=-self.power_from_env(cur_datetime))
        self.power_limit_high = Power()

    def power_from_env(
        self,
        start_dt=None,
        end_dt=None,
        energy_type=EnergyVector.ELECTRIC,
    ):
        """
        Calculate the power of the charger from the environment

        :param energy_type: energy type of the power to be returned
        :type energy_type: EnergyVector
        :param start_dt: start time of period
        :type start_dt: datetime.datetime
        :param end_dt: end time of period
        :type end_dt: datetime.datetime

        :return: the power input/output
        :rtype: float | list[float]
        """

        environment = self.parent_node.environment
        env_key = f"{self.name}_{energy_type}"
        if env_key not in environment.env_values:
            return None

        cur_dt = self.parent_node.microgrid.datetime

        if start_dt is None and end_dt is None:
            start_dt = self.parent_node.microgrid.start_time
            end_dt = self.parent_node.microgrid.end_time

        if end_dt is None:
            end_dt = start_dt
        simul = start_dt == end_dt == cur_dt

        if simul:
            power = environment.env_values[f"{self.name}_{energy_type}"].value
        else:
            power = environment.env_values[f"{self.name}_{energy_type}"].sample_range(
                start_dt, end_dt
            )["values"]

        return power

    def opex_calc(self):
        """
        Calculation of the operational cost of the asset

        :return: float
        """
        time_step = self.parent_node.microgrid.time_step
        t_s_hours = time_step.total_seconds() / 3600
        power_short = -(self.power_limit_low - self.power_output)
        energy_short = power_short.electrical * t_s_hours

        return -energy_short

    def get_forecast(
        self,
        start_time,
        end_time,
        quality="perfect",
        naive_back=datetime.timedelta(days=7),
    ):
        return super().get_forecast(start_time, end_time, quality, naive_back)
