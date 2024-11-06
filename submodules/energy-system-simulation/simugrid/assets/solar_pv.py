from __future__ import annotations

from numpy import ndarray
from simugrid.assets.asset import Asset, AssetType
from simugrid.simulation.power import Power
from simugrid.simulation.definitions import EnergyVector
import numpy as np


class SolarPv(Asset):
    """
    Creates solar pv type asset

    :ivar efficiency: efficiency of the module in the installation [-]
    :type efficiency: float
    :ivar area: total area of the installation [m²]
    :type area: float
    :ivar mod_num: number of module present [-]
    :type mod_num: float
    :ivar mod_area: area of one module [m²]
    :type mod_area: float
    :ivar money_per_kWh: cost in euros per kWh [€/kWh]
    :type money_per_kWh: float
    :ivar scale_env: Name of environment value of solar pv power profile
                     to scale by the max_production_power
    :type scale_env: str
    :ivar inverter: PV inverter limit (=None if no limit) [kW]
    :type inverter: int
    """

    def __init__(self, node, name):
        """
        Initialization of the SolarPv class

        :param node: the parent node of the asset
        :type node: Node
        :param name: name of the asset
        :type name: str
        """
        super().__init__(node, name)

        self.efficiency = 0.2
        self.area = 4  # equivalent to 1 kWp
        self.mod_num = None
        self.mod_area = None
        self.money_per_kWh = 0
        self.sizing = False
        self.inverter = None

        self.scale_env = None

        self.energy_type = {EnergyVector.ELECTRIC}
        self.asset_type = AssetType.PRODUCER

    @property
    def environment_keys(self):
        return {
            "PowerProfile": [self.name],
            type(self).__name__: ["irradiation"],
            "Scaled profile": ["scale_env"],
        }

    def set_attributes(self, var_dict):
        super().set_attributes(var_dict)
        self.max_production_power = self.size
        if self.scale_env is not None:
            self.environment_keys["Scaled profile"] = [self.scale_env]

    def set_power_limits(self, environment):
        """
        Set the high and low power limits

        :param environment: Environment with information to set the
                            power limits
        :type environment: Environment
        """
        # Set limits
        cur_datetime = self.parent_node.microgrid.datetime
        self.power_limit_low = Power(electrical=self.power_from_env(cur_datetime))
        self.power_limit_high = self.power_limit_low

    def power_from_env(
        self, start_dt=None, end_dt=None, energy_type=EnergyVector.ELECTRIC
    ):
        """
        Calculate the power of the wind turbine from the environment

        :param energy_type: the energy type of the power
        :type energy_type: EnergyVector
        :param start_dt: start time of period
        :type start_dt: datetime.datetime
        :param end_dt: end time of period
        :type end_dt: datetime.datetime

        :return: the power input/output [???]
        :rtype: float | list[float]
        """

        environment = self.parent_node.environment
        power = None

        if start_dt is None and end_dt is None:
            start_dt = self.parent_node.microgrid.start_time
            end_dt = self.parent_node.microgrid.end_time

        if end_dt is None:
            end_dt = start_dt
        simul = start_dt == end_dt == self.parent_node.microgrid.datetime

        if self.simulated_power[energy_type] is not None:
            env_value = self.simulated_power[energy_type]
            power = SolarPv.get_env_data(env_value, simul, start_dt, end_dt)
        elif self.name in environment.env_values.keys():
            env_value = environment.env_values[self.name]
            power = SolarPv.get_env_data(env_value, simul, start_dt, end_dt)

        elif self.scale_env in environment.env_values.keys():
            env_value = environment.env_values[self.scale_env]
            power = SolarPv.get_env_data(env_value, simul, start_dt, end_dt)

            power *= self.size

        elif "irradiation" in environment.env_values.keys():
            env_value = environment.env_values["irradiation"]
            irradiation = SolarPv.get_env_data(env_value, simul, start_dt, end_dt)

            if self.mod_num is not None:
                power = (
                    self.mod_num
                    * self.mod_area
                    * self.efficiency
                    * (irradiation / 1000)
                )
            else:
                power = (irradiation / 1000) * self.efficiency * self.area

        return power

    def set_inverter_limit(self):
        """
        Modifies the environment power profile by limiting the power
        to the maximum power of the inverter
        """
        asset_list = self.simulated_power[EnergyVector.ELECTRIC].list
        for i in range(len(asset_list)):
            asset_list[i] = (asset_list[i][0], min(asset_list[i][1], self.inverter))

    def check_and_set_model(self):
        """
        Sets the model for the solar pv asset and
        if succesfull execute the vector_solution_init function.

        """
        success = super().check_and_set_model()
        if self.parent_node.environment:
            if self.inverter is not None:
                self.set_inverter_limit()

        return success

    def opex_calc(self):
        """
        Calculation of the operational cost of the asset

        :return: the operational cost of the time step
        :rtype: float
        """
        time_step = self.parent_node.microgrid.time_step
        t_s_hours = time_step.total_seconds() / 3600
        energy_prod = self.power_output.total_power * t_s_hours

        return energy_prod * self.money_per_kWh

    @staticmethod
    def get_env_data(env_value, simul, start_dt, end_dt):
        if simul:
            data = env_value.value
        else:
            data = env_value.sample_range(start_dt, end_dt)["values"]
        return data
