from __future__ import annotations

from simugrid.assets.asset import Asset, AssetType

from simugrid.simulation.power import Power
from simugrid.simulation.definitions import EnergyVector


class Battery(Asset):
    """
    Creates battery type asset

    :ivar size: Energy capacity of the battery [kWh]
    :type size: float
    :ivar soc: Current state of charge  of the battery [-]
    :type soc: float
    :ivar soc_min: Minimum stat of charge usable [-]
    :type soc_min: float
    :ivar soc_max: Maximum state of charge usable [-]
    :type soc_max: float
    :ivar soc_init: Initial state of charge [-]
    :type soc_init: float
    :ivar charge_eff: Efficiency of the charging process [-]
    :type charge_eff: float
    :ivar disch_eff: Efficiency of the discharging process [-]
    :type disch_eff: float
    :ivar max_charge: Maximum C-rate permitted in charging
                      mode [1/h] (kW/kWh)
    :type max_charge: float
    :ivar max_discharge: Maximum C-rate permitted in discharging
                         mode [1/h] (kW/kWh)
    :type max_discharge: float
    """

    def __init__(self, node, name):
        """
        Initialization of the Battery class

        :param node: the parent node of the asset
        :type node: Node
        :param name: name of the asset
        :type name: str
        """
        super().__init__(node, name)

        self.soc: float = 1
        self.soc_min: float = 0.10
        self.soc_max: float = 1
        self.soc_init: float = 1
        self.charge_eff: float = 0.95
        self.disch_eff: float = 0.95
        self.max_charge: float = 0.73
        self.max_discharge: float = 0.73
        self.loss: float = 0

        self.max_production_power = self.size * self.max_discharge
        self.max_consumption_power = self.size * self.max_charge

        self.energy_type = {EnergyVector.ELECTRIC}
        self.asset_type = AssetType.PROSUMER

    @property
    def environment_keys(self):
        return {"PowerProfile": [self.name], type(self).__name__: []}

    def set_attributes(self, var_dict):
        super().set_attributes(var_dict)

        # Correct soc based on soc_init for compatibility reason with desigrid
        self.soc = self.soc_init

        # Set the power limits (in kw)
        self.max_production_power = self.size * self.max_discharge
        self.max_consumption_power = self.size * self.max_charge

    def set_power_limits(self, environment):
        """
        Set the high and low power limits

        :param environment: Environment with information to set the power limits
        :type environment: Environment
        """
        time_step = self.parent_node.microgrid.time_step
        t_st_hours = time_step.total_seconds() / 3600

        if self.name in environment.env_values.keys():
            self.power_limit_low = Power(
                electrical=environment.env_values[self.name].value
            )
            self.power_limit_high = Power(
                electrical=environment.env_values[self.name].value
            )  # kW
            return

        charge_power_cap = self.max_consumption_power
        free_energy_space = (self.soc_max - self.soc) * self.size
        charge_power_energy = free_energy_space / t_st_hours / self.charge_eff
        self.power_limit_low = Power(
            electrical=-min(charge_power_cap, charge_power_energy)
        )

        discharge_power_cap = self.max_production_power
        energy_available = (self.soc - self.soc_min) * self.size
        discharge_power_energy = energy_available / t_st_hours * self.disch_eff
        self.power_limit_high = Power(
            electrical=min(discharge_power_cap, discharge_power_energy)
        )

    def power_consequences(self):
        """
        Apply all consequences of the asset's power_output value

        :param reward: Reward to update based on the value of power_output
        :type reward: Reward
        """
        if self.size <= 0:
            return
        time_step = self.parent_node.microgrid.time_step
        power = self.power_output.electrical

        self.soc += self.soc_change(power, time_step)

    def soc_change(self, power, time_step):
        """
        Calculate the change in state of charge of the battery

        :param power: Power of the battery [kW]
        :type power: float
        :param time_step: Time step of the simulation [s]
        :type time_step: float
        :param size: Energy capacity of the battery [kWh]
        :type size: float
        :return: Change in state of charge of the battery [-]
        :rtype: float
        """
        size = self.size
        disch_eff = self.disch_eff
        charge_eff = self.charge_eff

        t_st_hours = time_step.total_seconds() / 3600
        if power >= 0:
            return -power * t_st_hours / disch_eff / size
        elif power < 0:
            return -power * t_st_hours * charge_eff / size
