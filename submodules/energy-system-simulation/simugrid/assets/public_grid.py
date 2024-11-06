import typing

from simugrid.assets.asset import Asset, AssetType
from simugrid.simulation.power import Power
from simugrid.simulation.definitions import EnergyVector
import numpy as np
import datetime
import pandas as pd


class PublicGrid(Asset):
    """
    Create public grid type asset

    :ivar max_take_off_power: Maximum power the grid can provide to other
                          assets for one time-step [kW]
    :type max_take_off_power: float
    :ivar max_injection_power: Maximum power the grid can receive from other
                         assets for one time-step [kW]
    :type max_injection_power: float
    :ivar CO2_emissions: C02 emissions produced by kWh used [gCO2/kWh]
    :type CO2_emissions: float
    :ivar monthly_peak_cost: cost for max monthly peak power [€/kW]
    :type monthly_peak_cost: float
    :ivar monthly_peak_power: The maximum amount of power reached during
                              one time-step for the current month [kW]
    :type monthly_peak_power: float
    :ivar size: Size of the asset [m2]
    :type size: float

    """

    def __init__(self, node, name):
        """
        Initialization of the PublicGrid class

        :param node: Node object
        :type node: Node
        :param name: Name of the asset
        :type name: str

        """
        super().__init__(node, name)
        self.max_take_off_power = 1000000
        self.max_injection_power = 1000000

        self.CO2_emissions: float = 0
        self.distribution_cost: float = 0
        self.monthly_peak_cost: float = 0
        self.monthly_peak_power = None

        self.size = 1

        self.energy_type = {EnergyVector.ELECTRIC}
        self.asset_type = AssetType.GRID

    @property
    def environment_keys(self):
        """
        Return the environment keys needed by the asset

        :return: list
        """
        return {type(self).__name__: [f"sell_to_grid", f"buy_from_grid"]}

    def set_power_limits(self, environment):
        """
        Set the high and low power limits

        :param environment: Environment with information to set the
                            power limits
        :type environment: Environment
        """
        self.power_limit_high = Power(electrical=self.max_take_off_power)
        self.power_limit_low = Power(electrical=-self.max_injection_power)

    def opex_calc(self):
        """
        Calculation of the operational cost of the asset

        :return: float
        """
        opex = 0
        time_step = self.parent_node.microgrid.time_step
        t_s_hours = time_step.total_seconds() / 3600
        energy_prod = self.power_output.electrical * t_s_hours
        
        # Energy costs
        if energy_prod < 0:
            opex += (
                -self.parent_node.environment.env_values["sell_to_grid"].value
                * energy_prod
            )
        else:
            opex += (
                -self.parent_node.environment.env_values["buy_from_grid"].value
                * energy_prod
            )

        # Peak power costs
        if (
            self.monthly_peak_power is None
            or self.power_output.electrical > self.monthly_peak_power
        ):
            self.monthly_peak_power = self.power_output.electrical

        cur_datetime = self.parent_node.microgrid.datetime
        next_datetime = cur_datetime + time_step
        final_datetime = self.parent_node.microgrid.end_time

        # If change of month
        if cur_datetime.month != next_datetime.month:
            opex -= self.monthly_peak_cost * self.monthly_peak_power
            self.monthly_peak_power = None
        # If end of simulation 
        elif cur_datetime == final_datetime - time_step:
            opex -= self.monthly_peak_cost * self.monthly_peak_power
            self.monthly_peak_power = None

        return opex

    def get_month_index(self):
        environment = self.parent_node.environment
        time = environment.get_time()
        begin_year = time[0].year
        begin_year = time[0].year
        months_index = list(
            zip(
                *np.unique(
                    [t.month + (t.year - begin_year) * 12 for t in time],
                    return_index=True,
                    return_counts=True,
                )
            )
        )
        months_index = {
            month[0]: list(range(month[1], month[1] + month[2]))
            for month in months_index
        }

        return months_index

    def get_buy_prices(self):
        environment = self.parent_node.environment

        start_dt = self.parent_node.microgrid.start_time
        end_dt = self.parent_node.microgrid.end_time

        return np.array(
            environment.env_values["buy_from_grid"].sample_range(start_dt, end_dt)[
                "values"
            ]
        )

    def get_sell_prices(self):
        environment = self.parent_node.environment

        start_dt = self.parent_node.microgrid.start_time
        end_dt = self.parent_node.microgrid.end_time

        return np.array(
            environment.env_values["sell_to_grid"].sample_range(start_dt, end_dt)[
                "values"
            ]
        )

    def get_grid_co2_emissions(self):
        environment = self.parent_node.environment

        time_step = self.parent_node.microgrid.time_step

        start_dt = self.parent_node.microgrid.start_time
        end_dt = (self.parent_node.microgrid.end_time)
        
        if "grid_co2_emissions" in environment.env_values:
            return np.array(environment.env_values["grid_co2_emissions"].sample_range(start_dt, end_dt)["values"]
                            )
        else:
 
            time_index = pd.date_range(start=start_dt, end=(end_dt-time_step), freq='h') 
            return np.zeros(len(time_index))

    def get_forecast(
        self,
        start_time,
        end_time,
        quality="perfect",
        naive_back=datetime.timedelta(days=7),
    ):
        """
        Returns electricity prices forecast
        """
        forecast = super().get_forecast(start_time, end_time, quality, naive_back)

        environment = self.parent_node.environment
        if quality == "naive":
            start_time -= naive_back
            end_time -= naive_back

        time_sample = self.get_forec_time_samples(start_time, end_time)

        env_keys = environment.env_values.keys()
        if "buy_from_grid" in env_keys and "sell_to_grid" in env_keys:
            forecast["buy_from_grid"] = {"datetime": list(), "values": list()}
            forecast["sell_to_grid"] = {"datetime": list(), "values": list()}
            for times in time_sample:
                sell_to_grid = environment.env_values["sell_to_grid"].sample_range(
                    times[0], times[1]
                )
                forecast["sell_to_grid"]["datetime"] += sell_to_grid["datetime"]
                forecast["sell_to_grid"]["values"] += sell_to_grid["values"]

                buy_from_grid = environment.env_values["buy_from_grid"].sample_range(
                    times[0], times[1]
                )
                forecast["buy_from_grid"]["datetime"] += buy_from_grid["datetime"]
                forecast["buy_from_grid"]["values"] += buy_from_grid["values"]

        return forecast
