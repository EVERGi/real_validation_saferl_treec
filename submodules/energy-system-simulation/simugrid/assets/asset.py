from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from simugrid.simulation.power import Power
import pandas as pd
from simugrid.simulation.environment import SeriesValue
import datetime
from simugrid.simulation.definitions import EnergyVector


class AssetType(Enum):
    """
    The type of an asset.
    """

    ASSET = (0,)
    PRODUCER = (1,)
    CONSUMER = (2,)
    PROSUMER = (3,)
    GRID = (4,)

    def __str__(self):
        return self.name


class Asset(ABC):
    """
    Asset of the microgrid that consumes and/or produces power

    :ivar parent_node: Node containing the asset
    :type parent_node: Node
    :ivar name: Descriptive name of asset used for logging
    :type name: str
    :ivar init_var: The initial attributes values of the Asset
    :type init_var: dict
    :ivar mode: The mode of the asset
    :type mode: str
    :ivar energy_type: The energy type of the asset
    :type energy_type: set[EnergyVector]
    :ivar asset_type: The behaviour type of the asset
    :type asset_type: AssetType | None
    :ivar power_limit_high: Maximum production power of asset [kW]
    :type power_limit_high: float
    :ivar power_limit_low: Minimum production power of asset [kW]
    :type power_limit_low: float
    :ivar power_output: Production power of asset [kW]
    :type power_output: Power
    :ivar max_production_power: Maximum power the asset can produce [kW]
    :type max_production_power: float
    :ivar max_consumption_power: Maximum power the asset can consume [kW]
    :type max_consumption_power: float
    :ivar controllable: Can the asset be controlled?
    :type controllable: bool
    :ivar constraints: list of constraints
    :type constraints: list
    :ivar basic_type: is it a basic type of asset (used for working with csv files)
    :type basic_type: bool
    :ivar simulated_power:  Reference to environment value with
        the power profile of the asset
    :type simulated_power: dict
    """

    @abstractmethod
    def __init__(self, node, name):
        # Mandatory parent node for asset
        self.parent_node = node
        self.microgrid = self.parent_node.microgrid
        node.assets += [self]
        node.microgrid.assets += [self]

        # Descriptive name of asset
        self.name: str = name
        self.asset_id = ""  # todo make an unique id for all the assets
        self.init_var: dict = {}
        self.mode: str = "Error"
        self.energy_type = set()
        self.asset_type = AssetType.ASSET

        # Variables defining the power range at which the asset operate
        # power_output needs to be within this range when all actions
        # have been executed by the energy management system
        self.power_limit_high: Power = Power(0.0, 0.0)
        self.power_limit_low: Power = Power(0.0, 0.0)

        # Power production of asset
        # When negative, the asset draws power from other assets
        # When positive, the asset supplies power to other assets
        self.power_output: Power = Power(0.0, 0.0)

        self.max_production_power = 0
        self.max_consumption_power = 0

        self.controllable: bool = False
        self.constraints: list = list()

        self.basic_type: bool = True

        self.simulated_power: dict = {
            EnergyVector.ELECTRIC: None,
            EnergyVector.HEAT: None,
        }

        self.size = -1

        super().__init__()

    @property
    def environment_keys(self):
        return {type(self).__name__: []}

    def set_power_limits(self, environment):
        """
        Set the high and low power limits

        :param environment: Environment with information to set the power limits
        :type environment: Environment
        """
        pass

    def power_consequences(self):
        """
        Apply all consequences of the asset's power_output value

        :param reward: Reward to update based on the value of power_output
        :type reward: Reward
        """
        pass

    def set_attributes(self, var_dict):
        """
        Set asset attributes from dictionary and checks if the environment
            alues are present

        :param var_dict: dictionary with attribute name as key and attribute
            value as value
        :type var_dict: dict

        :return: checks if the simulation can run for the given attributes.
            Return True if it can run. False if not
        :rtype: bool
        """

        # check if all attributes are an atrribute of the asset
        attr_of_asset = list(filter(lambda x: not hasattr(self, x), var_dict.keys()))

        for var_name, value in var_dict.items():
            setattr(self, var_name, value)
        for key, value in var_dict.items():
            self.init_var[key] = value

        if attr_of_asset:
            message = "no such attributes {} in {}:".format(
                ", ".join(attr_of_asset), self.name
            )
            print("\033[93m" + message + "\033[0m")
        self.check_and_set_model()

    def check_and_set_model(self):
        environment = self.parent_node.environment
        microgrid = self.microgrid

        if environment is None:
            return False

        model_used = False
        for x in self.environment_keys.items():
            for b in x[1]:
                if all([b in environment.env_values]):
                    model_used = True

        # TODO! Strange issue with Consumer_0_electric
        # He does not find Consumer_0_electric in environment
        # The above commented code does not has this issue. Weird no ?
        # model_used = next(
        #    (
        #        x[0]
        #        for x in self.environment_keys.items()
        #        if all([b in self.parent_node.environment.env_values for b in x[1]])
        #    ),
        #    None,
        # )

        if not model_used:
            message = "to use the {} {} asset the environment should contain keys of one of the following " "conditions: ".format(
                type(self).__name__, self.name
            ) + "; ".join(
                (
                    list(
                        map(
                            lambda x: x[0] + " case: " + ", ".join(x[1]),
                            self.environment_keys.items(),
                        )
                    )
                )
            )
            # print("\033[93m" + message + "\033[0m")
            return False

        # Add the power of the asset to the simulated_power variable
        for energy_vector in self.energy_type:
            env_power_key = f"{self.name}_{energy_vector}"
            if env_power_key in environment.env_values.keys():
                self.simulated_power[energy_vector] = environment.env_values[
                    env_power_key
                ]
            elif hasattr(self, "power_from_env"):
                power = self.power_from_env(
                    microgrid.start_time,
                    microgrid.end_time,
                    energy_type=energy_vector,
                )
                if power is not None:
                    date_range = pd.date_range(
                        microgrid.start_time,
                        microgrid.end_time,
                        freq=microgrid.time_step,
                        tz=microgrid.local_tz,
                        inclusive="left",
                    )

                    power_ser = pd.Series(power, index=date_range)
                    power_ser.name = self.name
                    series_value = SeriesValue(self.parent_node.microgrid, power_ser)
                    self.simulated_power[energy_vector] = series_value
                    environment.env_values[self.name] = series_value

        self.mode = model_used
        return True

    def get_forecast(
        self,
        start_time,
        end_time,
        quality="perfect",
        naive_back=datetime.timedelta(days=1),
    ):
        forecast = dict()

        for energy_vector, series_value in self.simulated_power.items():
            if series_value is not None:
                forecast[f"{energy_vector}_power"] = series_value.get_forecast(
                    start_time, end_time, quality, naive_back
                )

        return forecast

    def opex_calc(self) -> float:
        """
        Calculation of the operational cost of the asset

        :return: float
        """
        return 0.0

    def get_id(self) -> int:
        """
        Search for the id of the asset

        :return: the id of the asset
        :rtype: int
        """
        count = 0
        for node in self.parent_node.microgrid.nodes:
            for asset in node.assets:
                if asset is self:
                    break
                count += 1

        return count

    def power_reset(self):
        self.power_output = Power(0.0, 0.0)

    @classmethod
    def asset_from_csv(cls, csv):
        pass

    @classmethod
    def asset_from_json(cls, json, node):
        """
        Create an asset type from a json file.

        :param json: json object of the asset
        :type json: Dict
        :param node: the parent node
        :type node: Node

        :return: the created asset
        :rtype: Asset
        """

        attributes = json["attributes"]
        name = json["name"]
        asset = cls(node, name)
        asset.set_attributes(attributes)

        return asset
