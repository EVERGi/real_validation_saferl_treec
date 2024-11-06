from __future__ import annotations

import pytz
from copy import deepcopy

from simugrid.simulation.power import Power
from simugrid.simulation.branch import get_limit_to_max_branch_power, get_branches_power


class Microgrid:
    """
    Microgrid that contains information about all the components for the simulation

    :ivar local_tz: the timezone of the microgrid
    :type local_tz: timezone
    :ivar start_time: start time of the simulation
    :type start_time: datetime.datetime
    :ivar end_time: end time of the simulation
    :type end_time: datetime.datetime
    :ivar time_step: time between each step
    :type time_step: datetime.timedelta
    :ivar utc_datetime: simulation time in utc
    :type utc_datetime: datetime.datetime
    :ivar datetime: simulation time in microgrid zone
    :type datetime: datetime.datetime
    :ivar environments: list containing all the environments of microgrid
    :type environments: list[Environment]
    :ivar nodes: list containing all the nodes of microgrid
    :type nodes: list[Node]
    :ivar branches:

    """

    def __init__(self, start_time, end_time, time_step, timezone="UTC"):
        """
        :param start_time: start time of the microgrid
        :type start_time: datetime.datetime
        :param end_time: end time of the microgrid
        :type end_time: datetime.datetime
        :param time_step: time it take to go to the next step
        :type time_step: datetime.timedelta
        :param timezone: timezone of the microgrid
        :type timezone: str
        """

        self.local_tz = pytz.timezone(timezone)
        self.start_time = self.local_tz.localize(start_time)
        self.end_time = self.local_tz.localize(end_time)
        self.time_step = time_step
        self.utc_datetime = self.start_time.astimezone(pytz.utc)
        self.datetime = self.utc_datetime.astimezone(self.local_tz)
        self.environments = list()
        self.nodes = list()
        self.assets = list()
        self.branches = list()
        self.management_system = None
        self.actions = list()
        self.tot_reward = None
        self.reward_hist = dict()
        self.power_hist = list()
        self.branches_hist = list()
        self.attributes_hist = dict()
        self.env_hist = list()

        self.log_dir = ""

        self.microgrid_copy = None

        self.model = None

    def set_management_system(self, management_system):
        """
        Sets the management systems of the microgrid

        :param management_system: the management system
        :type management_system: Manager
        """
        self.management_system = management_system

    def set_reward(self, reward):
        """
        Sets all the rewards of the microgrid

        :param reward: the reward information
        :type reward: Reward
        """
        reward.microgrid = self
        self.tot_reward = reward
        self.reward_hist["datetime"] = []
        for key in reward.KPIs.keys():
            self.reward_hist[key] = []

    def new_start_time(self, start_time):
        self.start_time = self.local_tz.localize(start_time)
        self.utc_datetime = self.start_time.astimezone(pytz.utc)
        self.datetime = self.utc_datetime.astimezone(self.local_tz)

        self.env_simulate()

    def env_simulate(self):
        """
        Simulates the environment to the current timestep
        """
        for environment in self.environments:
            environment.update()
        for node in self.nodes:
            node.simulate_step()

    def reward_simulate(self):
        """
        calculates the rewards of the energies transfers that were schedules by the EMS

        """
        # Nodes
        if not self.power_hist:
            self.power_hist = [dict() for _ in self.nodes]
            for i, node in enumerate(self.nodes):
                self.power_hist[i]["datetime"] = list()
                asset_names = [asset.name for asset in node.assets]
                for asset_name in asset_names:
                    self.power_hist[i][asset_name] = list()

        for i, node in enumerate(self.nodes):
            node.power_consequences(self.tot_reward)
            self.power_hist[i]["datetime"].append(self.datetime)

            for asset in node.assets:
                self.power_hist[i][asset.name].append(asset.power_output)
        # Calculate kpi's
        self.tot_reward.calculate_kpi()

        # Branches
        if not self.branches_hist:
            self.branches_hist = [dict() for _ in self.branches]
            for i, branch in enumerate(self.branches):
                self.branches_hist[i]["datetime"] = list()
                self.branches_hist[i]["power"] = list()
                self.branches_hist[i]["losses"] = list()

        if len(self.branches) != 0:
            branches_power = get_branches_power(self)

        for i, _ in enumerate(self.branches):
            self.branches_hist[i]["datetime"].append(self.datetime)
            self.branches_hist[i]["power"].append(branches_power[i])

        for asset, attributes in self.attributes_hist.items():
            for att_name in attributes.keys():
                if att_name == "datetime":
                    self.attributes_hist[asset]["datetime"].append(
                        self.datetime + self.time_step
                    )
                elif att_name in ["electrical", "heating"]:
                    att_value = getattr(asset.power_output, att_name)
                    self.attributes_hist[asset][att_name].append(att_value)
                else:
                    att_value = getattr(asset, att_name)
                    self.attributes_hist[asset][att_name].append(att_value)

        # KPIs
        for key, value in self.tot_reward.KPIs.items():
            self.reward_hist[key].append(value)
        self.reward_hist["datetime"].append(self.datetime)

        # Environment
        if len(self.env_hist) == 0:
            self.env_hist = [dict() for _ in self.environments]
        for i, env_dict in enumerate(self.env_hist):
            for env_key in env_dict.keys():
                environment = self.environments[i]
                if env_key == "datetime":
                    env_dict[env_key].append(self.datetime)
                else:
                    env_dict[env_key].append(environment.env_values[env_key].value)

    def increase_datetime(self):
        self.utc_datetime += self.time_step
        self.datetime = self.utc_datetime.astimezone(self.local_tz)

    def set_power_zero(self):
        for asset in self.assets:
            asset.power_output = Power()
            asset.power_limit_low = Power()
            asset.power_limit_high = Power()

    def incr_time_and_environment(self):
        self.set_power_zero()

        self.increase_datetime()

        if self.datetime < self.end_time:
            self.env_simulate()

    def simulate_after_action(self):
        self.reward_simulate()
        self.incr_time_and_environment()

    def simulate(self):
        """
        Resets the microgrid and then simulates the microgrid
            from start until end (end time not included).

        """
        while self.utc_datetime < self.end_time.astimezone(
            pytz.utc
        ):  # the end time is not included
            self.management_system.simulate_step()

    def execute(self, action):
        """
        Transfers the power action from one asset to another asset if possible.

        :param action: the action which should happen
        :type action: Action
        """
        action_valid = self.check_action_valid(action)
        if action_valid:
            action.prod_asset.power_output += action.power
            action.demand_asset.power_output -= action.power
            self.actions += [action]
            difference1 = abs(
                action.prod_asset.power_output - action.prod_asset.power_limit_high
            )
            difference2 = abs(
                action.demand_asset.power_limit_low - action.demand_asset.power_output
            )
            coversion_to_mW = 0.000001
            if difference1.total_power < coversion_to_mW:
                action.prod_asset.power_output = deepcopy(
                    action.prod_asset.power_limit_high
                )
            elif difference2.total_power < coversion_to_mW:
                action.demand_asset.power_output = deepcopy(
                    action.demand_asset.power_limit_low
                )

    def reset_logs(self):
        """
        resets the logs and envs

        """
        # Log reset
        for node_hist in self.power_hist:
            for key in node_hist.keys():
                node_hist[key] = list()

        for key in self.reward_hist.keys():
            self.reward_hist[key] = list()

        for asset in self.attributes_hist.values():
            for attri in asset:
                asset[attri] = list()

        # Environments reset
        self.utc_datetime = self.start_time.astimezone(pytz.utc)
        self.datetime = self.utc_datetime.astimezone(self.local_tz)

    def save_deepcopy(self):
        """
        Save a deepcopy of the microgrid
        """
        self.microgrid_copy = deepcopy(self)

    def check_action_valid(self, action):
        """
        Check if the action satisfies the constraints of the grid

        :param action: the action which should happen
        :type action: Action
        """
        producer = action.prod_asset
        demand = action.demand_asset
        power = action.power

        pow_limit_check = (
            producer.power_limit_high >= power <= abs(demand.power_limit_low)
        )

        power_changes = {producer: power.electrical, demand: -power.electrical}

        branch_max_exceeded = False

        if len(self.branches) != 0:
            powers_to_max = get_limit_to_max_branch_power(self, power_changes)

            for power_to_max in powers_to_max:
                if power_to_max < 0:
                    branch_max_exceeded = True

        action_valid = pow_limit_check and not branch_max_exceeded

        return action_valid

    def attribute_to_log(self, asset_name, attribute_name):
        """
        Sets the attributes of an asset which should be logged while simulating

        :param asset_name: the name of which asset an attributes should be logged
        :type asset_name: str
        :param attribute_name: the name of the attributes
            of the asset that should be logged
        :type attribute_name: str

        """
        asset = None
        for node in self.nodes:
            for node_asset in node.assets:
                if node_asset.name == asset_name:
                    asset = node_asset
        if asset is None:
            return

        if asset not in self.attributes_hist.keys():
            self.attributes_hist[asset] = {"datetime": [self.datetime]}

        if attribute_name == "power_output":
            self.attributes_hist[asset]["electrical"] = [0.0]
            self.attributes_hist[asset]["heating"] = [0.0]
        else:
            att_value = getattr(asset, attribute_name)
            self.attributes_hist[asset][attribute_name] = [att_value]

    def env_to_log(self, env_name, env_key=0):
        """
        Sets the environment of a node which should be logged while simulating

        :param env_name: the name of the environment
        :type env_name: str
        :param env_key: the key of the environment
        :type env_key: int
        """
        if len(self.env_hist) == 0:
            self.env_hist = [dict() for _ in self.environments]
        if "datetime" not in self.env_hist[env_key].keys():
            self.env_hist[env_key]["datetime"] = list()
        self.env_hist[env_key][env_name] = list()

    def set_model_all_assets(self):
        for node in self.nodes:
            for asset in node.assets:
                asset.check_and_set_model()
