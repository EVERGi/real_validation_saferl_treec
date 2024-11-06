import datetime


class RewardInformation:
    """
    Information necessary to calculate reward

    :ivar asset: Asset for which the reward needs to be calculated
    :type asset: Asset
    :ivar power: Power produced by the asset
    :type power: float
    :ivar environment: Environment in which the asset
    :type environment: Environment
    """

    def __init__(self, asset, power, environment):
        self.asset = asset
        self.power = power
        self.environment = environment


class Reward:
    """
    Reward base class defining the base functions and variables necessary

    :ivar reward_information: List containing all the reward information from
    assets for the current timestep
    :type reward_information: list
    :ivar KPIs: Name and value of all the KPIs to evaluate
    :type KPIs: dict
    """

    def __init__(self, list_KPI):
        """
        Constructor of Reward object

        :param list_KPI: Names of all the KPIs to evaluate
        :type list_KPI: list
        """
        self.reward_information = list()
        self.KPIs = {name: 0 for name in list_KPI}
        self.microgrid = None

    def calculate_kpi(self, time_step):
        """
        Calculate the values for the different KPI's within the function

        :param time_step: Time length of one simulation step
        :type time_step: datetime.timedelta
        """
        pass

    def reset(self):
        self.KPIs = {name: 0 for name in self.KPIs.keys()}


class DefaultReward(Reward):
    def __init__(self):
        list_KPI = ["OPEX"]
        Reward.__init__(self, list_KPI)

    def calculate_kpi(self):
        for asset in self.microgrid.assets:
            self.KPIs["OPEX"] += asset.opex_calc()
        self.reward_information = list()
