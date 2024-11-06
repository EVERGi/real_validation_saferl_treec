from simugrid.assets import *
from simugrid.management.manager import Manager


class RationalManager(Manager):
    """
    Specificity of this manager : Good rule-base
    """

    def __init__(self, microgrid):
        """
        Execture constructor of Manager class
        """
        super().__init__(microgrid)

    def simulate_step(self):
        """
        Function called at each simulation timestep
        """
        # Assemble all production assets together
        producers = (
            self.renewable_assets
            + self.batteries
            + self.gas_turbines
            + self.public_grid
        )
        # Include chargers as a consumer
        consumers = self.consumers + self.chargers

        # Satisfy consumer demand
        for cons in consumers:
            for pro in producers:
                self.exec_power_trans(pro, cons)

        # Charge battery with surplus and send to grid if too much
        surplus_power_assets = self.renewable_assets
        store_or_sell = self.batteries + self.public_grid
        for cons in store_or_sell:
            for pro in surplus_power_assets:
                self.exec_power_trans(pro, cons)

        # Charge from grid if required by power_limits
        for cons in self.batteries:
            for pro in self.public_grid:
                self.exec_power_trans(pro, cons, to_pow_min=True)

        # Disharge to grid if required by power_limits
        for cons in self.public_grid:
            for pro in self.batteries:
                if pro.power_limit_high.electrical == pro.power_limit_low.electrical:
                    self.exec_power_trans(
                        pro, cons, power_send=pro.power_limit_high.electrical
                    )

        # Execute the decision from Manager class (mandatory)
        super().simulate_step()
