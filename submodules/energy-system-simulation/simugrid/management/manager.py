from simugrid.simulation.action import Action
from simugrid.simulation.power import Power

from simugrid.assets import *

from typing import Type


class Manager:
    def __init__(self, microgrid):
        self.microgrid = microgrid
        self.microgrid.management_system = self

        self.renewable_classes: list[Type[Asset]] = [
            SolarPv,
        ]
        self.battery_classes: list[Type[Asset]] = [Battery]
        self.renewable_assets: list[Asset] = list()
        self.batteries: list[Asset] = list()
        self.gas_turbines: list[Asset] = list()
        self.public_grid: list[Asset] = list()
        self.consumers: list[Asset] = list()
        self.chargers: list[Asset] = list()

        for node in self.microgrid.nodes:
            for asset in node.assets:
                asset_class = type(asset)

                isrenewable = any(
                    [issubclass(asset_class, i) for i in self.renewable_classes]
                )
                isbattery = any(
                    [issubclass(asset_class, i) for i in self.battery_classes]
                )

                if isbattery:
                    self.batteries += [asset]
                elif isrenewable:
                    self.renewable_assets += [asset]
                elif issubclass(asset_class, Consumer):
                    self.consumers += [asset]
                elif issubclass(asset_class, PublicGrid):
                    self.public_grid += [asset]
                elif issubclass(asset_class, Charger):
                    self.chargers += [asset]

    def simulate_step(self):
        self.microgrid.simulate_after_action()

    @classmethod
    def manager_from_json(cls, json, microgrid):
        """
        Create manager from json

        :param json: the object in json form
        :type json: dict
        :param microgrid: the microgrid
        :type microgrid: Microgrid

        :return: created manager
        :rtype: Manager
        """

        manager = cls(microgrid)

        return manager

    def exec_power_trans(
        self,
        send_asset,
        rcver_asset,
        power_type="electrical",
        to_pow_min=False,
        power_send=None,
    ):
        microgrid = self.microgrid

        max_pow_send = getattr(send_asset.power_limit_high, power_type) - getattr(
            send_asset.power_output, power_type
        )

        if to_pow_min:
            max_pow_rcv = getattr(rcver_asset.power_output, power_type) - getattr(
                rcver_asset.power_limit_high, power_type
            )
        else:
            max_pow_rcv = getattr(rcver_asset.power_output, power_type) - getattr(
                rcver_asset.power_limit_low, power_type
            )

        if power_send is None:
            power_trans = min(max_pow_send, max_pow_rcv)
        else:
            power_trans = min(max_pow_send, max_pow_rcv, power_send)

        if power_trans <= 0:
            power_trans = 0

        if power_type == "electrical":
            power = Power(electrical=power_trans)
        elif power_type == "heating":
            power = Power(heating=power_trans)

        action_obj = Action(send_asset, rcver_asset, power)
        microgrid.execute(action_obj)

        return power_trans
