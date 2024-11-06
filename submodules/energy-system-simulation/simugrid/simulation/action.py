from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from simugrid.assets import Asset
    from simugrid.simulation import Power


class Action:
    """
    A class to structure information of the energy transfer
        from an asset to another asset

    :ivar prod_asset: the asset that send energy
    :type prod_asset: Asset
    :ivar demand_asset: the asset that receives energy
    :type demand_asset: Asset
    :ivar power: the amount of energy transfer
    :type power: Power
    """

    def __init__(self, prod_asset: Asset, demand_asset: Asset, power: Power):
        self.prod_asset = prod_asset
        self.demand_asset = demand_asset
        self.power = power
